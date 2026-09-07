"""IVF-Flat (Inverted File with Flat vectors) approximate vector index from scratch."""

from typing import Iterable, Optional, Union
import numpy as np

from app.indexes.kmeans import KMeans


class IVFFlat:
    """Inverted File with Flat vectors (IVF-Flat) approximate nearest-neighbor index.

    Concept and Architecture:
    -------------------------
    1. What IVF Means:
       IVF (Inverted File) is an approximate nearest neighbor (ANN) indexing method.
       Instead of performing a linear exhaustive search against every stored vector (O(N)),
       IVF partitions the high-dimensional vector space into Voronoi cells using K-Means clustering.

    2. What an Inverted List is:
       Each cluster centroid represents a Voronoi partition. The "inverted list" for cluster `c`
       stores the row indices (positions) of all vectors whose closest centroid is `c`.

    3. How `nprobe` Works (Speed vs. Recall Trade-off):
       During search, the query vector is first compared against the `n_clusters` centroids.
       The `nprobe` closest centroids are identified, and ONLY the vectors residing in those
       `nprobe` inverted lists are evaluated for cosine similarity.
       - Smaller `nprobe` (e.g. 1-5): Fast query execution, searches a tiny fraction of data,
         lower recall.
       - Larger `nprobe` (e.g. 20-100): Higher candidate count, higher recall, slightly slower.
       - `nprobe = n_clusters`: Exhaustive search across all clusters, 100% recall.

    4. Why IVF is Approximate:
       Due to high-dimensional geometry, a query's true nearest neighbor might lie just across
       a Voronoi cell boundary in a cluster not included in the `nprobe` nearest centroids.
    """

    def __init__(
        self,
        n_clusters: int = 100,
        nprobe: int = 5,
        max_iterations: int = 20,
        random_seed: int = 42,
    ) -> None:
        """Initialize IVFFlat index hyperparameters.

        Args:
            n_clusters: Number of Voronoi partitions (centroids) to form.
            nprobe: Number of nearest cluster centroids to inspect during search.
            max_iterations: Maximum K-Means training iterations.
            random_seed: Seed for reproducible clustering.
        """
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}")
        if nprobe <= 0:
            raise ValueError(f"nprobe must be >= 1, got {nprobe}")
        if nprobe > n_clusters:
            raise ValueError(f"nprobe ({nprobe}) cannot exceed n_clusters ({n_clusters})")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        self._n_clusters = n_clusters
        self._nprobe = nprobe
        self._max_iterations = max_iterations
        self._random_seed = random_seed

        self._kmeans: Optional[KMeans] = None
        self._centroids: Optional[np.ndarray] = None  # shape (n_clusters, dimension)
        self._vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self._ids: np.ndarray = np.empty((0,), dtype=np.int64)
        self._id_to_pos: dict[int, int] = {}
        self._inverted_lists: list[list[int]] = [[] for _ in range(n_clusters)]
        self._dim: int = 0
        self._is_trained: bool = False

    @property
    def count(self) -> int:
        """Return total number of vectors stored in the index."""
        return len(self._ids)

    @property
    def dimension(self) -> int:
        """Return vector dimension."""
        return self._dim

    @property
    def n_clusters(self) -> int:
        """Return number of clusters (Voronoi partitions)."""
        return self._n_clusters

    @property
    def nprobe(self) -> int:
        """Return default number of probes during search."""
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        """Set default number of probes during search."""
        if value <= 0 or value > self._n_clusters:
            raise ValueError(
                f"nprobe must be between 1 and n_clusters ({self._n_clusters}), got {value}"
            )
        self._nprobe = value

    @property
    def centroids(self) -> np.ndarray:
        """Return cluster centroids."""
        if self._centroids is None:
            raise RuntimeError("Index is not trained yet. Call fit() first.")
        return self._centroids

    @property
    def is_trained(self) -> bool:
        """Return whether index has been trained with KMeans."""
        return self._is_trained

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """Normalize 2D vectors to unit L2 length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    def fit(self, vectors: np.ndarray, ids: np.ndarray) -> "IVFFlat":
        """Train K-Means centroids and construct inverted lists from input vectors.

        Args:
            vectors: 2D NumPy array of shape (N, D).
            ids: 1D NumPy array of shape (N,) containing unique integer IDs.

        Returns:
            self: The fitted IVFFlat instance.
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        id_arr = np.asarray(ids, dtype=np.int64)

        if vecs.ndim != 2:
            raise ValueError(f"Vectors must be 2D array of shape (N, D), got shape {vecs.shape}")
        if id_arr.ndim != 1:
            raise ValueError(f"IDs must be 1D array of shape (N,), got shape {id_arr.shape}")
        if len(vecs) != len(id_arr):
            raise ValueError(
                f"Number of vectors ({len(vecs)}) must match number of IDs ({len(id_arr)})"
            )
        if len(vecs) < self._n_clusters:
            raise ValueError(
                f"Number of vectors ({len(vecs)}) must be >= n_clusters ({self._n_clusters})"
            )
        if len(np.unique(id_arr)) != len(id_arr):
            raise ValueError("Duplicate IDs found in initialization data.")

        self._dim = vecs.shape[1]

        # 1. Train our own KMeans to find Voronoi centroids
        self._kmeans = KMeans(
            n_clusters=self._n_clusters,
            max_iterations=self._max_iterations,
            random_seed=self._random_seed,
        )
        self._kmeans.fit(vecs)
        self._centroids = self._kmeans.centroids
        labels = self._kmeans.labels_

        # 2. Store vectors and IDs
        self._vectors = np.ascontiguousarray(vecs, dtype=np.float32)
        self._ids = np.ascontiguousarray(id_arr, dtype=np.int64)

        # 3. Build _id_to_pos mapping: ID -> row position
        self._id_to_pos = {int(id_val): pos for pos, id_val in enumerate(self._ids)}

        # 4. Build inverted lists: cluster_id -> list of row positions
        self._inverted_lists = [[] for _ in range(self._n_clusters)]
        for pos, cluster_id in enumerate(labels):
            self._inverted_lists[cluster_id].append(pos)

        self._is_trained = True
        return self

    def _find_nearest_centroids(self, query: np.ndarray, nprobe: int) -> np.ndarray:
        """Find the indices of the nprobe nearest centroids to query vector using squared Euclidean distance."""
        # Query is shape (D,), centroids is shape (K, D)
        # ||q - c||^2 = ||q||^2 + ||c||^2 - 2 * (c @ q)
        c_norm = np.sum(self._centroids * self._centroids, axis=1)
        q_norm = float(np.dot(query, query))
        distances_sq = q_norm + c_norm - 2.0 * (self._centroids @ query)

        if nprobe < self._n_clusters:
            # Select top nprobe lowest distances
            partition_idx = np.argpartition(distances_sq, nprobe - 1)[:nprobe]
            sorted_nprobe = partition_idx[np.argsort(distances_sq[partition_idx])]
            return sorted_nprobe
        else:
            return np.argsort(distances_sq)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        nprobe: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for top-k approximate nearest neighbors using IVF partitioning.

        Evaluates cosine similarity only against candidate vectors stored in the
        selected `nprobe` inverted lists.

        Args:
            query: 1D NumPy array of shape (D,).
            k: Number of nearest neighbors to return.
            nprobe: Number of centroid clusters to probe. If None, uses self.nprobe.

        Returns:
            Tuple of (top_k_ids, top_k_scores) sorted in descending order of similarity.
        """
        if not self._is_trained or self.count == 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError(f"Query vector must be 1D (D,), got shape {q.shape}")
        if q.shape[0] != self._dim:
            raise ValueError(
                f"Query vector dimension ({q.shape[0]}) does not match index dimension ({self._dim})"
            )

        probes = self._nprobe if nprobe is None else nprobe
        if probes < 1 or probes > self._n_clusters:
            raise ValueError(
                f"nprobe must be between 1 and n_clusters ({self._n_clusters}), got {probes}"
            )

        if k <= 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        # Normalize query vector to unit length for cosine similarity
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # 1. Identify the nprobe nearest cluster centroids
        target_clusters = self._find_nearest_centroids(q, probes)

        # 2. Gather candidate vector positions exclusively from the selected inverted lists
        candidate_positions: list[int] = []
        for cluster_id in target_clusters:
            candidate_positions.extend(self._inverted_lists[cluster_id])

        if not candidate_positions:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        cand_pos_arr = np.array(candidate_positions, dtype=np.int64)

        # 3. Retrieve only candidate vectors and compute cosine similarity via dot product
        cand_vectors = self._vectors[cand_pos_arr]
        cand_ids = self._ids[cand_pos_arr]

        similarities = cand_vectors @ q

        # 4. Select top-k highest scoring candidates
        actual_k = min(k, len(similarities))

        if actual_k < len(similarities):
            partition_idx = np.argpartition(-similarities, actual_k - 1)[:actual_k]
            top_k_idx = partition_idx[np.argsort(-similarities[partition_idx])]
        else:
            top_k_idx = np.argsort(-similarities)

        top_k_ids = cand_ids[top_k_idx]
        top_k_scores = similarities[top_k_idx]

        return top_k_ids, top_k_scores

    def insert(
        self,
        vectors: np.ndarray,
        ids: Union[np.ndarray, int, list[int]],
    ) -> None:
        """Insert one or more vectors and assign them to existing IVF cluster centroids.

        Newly inserted vectors are normalized and assigned to their closest existing centroid
        via Euclidean distance without retraining K-Means. Duplicate IDs raise a ValueError.

        Args:
            vectors: 1D array of shape (D,) or 2D array of shape (M, D).
            ids: Single integer ID or collection of integer IDs.
        """
        if not self._is_trained:
            raise RuntimeError("IVFFlat must be trained with fit() before inserting vectors.")

        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        elif vecs.ndim != 2:
            raise ValueError(f"Vectors must be 1D (D,) or 2D (M, D), got shape {vecs.shape}")

        if isinstance(ids, (int, np.integer)):
            id_list = [int(ids)]
        elif isinstance(ids, Iterable):
            id_list = [int(i) for i in ids]
        else:
            raise TypeError(f"Unsupported ID type: {type(ids)}")

        ids_arr = np.asarray(id_list, dtype=np.int64)

        if len(vecs) != len(ids_arr):
            raise ValueError(
                f"Number of vectors ({len(vecs)}) must match number of IDs ({len(ids_arr)})"
            )
        if vecs.shape[1] != self._dim:
            raise ValueError(
                f"Vector dimension ({vecs.shape[1]}) does not match index dimension ({self._dim})"
            )

        # Check for duplicates in batch
        if len(np.unique(ids_arr)) != len(ids_arr):
            raise ValueError("Duplicate IDs found in insertion batch.")

        # Check for duplicates against existing index
        for new_id in id_list:
            if new_id in self._id_to_pos:
                raise ValueError(f"ID {new_id} already exists in the index.")

        # Normalize newly inserted vectors
        norm_vecs = self._normalize_vectors(vecs)

        # Find closest existing centroid for each new vector
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * (x @ c.T)
        c_norm = np.sum(self._centroids * self._centroids, axis=1)
        x_norm = np.sum(norm_vecs * norm_vecs, axis=1, keepdims=True)
        distances_sq = x_norm + c_norm - 2.0 * (norm_vecs @ self._centroids.T)
        assigned_clusters = np.argmin(distances_sq, axis=1)

        start_pos = len(self._vectors)
        self._vectors = np.vstack([self._vectors, norm_vecs])
        self._ids = np.concatenate([self._ids, ids_arr])

        for offset, (new_id, cluster_id) in enumerate(zip(id_list, assigned_clusters)):
            current_pos = start_pos + offset
            self._id_to_pos[new_id] = current_pos
            self._inverted_lists[cluster_id].append(current_pos)

    def delete(self, ids: Union[np.ndarray, int, list[int], set[int]]) -> None:
        """Delete vectors corresponding to provided IDs and rebuild inverted lists.

        Removes requested vectors, compacts internal arrays, rebuilds _id_to_pos,
        and cleanly reconstructs inverted lists from the remaining vectors.

        Args:
            ids: Single integer ID or collection of integer IDs to delete.
        """
        if not self._is_trained or self.count == 0:
            return

        if isinstance(ids, (int, np.integer)):
            target_ids = {int(ids)}
        elif isinstance(ids, Iterable):
            target_ids = {int(i) for i in ids}
        else:
            raise TypeError(f"Unsupported ID type: {type(ids)}")

        existing_to_delete = target_ids.intersection(set(self._id_to_pos.keys()))
        if not existing_to_delete:
            return

        # Create keep mask for arrays
        keep_mask = np.isin(self._ids, list(existing_to_delete), invert=True)
        self._vectors = np.ascontiguousarray(self._vectors[keep_mask])
        self._ids = np.ascontiguousarray(self._ids[keep_mask])

        # Rebuild _id_to_pos
        self._id_to_pos = {int(id_val): pos for pos, id_val in enumerate(self._ids)}

        # Rebuild ALL inverted lists from remaining vectors without retraining KMeans
        self._inverted_lists = [[] for _ in range(self._n_clusters)]
        if len(self._vectors) > 0:
            c_norm = np.sum(self._centroids * self._centroids, axis=1)
            x_norm = np.sum(self._vectors * self._vectors, axis=1, keepdims=True)
            distances_sq = x_norm + c_norm - 2.0 * (self._vectors @ self._centroids.T)
            labels = np.argmin(distances_sq, axis=1)

            for pos, cluster_id in enumerate(labels):
                self._inverted_lists[cluster_id].append(pos)
