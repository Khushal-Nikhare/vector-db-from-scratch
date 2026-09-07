"""Exact brute-force vector index implementation using NumPy."""

from typing import Iterable, Union
import numpy as np


class ExactIndex:
    """Brute-force exact nearest neighbor vector index using cosine similarity."""

    def __init__(
        self,
        vectors: np.ndarray | None = None,
        ids: np.ndarray | None = None,
    ) -> None:
        """Initialize the exact index with optional initial vectors and IDs.

        Args:
            vectors: Optional 2D NumPy array of shape (N, D). Pre-normalized dataset
                     vectors are stored directly as float32 without redundant renormalization.
            ids: Optional 1D NumPy array of shape (N,) containing unique integer IDs.
        """
        self._dim: int = 0
        self._id_set: set[int] = set()

        if vectors is not None or ids is not None:
            if vectors is None or ids is None:
                raise ValueError("Both vectors and ids must be provided together.")

            vectors_arr = np.asarray(vectors, dtype=np.float32)
            ids_arr = np.asarray(ids, dtype=np.int64)

            if vectors_arr.ndim != 2:
                raise ValueError(
                    f"Vectors must be a 2D array of shape (N, D), got shape {vectors_arr.shape}"
                )
            if ids_arr.ndim != 1:
                raise ValueError(
                    f"IDs must be a 1D array of shape (N,), got shape {ids_arr.shape}"
                )
            if len(vectors_arr) != len(ids_arr):
                raise ValueError(
                    f"Number of vectors ({len(vectors_arr)}) must match number of IDs ({len(ids_arr)})"
                )

            # Check for duplicate IDs
            if len(np.unique(ids_arr)) != len(ids_arr):
                raise ValueError("Duplicate IDs found in initialization data.")

            self._vectors: np.ndarray = np.ascontiguousarray(vectors_arr, dtype=np.float32)
            self._ids: np.ndarray = np.ascontiguousarray(ids_arr, dtype=np.int64)
            self._dim = self._vectors.shape[1] if len(self._vectors) > 0 else 0
            self._id_set = set(self._ids.tolist())
        else:
            self._vectors = np.empty((0, 0), dtype=np.float32)
            self._ids = np.empty((0,), dtype=np.int64)

    @property
    def count(self) -> int:
        """Return the total number of vectors currently stored in the index."""
        return len(self._ids)

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._dim

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        """Normalize 2D vectors to unit L2 length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    def insert(
        self,
        vectors: np.ndarray,
        ids: Union[np.ndarray, int, list[int]],
    ) -> None:
        """Insert one or more vectors and their corresponding IDs.

        Newly inserted vectors are automatically normalized to unit length.
        Duplicate IDs are rejected with a ValueError.

        Args:
            vectors: 1D array of shape (D,) or 2D array of shape (M, D).
            ids: Single integer ID or 1D array/list of M integer IDs.
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        elif vecs.ndim != 2:
            raise ValueError(
                f"Vectors must be 1D (D,) or 2D (M, D), got shape {vecs.shape}"
            )

        if isinstance(ids, (int, np.integer)):
            id_list = [int(ids)]
        elif isinstance(ids, Iterable):
            id_list = [int(i) for i in ids]
        else:
            raise TypeError(f"Unsupported ID type: {type(ids)}")

        ids_arr = np.asarray(id_list, dtype=np.int64)

        if len(vecs) != len(ids_arr):
            raise ValueError(
                f"Number of vectors ({len(vecs)}) does not match number of IDs ({len(ids_arr)})"
            )

        # Validate vector dimension
        if self.count > 0:
            if vecs.shape[1] != self._dim:
                raise ValueError(
                    f"Vector dimension {vecs.shape[1]} does not match index dimension {self._dim}"
                )
        else:
            self._dim = vecs.shape[1]

        # Check for duplicate IDs in the insertion batch
        if len(np.unique(ids_arr)) != len(ids_arr):
            raise ValueError("Duplicate IDs found in the insertion batch.")

        # Check for duplicate IDs against existing index
        for new_id in id_list:
            if new_id in self._id_set:
                raise ValueError(f"ID {new_id} already exists in the index.")

        # Normalize newly inserted vectors
        norm_vecs = self._normalize_vectors(vecs)

        if self.count == 0:
            self._vectors = np.ascontiguousarray(norm_vecs, dtype=np.float32)
            self._ids = np.ascontiguousarray(ids_arr, dtype=np.int64)
        else:
            self._vectors = np.vstack([self._vectors, norm_vecs])
            self._ids = np.concatenate([self._ids, ids_arr])

        self._id_set.update(id_list)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search for top-k most similar vectors to a query using cosine similarity.

        Args:
            query: 1D NumPy array of shape (D,).
            k: Number of nearest neighbors to return.

        Returns:
            Tuple (top_k_ids, top_k_scores) sorted in descending order of similarity.
        """
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError(f"Query vector must be 1D (D,), got shape {q.shape}")

        if self.count == 0 or k <= 0:
            return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

        if q.shape[0] != self._dim:
            raise ValueError(
                f"Query vector dimension ({q.shape[0]}) does not match index dimension ({self._dim})"
            )

        # Normalize query vector to unit length
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Core cosine similarity calculation via dot product with normalized vectors
        similarities = self._vectors @ q

        actual_k = min(k, self.count)

        if actual_k < self.count:
            # Use argpartition for O(N) top-k selection
            partition_idx = np.argpartition(-similarities, actual_k - 1)[:actual_k]
            # Sort only top-k candidates in descending order
            top_k_indices = partition_idx[np.argsort(-similarities[partition_idx])]
        else:
            top_k_indices = np.argsort(-similarities)

        top_k_ids = self._ids[top_k_indices]
        top_k_scores = similarities[top_k_indices]

        return top_k_ids, top_k_scores

    def delete(self, ids: Union[np.ndarray, int, list[int], set[int]]) -> None:
        """Delete vectors corresponding to provided IDs.

        Non-existent IDs are ignored cleanly.

        Args:
            ids: Single integer ID or collection of integer IDs to delete.
        """
        if isinstance(ids, (int, np.integer)):
            target_ids = {int(ids)}
        elif isinstance(ids, Iterable):
            target_ids = {int(i) for i in ids}
        else:
            raise TypeError(f"Unsupported ID type: {type(ids)}")

        if self.count == 0 or not target_ids:
            return

        # Find existing IDs to delete
        existing_to_delete = target_ids.intersection(self._id_set)
        if not existing_to_delete:
            return

        # Create keep mask
        keep_mask = np.isin(self._ids, list(existing_to_delete), invert=True)
        self._vectors = np.ascontiguousarray(self._vectors[keep_mask])
        self._ids = np.ascontiguousarray(self._ids[keep_mask])
        self._id_set.difference_update(existing_to_delete)
