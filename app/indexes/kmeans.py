"""K-Means clustering from scratch using NumPy."""

from typing import Optional
import numpy as np


class KMeans:
    """K-Means clustering implementation using vectorized squared Euclidean distance."""

    def __init__(
        self,
        n_clusters: int,
        max_iterations: int = 20,
        random_seed: int = 42,
    ) -> None:
        """Initialize KMeans model hyperparameters.

        Args:
            n_clusters: Number of clusters to form. Must be greater than 0.
            max_iterations: Maximum number of iterations for the algorithm. Must be greater than 0.
            random_seed: Seed for reproducible initialization and empty-cluster fallback.
        """
        if n_clusters <= 0:
            raise ValueError(f"n_clusters must be positive, got {n_clusters}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        self._n_clusters = n_clusters
        self._max_iterations = max_iterations
        self._random_seed = random_seed

        self._centroids: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._n_iter: int = 0
        self._dimension: int = 0

    @property
    def n_clusters(self) -> int:
        """Return number of clusters."""
        return self._n_clusters

    @property
    def max_iterations(self) -> int:
        """Return maximum number of iterations."""
        return self._max_iterations

    @property
    def random_seed(self) -> int:
        """Return random seed."""
        return self._random_seed

    @property
    def dimension(self) -> int:
        """Return dimensionality of fitted vectors."""
        return self._dimension

    @property
    def centroids(self) -> np.ndarray:
        """Return cluster centroids array of shape (n_clusters, dimension)."""
        if self._centroids is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._centroids

    @property
    def labels_(self) -> np.ndarray:
        """Return labels assigned to training vectors of shape (N,)."""
        if self._labels is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self._labels

    @property
    def n_iter_(self) -> int:
        """Return number of iterations executed during fit()."""
        return self._n_iter

    def _compute_distances_sq(
        self, vectors: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Compute squared Euclidean distances using vectorized matrix expansion:
        ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * (x . c).
        """
        x_norm = np.sum(vectors * vectors, axis=1, keepdims=True)
        c_norm = np.sum(centroids * centroids, axis=1)
        distances_sq = x_norm + c_norm - 2.0 * (vectors @ centroids.T)
        return distances_sq

    def fit(self, vectors: np.ndarray) -> "KMeans":
        """Compute K-Means clustering.

        Args:
            vectors: 2D NumPy array of shape (N, D).

        Returns:
            self: The fitted KMeans instance.
        """
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2:
            raise ValueError(f"Vectors must be 2D array of shape (N, D), got shape {vecs.shape}")

        n_samples, n_features = vecs.shape
        if n_samples < self._n_clusters:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= n_clusters ({self._n_clusters})"
            )

        self._dimension = n_features
        rng = np.random.default_rng(self._random_seed)

        # 1. Initialize centroids by selecting n_clusters unique vectors from input
        initial_indices = rng.choice(n_samples, size=self._n_clusters, replace=False)
        centroids = vecs[initial_indices].copy()

        labels = np.zeros(n_samples, dtype=np.int64)
        self._n_iter = 0

        # Precompute vector squared norms since they remain constant during iterations
        x_norm = np.sum(vecs * vecs, axis=1, keepdims=True)

        for iteration in range(1, self._max_iterations + 1):
            self._n_iter = iteration

            # a. Calculate squared Euclidean distances: ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * (x . c)
            c_norm = np.sum(centroids * centroids, axis=1)
            distances_sq = x_norm + c_norm - 2.0 * (vecs @ centroids.T)

            # b. Assign each vector to nearest centroid
            new_labels = np.argmin(distances_sq, axis=1)

            # e. Stop when assignments no longer change
            if iteration > 1 and np.array_equal(new_labels, labels):
                labels = new_labels
                break

            labels = new_labels

            # c. Recalculate each centroid as mean of vectors assigned to it
            new_centroids = np.empty_like(centroids)
            for k in range(self._n_clusters):
                cluster_mask = labels == k
                if np.any(cluster_mask):
                    new_centroids[k] = np.mean(vecs[cluster_mask], axis=0)
                else:
                    # d. Handle empty clusters by reinitializing to a randomly chosen existing sample
                    random_idx = rng.choice(n_samples)
                    new_centroids[k] = vecs[random_idx].copy()

            centroids = new_centroids

        self._centroids = centroids.astype(np.float32)
        self._labels = labels.astype(np.int64)

        return self

    def predict(self, vectors: np.ndarray) -> np.ndarray:
        """Predict the closest cluster index for each input vector.

        Args:
            vectors: 1D array of shape (D,) or 2D array of shape (M, D).

        Returns:
            1D integer array of shape (1,) if input is (D,), or (M,) if input is (M, D).
        """
        if self._centroids is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        vecs = np.asarray(vectors, dtype=np.float32)
        single_vector = False

        if vecs.ndim == 1:
            if vecs.shape[0] != self._dimension:
                raise ValueError(
                    f"Vector dimension {vecs.shape[0]} does not match model dimension {self._dimension}"
                )
            vecs = vecs.reshape(1, -1)
            single_vector = True
        elif vecs.ndim == 2:
            if vecs.shape[1] != self._dimension:
                raise ValueError(
                    f"Vector dimension {vecs.shape[1]} does not match model dimension {self._dimension}"
                )
        else:
            raise ValueError(f"Vectors must be 1D (D,) or 2D (M, D), got shape {vecs.shape}")

        # Compute distances to learned centroids
        distances_sq = self._compute_distances_sq(vecs, self._centroids)
        cluster_ids = np.argmin(distances_sq, axis=1).astype(np.int64)

        if single_vector:
            return cluster_ids  # shape (1,)
        return cluster_ids  # shape (M,)
