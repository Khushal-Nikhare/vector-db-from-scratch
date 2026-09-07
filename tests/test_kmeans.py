"""Tests for Step 3 KMeans clustering implementation."""

from pathlib import Path
import numpy as np
import pytest

from app.indexes.kmeans import KMeans


@pytest.fixture
def small_clustered_dataset():
    """Create a small 2D dataset with 3 clear clusters."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[-5.0, -5.0], scale=0.5, size=(30, 2))
    c2 = rng.normal(loc=[0.0, 5.0], scale=0.5, size=(30, 2))
    c3 = rng.normal(loc=[5.0, -5.0], scale=0.5, size=(30, 2))
    vectors = np.vstack([c1, c2, c3]).astype(np.float32)
    return vectors


@pytest.fixture
def loaded_50k_dataset():
    """Load the 50K x 128 dataset generated in Step 1."""
    project_root = Path(__file__).resolve().parent.parent
    vectors_path = project_root / "data" / "vectors.npy"
    if not vectors_path.exists():
        pytest.skip("Dataset file data/vectors.npy not found.")
    return np.load(vectors_path)


def test_fit_small_dataset(small_clustered_dataset):
    """Test 1: KMeans can fit a small synthetic clustered dataset."""
    kmeans = KMeans(n_clusters=3, max_iterations=20, random_seed=42)
    kmeans.fit(small_clustered_dataset)

    assert kmeans.centroids is not None
    assert kmeans.n_iter_ >= 1
    assert kmeans.dimension == 2


def test_centroid_shape(small_clustered_dataset):
    """Test 2: Resulting centroid shape is (n_clusters, dimension)."""
    kmeans = KMeans(n_clusters=3, max_iterations=20, random_seed=42)
    kmeans.fit(small_clustered_dataset)

    assert kmeans.centroids.shape == (3, 2)
    assert kmeans.centroids.dtype == np.float32


def test_labels_length(small_clustered_dataset):
    """Test 3: labels_ has one label for every input vector."""
    kmeans = KMeans(n_clusters=3, max_iterations=20, random_seed=42)
    kmeans.fit(small_clustered_dataset)

    assert kmeans.labels_.shape == (len(small_clustered_dataset),)
    assert kmeans.labels_.dtype == np.int64


def test_labels_valid_indices(small_clustered_dataset):
    """Test 4: labels are valid cluster indices in [0, n_clusters - 1]."""
    kmeans = KMeans(n_clusters=3, max_iterations=20, random_seed=42)
    kmeans.fit(small_clustered_dataset)

    unique_labels = np.unique(kmeans.labels_)
    for label in unique_labels:
        assert 0 <= label < 3


def test_predict_shapes_and_types(small_clustered_dataset):
    """Test 5: predict() returns correct-shaped integer cluster IDs."""
    kmeans = KMeans(n_clusters=3, max_iterations=20, random_seed=42)
    kmeans.fit(small_clustered_dataset)

    # 1D single vector shape (D,) -> shape (1,)
    single_query = small_clustered_dataset[0]
    pred_single = kmeans.predict(single_query)
    assert isinstance(pred_single, np.ndarray)
    assert pred_single.shape == (1,)
    assert np.issubdtype(pred_single.dtype, np.integer)

    # 2D batch shape (M, D) -> shape (M,)
    batch_query = small_clustered_dataset[:5]
    pred_batch = kmeans.predict(batch_query)
    assert isinstance(pred_batch, np.ndarray)
    assert pred_batch.shape == (5,)
    assert np.issubdtype(pred_batch.dtype, np.integer)


def test_predict_before_fit_raises_error():
    """Test 6: predict() cannot be called before fit()."""
    kmeans = KMeans(n_clusters=3)
    with pytest.raises(RuntimeError, match="not been fitted"):
        kmeans.predict(np.array([1.0, 2.0]))


def test_invalid_n_clusters(small_clustered_dataset):
    """Test 7: Invalid n_clusters raises ValueError."""
    with pytest.raises(ValueError, match="n_clusters"):
        KMeans(n_clusters=0)

    with pytest.raises(ValueError, match="n_clusters"):
        KMeans(n_clusters=-5)

    kmeans = KMeans(n_clusters=500)
    with pytest.raises(ValueError, match="must be >= n_clusters"):
        kmeans.fit(small_clustered_dataset)  # only 90 samples


def test_invalid_max_iterations():
    """Test 8: Invalid max_iterations raises ValueError."""
    with pytest.raises(ValueError, match="max_iterations"):
        KMeans(n_clusters=3, max_iterations=0)

    with pytest.raises(ValueError, match="max_iterations"):
        KMeans(n_clusters=3, max_iterations=-10)


def test_reproducibility(small_clustered_dataset):
    """Test 9: Repeated runs with same random seed produce identical centroids/labels."""
    km1 = KMeans(n_clusters=3, max_iterations=20, random_seed=42).fit(small_clustered_dataset)
    km2 = KMeans(n_clusters=3, max_iterations=20, random_seed=42).fit(small_clustered_dataset)

    assert np.allclose(km1.centroids, km2.centroids)
    assert np.array_equal(km1.labels_, km2.labels_)
    assert km1.n_iter_ == km2.n_iter_


def test_fit_50k_dataset(loaded_50k_dataset):
    """Test 10: KMeans can fit the actual 50,000 x 128 dataset with 100 clusters."""
    # Use max_iterations=5 for rapid test verification
    kmeans = KMeans(n_clusters=100, max_iterations=5, random_seed=42)
    kmeans.fit(loaded_50k_dataset)

    assert kmeans.centroids.shape == (100, 128)
    assert kmeans.labels_.shape == (50000,)
    assert kmeans.n_iter_ >= 1
