"""Tests for Step 4 IVFFlat approximate index implementation."""

from pathlib import Path
import numpy as np
import pytest

from app.indexes.ivf_flat import IVFFlat
from app.indexes.exact_index import ExactIndex


@pytest.fixture
def dataset_50k():
    project_root = Path(__file__).resolve().parent.parent
    vectors_path = project_root / "data" / "vectors.npy"
    ids_path = project_root / "data" / "ids.npy"
    if not vectors_path.exists() or not ids_path.exists():
        pytest.skip("Dataset files not found in data/.")
    return np.load(vectors_path), np.load(ids_path)


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(42)
    # 5 clusters with 20 vectors each = 100 vectors, dim 8
    centers = rng.standard_normal((5, 8)).astype(np.float32)
    vecs = []
    for c in centers:
        noise = rng.normal(0, 0.05, size=(20, 8)).astype(np.float32)
        v = c + noise
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        vecs.append(v)
    vectors = np.vstack(vecs).astype(np.float32)
    ids = np.arange(100, dtype=np.int64)
    return vectors, ids


def test_ivf_fit_50k(dataset_50k):
    """Test 1: IVF can fit the 50,000-vector dataset."""
    vectors, ids = dataset_50k
    ivf = IVFFlat(n_clusters=100, nprobe=5, max_iterations=5, random_seed=42)
    ivf.fit(vectors, ids)

    assert ivf.count == 50000
    assert ivf.dimension == 128
    assert ivf.is_trained is True


def test_exactly_100_clusters_created(dataset_50k):
    """Test 2: Exactly 100 clusters are created."""
    vectors, ids = dataset_50k
    ivf = IVFFlat(n_clusters=100, max_iterations=5, random_seed=42).fit(vectors, ids)

    assert len(ivf._inverted_lists) == 100
    assert ivf.centroids.shape == (100, 128)
    assert ivf.n_clusters == 100


def test_every_vector_belongs_to_one_cluster(small_dataset):
    """Test 3: Every vector belongs to exactly one cluster."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, max_iterations=10, random_seed=42).fit(vectors, ids)

    all_positions = []
    for cluster_id, pos_list in enumerate(ivf._inverted_lists):
        all_positions.extend(pos_list)

    assert len(all_positions) == len(vectors)
    assert len(set(all_positions)) == len(vectors)  # No duplicates across clusters


def test_inverted_lists_collectively_contain_all_vectors(small_dataset):
    """Test 4: Inverted lists collectively contain all vector positions."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, max_iterations=10, random_seed=42).fit(vectors, ids)

    collected_positions = set()
    for pos_list in ivf._inverted_lists:
        collected_positions.update(pos_list)

    assert collected_positions == set(range(len(vectors)))


def test_search_returns_at_most_k(small_dataset):
    """Test 5: Search returns at most k results."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, nprobe=2, max_iterations=10, random_seed=42).fit(vectors, ids)

    query = vectors[0]
    for k in [1, 5, 10, 50, 200]:
        res_ids, res_scores = ivf.search(query, k=k)
        assert len(res_ids) <= k
        assert len(res_scores) == len(res_ids)


def test_search_results_ordered_by_descending_similarity(small_dataset):
    """Test 6: Search results are ordered by descending similarity."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, nprobe=3, max_iterations=10, random_seed=42).fit(vectors, ids)

    query = vectors[10]
    res_ids, res_scores = ivf.search(query, k=15)

    for i in range(len(res_scores) - 1):
        assert res_scores[i] >= res_scores[i + 1]


def test_search_uses_only_candidate_clusters(small_dataset):
    """Test 7: Search actually uses only vectors from selected nprobe clusters."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, nprobe=1, max_iterations=10, random_seed=42).fit(vectors, ids)

    query = vectors[0]
    # Identify the 1 nearest centroid
    target_cluster = ivf._find_nearest_centroids(query, nprobe=1)[0]
    allowed_positions = set(ivf._inverted_lists[target_cluster])
    allowed_ids = set(ivf._ids[list(allowed_positions)])

    res_ids, res_scores = ivf.search(query, k=100, nprobe=1)
    for result_id in res_ids:
        assert result_id in allowed_ids


def test_nprobe_1_works(small_dataset):
    """Test 8: nprobe=1 works."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, nprobe=1, max_iterations=10, random_seed=42).fit(vectors, ids)

    res_ids, res_scores = ivf.search(vectors[5], k=5, nprobe=1)
    assert len(res_ids) > 0
    assert len(res_scores) == len(res_ids)


def test_nprobe_5_works(dataset_50k):
    """Test 9: nprobe=5 works on 50K dataset."""
    vectors, ids = dataset_50k
    ivf = IVFFlat(n_clusters=100, nprobe=5, max_iterations=5, random_seed=42).fit(vectors, ids)

    res_ids, res_scores = ivf.search(vectors[0], k=10, nprobe=5)
    assert len(res_ids) == 10
    assert len(res_scores) == 10


def test_nprobe_n_clusters_exhaustive(small_dataset):
    """Test 10: nprobe=n_clusters works and matches exact search results."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, nprobe=5, max_iterations=10, random_seed=42).fit(vectors, ids)
    exact = ExactIndex(vectors, ids)

    query = vectors[7]
    ivf_ids, ivf_scores = ivf.search(query, k=10, nprobe=5)
    exact_ids, exact_scores = exact.search(query, k=10)

    assert np.array_equal(ivf_ids, exact_ids)
    assert np.allclose(ivf_scores, exact_scores, atol=1e-5)


def test_invalid_nprobe_raises_error(small_dataset):
    """Test 11: Invalid nprobe values raise ValueError."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, nprobe=2, max_iterations=10, random_seed=42).fit(vectors, ids)

    with pytest.raises(ValueError, match="nprobe"):
        ivf.search(vectors[0], nprobe=0)

    with pytest.raises(ValueError, match="nprobe"):
        ivf.search(vectors[0], nprobe=6)  # greater than n_clusters=5

    with pytest.raises(ValueError, match="nprobe"):
        ivf.nprobe = 10


def test_insert_increases_count(small_dataset):
    """Test 12: Insert works and increases count."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, max_iterations=10, random_seed=42).fit(vectors, ids)
    initial_count = ivf.count

    new_vec = np.ones((1, 8), dtype=np.float32)
    ivf.insert(new_vec, ids=999)

    assert ivf.count == initial_count + 1
    assert 999 in ivf._id_to_pos

    # Confirm it is inside one of the inverted lists
    new_pos = ivf._id_to_pos[999]
    found = any(new_pos in pos_list for pos_list in ivf._inverted_lists)
    assert found


def test_duplicate_ids_rejected(small_dataset):
    """Test 13: Duplicate IDs are rejected during insert."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, max_iterations=10, random_seed=42).fit(vectors, ids)

    # Duplicate existing ID
    with pytest.raises(ValueError, match="already exists"):
        ivf.insert(np.ones(8), ids=0)

    # Duplicate in batch
    with pytest.raises(ValueError, match="Duplicate IDs"):
        ivf.insert(np.ones((2, 8)), ids=[500, 500])


def test_delete_decreases_count(small_dataset):
    """Test 14: Delete works and decreases count."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, max_iterations=10, random_seed=42).fit(vectors, ids)
    initial_count = ivf.count

    ivf.delete(ids=0)
    assert ivf.count == initial_count - 1
    assert 0 not in ivf._id_to_pos


def test_deleted_ids_cannot_be_returned(small_dataset):
    """Test 15: Deleted IDs cannot be returned during search."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, nprobe=5, max_iterations=10, random_seed=42).fit(vectors, ids)

    target_id = ids[0]
    query = vectors[0]

    # Verify target was originally returned
    res_ids, _ = ivf.search(query, k=1, nprobe=5)
    assert res_ids[0] == target_id

    # Delete target ID
    ivf.delete(target_id)

    # Verify target is never returned after deletion
    res_ids_after, _ = ivf.search(query, k=10, nprobe=5)
    assert target_id not in res_ids_after


def test_query_dimension_mismatch(small_dataset):
    """Test 16: Query dimension mismatch raises ValueError."""
    vectors, ids = small_dataset
    ivf = IVFFlat(n_clusters=5, max_iterations=10, random_seed=42).fit(vectors, ids)

    with pytest.raises(ValueError, match="dimension"):
        ivf.search(np.array([1.0, 2.0]))  # 2D instead of 8D
