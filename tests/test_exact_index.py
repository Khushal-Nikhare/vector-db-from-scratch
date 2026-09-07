"""Tests for Step 2 ExactIndex implementation."""

from pathlib import Path
import numpy as np
import pytest

from app.indexes.exact_index import ExactIndex


@pytest.fixture
def dataset_paths():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    vectors_path = data_dir / "vectors.npy"
    ids_path = data_dir / "ids.npy"
    return vectors_path, ids_path


@pytest.fixture
def loaded_dataset(dataset_paths):
    vectors_path, ids_path = dataset_paths
    vectors = np.load(vectors_path)
    ids = np.load(ids_path)
    return vectors, ids


def test_initialization_with_50k_dataset(loaded_dataset):
    """Test 1: Initialization works with the generated 50,000-vector dataset."""
    vectors, ids = loaded_dataset
    index = ExactIndex(vectors, ids)

    assert index.count == 50000
    assert index.dimension == 128


def test_search_returns_exact_k_results(loaded_dataset):
    """Test 2: Search returns exactly k results."""
    vectors, ids = loaded_dataset
    index = ExactIndex(vectors, ids)

    query = vectors[0]
    for k in [1, 5, 10, 50]:
        res_ids, res_scores = index.search(query, k=k)
        assert len(res_ids) == k
        assert len(res_scores) == k


def test_results_ordered_by_descending_similarity(loaded_dataset):
    """Test 3: Results are ordered by descending similarity."""
    vectors, ids = loaded_dataset
    index = ExactIndex(vectors, ids)

    query = vectors[100]
    res_ids, res_scores = index.search(query, k=20)

    # Verify descending order
    for i in range(len(res_scores) - 1):
        assert res_scores[i] >= res_scores[i + 1]


def test_returned_ids_correspond_to_stored_vectors(loaded_dataset):
    """Test 4: Returned IDs correspond to the stored vectors."""
    vectors, ids = loaded_dataset
    index = ExactIndex(vectors, ids)

    # Searching with an exact stored vector should return itself as top match (id == 42, score ~ 1.0)
    target_idx = 42
    query = vectors[target_idx]
    target_id = ids[target_idx]

    res_ids, res_scores = index.search(query, k=1)
    assert res_ids[0] == target_id
    assert pytest.approx(res_scores[0], rel=1e-5) == 1.0


def test_exact_similarity_score_calculation(loaded_dataset):
    """Test 5: The exact similarity score is correct by independently calculating the dot product."""
    vectors, ids = loaded_dataset
    index = ExactIndex(vectors, ids)

    rng = np.random.default_rng(999)
    raw_query = rng.standard_normal(128).astype(np.float32)
    normalized_query = raw_query / np.linalg.norm(raw_query)

    res_ids, res_scores = index.search(raw_query, k=10)

    for result_id, score in zip(res_ids, res_scores):
        # Find vector matching result_id
        vec_idx = np.where(ids == result_id)[0][0]
        stored_vec = vectors[vec_idx]
        expected_score = float(np.dot(stored_vec, normalized_query))
        assert pytest.approx(score, rel=1e-5) == expected_score


def test_k_larger_than_dataset():
    """Test 6: k larger than the dataset size is handled correctly."""
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.array([10, 20], dtype=np.int64)
    index = ExactIndex(vecs, ids)

    res_ids, res_scores = index.search(np.array([1.0, 0.0], dtype=np.float32), k=100)
    assert len(res_ids) == 2
    assert len(res_scores) == 2
    assert res_ids[0] == 10
    assert res_ids[1] == 20


def test_insert_single_and_batch():
    """Test 7: Insert works for single vectors and batches."""
    index = ExactIndex()
    assert index.count == 0

    # Insert single vector
    v1 = np.array([3.0, 4.0], dtype=np.float32)  # Unnormalized, norm is 5.0
    index.insert(v1, ids=1)
    assert index.count == 1
    assert index.dimension == 2

    res_ids, res_scores = index.search(np.array([0.6, 0.8], dtype=np.float32), k=1)
    assert res_ids[0] == 1
    assert pytest.approx(res_scores[0], rel=1e-5) == 1.0

    # Insert batch
    v_batch = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index.insert(v_batch, ids=[2, 3])
    assert index.count == 3


def test_duplicate_ids_rejected():
    """Test 8: Duplicate IDs are rejected."""
    vecs = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    ids = np.array([1, 2], dtype=np.int64)
    index = ExactIndex(vecs, ids)

    # Duplicate in initialization
    with pytest.raises(ValueError, match="Duplicate IDs"):
        ExactIndex(vecs, np.array([1, 1], dtype=np.int64))

    # Duplicate in insert batch
    with pytest.raises(ValueError, match="Duplicate IDs"):
        index.insert(np.array([[1.0, 0.0], [0.0, 1.0]]), ids=[3, 3])

    # Duplicate existing ID
    with pytest.raises(ValueError, match="already exists"):
        index.insert(np.array([0.5, 0.5]), ids=1)


def test_delete_functionality():
    """Test 9: Delete works and deleted IDs are no longer returned."""
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]], dtype=np.float32)
    ids = np.array([101, 102, 103], dtype=np.int64)
    index = ExactIndex(vecs, ids)
    assert index.count == 3

    # Delete non-existent ID (should not fail)
    index.delete(999)
    assert index.count == 3

    # Delete single existing ID
    index.delete(102)
    assert index.count == 2
    res_ids, _ = index.search(np.array([0.0, 1.0], dtype=np.float32), k=5)
    assert 102 not in res_ids

    # Delete multiple IDs
    index.delete([101, 103])
    assert index.count == 0


def test_query_dimension_mismatch_raises_error():
    """Test 10: Searching with an incorrectly sized query raises a clear ValueError."""
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    ids = np.array([1, 2], dtype=np.int64)
    index = ExactIndex(vecs, ids)

    with pytest.raises(ValueError, match="dimension"):
        index.search(np.array([1.0, 0.0], dtype=np.float32))

    with pytest.raises(ValueError, match="dimension"):
        index.search(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
