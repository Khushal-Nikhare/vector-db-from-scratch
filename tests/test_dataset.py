"""Unit tests for Step 1 dataset generation."""

from pathlib import Path
import json
import numpy as np
import pytest

from scripts.generate_data import (
    NUM_VECTORS,
    DIMENSION,
    NUM_CLUSTERS,
    generate_dataset,
    validate_dataset,
)


def test_generate_dataset_shapes_and_types():
    vectors, ids, metadata = generate_dataset()

    assert vectors.shape == (NUM_VECTORS, DIMENSION)
    assert ids.shape == (NUM_VECTORS,)
    assert vectors.dtype == np.float32
    assert ids.dtype == np.int64
    assert metadata["number_of_vectors"] == NUM_VECTORS
    assert metadata["dimension"] == DIMENSION
    assert metadata["number_of_clusters"] == NUM_CLUSTERS
    assert metadata["vectors_per_cluster"] == 500
    assert metadata["normalized"] is True

    # Validate norms are unit length
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_reproducibility():
    v1, ids1, _ = generate_dataset(random_seed=42)
    v2, ids2, _ = generate_dataset(random_seed=42)

    assert np.array_equal(v1, v2)
    assert np.array_equal(ids1, ids2)


def test_saved_files_exist():
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"

    assert (data_dir / "vectors.npy").exists()
    assert (data_dir / "ids.npy").exists()
    assert (data_dir / "metadata.json").exists()

    vectors = np.load(data_dir / "vectors.npy")
    ids = np.load(data_dir / "ids.npy")
    with open(data_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    validate_dataset(vectors, ids)
    assert meta["number_of_vectors"] == NUM_VECTORS
