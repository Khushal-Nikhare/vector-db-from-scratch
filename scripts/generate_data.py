"""Synthetic clustered vector dataset generator for Vector DB benchmarking."""

import json
from pathlib import Path
import numpy as np

# Dataset configuration constants
NUM_VECTORS = 50_000
DIMENSION = 128
NUM_CLUSTERS = 100
VECTORS_PER_CLUSTER = NUM_VECTORS // NUM_CLUSTERS  # Exactly 500
RANDOM_SEED = 42
CLUSTER_STD = 0.1


def generate_dataset(
    num_vectors: int = NUM_VECTORS,
    dimension: int = DIMENSION,
    num_clusters: int = NUM_CLUSTERS,
    random_seed: int = RANDOM_SEED,
    cluster_std: float = CLUSTER_STD,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate a synthetic clustered dataset of unit-normalized vectors.

    Args:
        num_vectors: Total number of vectors to generate (must be divisible by num_clusters).
        dimension: Number of dimensions per vector.
        num_clusters: Number of cluster centroids.
        random_seed: Seed for reproducible random number generation.
        cluster_std: Standard deviation of Gaussian noise around cluster centers.

    Returns:
        tuple containing (vectors, ids, metadata_dict)
    """
    if num_vectors % num_clusters != 0:
        raise ValueError(
            f"num_vectors ({num_vectors}) must be evenly divisible by num_clusters ({num_clusters})"
        )

    vectors_per_cluster = num_vectors // num_clusters

    # Initialize random generator with fixed seed
    rng = np.random.default_rng(random_seed)

    # 1. Generate cluster centers
    cluster_centers = rng.standard_normal((num_clusters, dimension))

    # 2. Generate vectors around their assigned cluster centers (500 per cluster)
    vectors = np.empty((num_vectors, dimension), dtype=np.float32)

    for i in range(num_clusters):
        start_idx = i * vectors_per_cluster
        end_idx = start_idx + vectors_per_cluster
        noise = rng.normal(
            loc=0.0,
            scale=cluster_std,
            size=(vectors_per_cluster, dimension),
        )
        vectors[start_idx:end_idx] = (cluster_centers[i] + noise).astype(np.float32)

    # 3. Normalize all vectors to unit length (L2 norm = 1.0)
    print("Normalizing vectors...")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Avoid division by zero if norm is zero
    norms = np.where(norms == 0, 1.0, norms)
    vectors = (vectors / norms).astype(np.float32)

    # 4. Generate integer IDs from 0 to num_vectors - 1
    ids = np.arange(num_vectors, dtype=np.int64)

    # 5. Build metadata
    metadata = {
        "number_of_vectors": num_vectors,
        "dimension": dimension,
        "number_of_clusters": num_clusters,
        "vectors_per_cluster": vectors_per_cluster,
        "random_seed": random_seed,
        "dtype": "float32",
        "normalized": True,
        "description": "Synthetic clustered vector dataset for vector database benchmarking",
    }

    return vectors, ids, metadata


def validate_dataset(vectors: np.ndarray, ids: np.ndarray) -> None:
    """Validate shapes, types, and properties of generated vectors."""
    assert vectors.shape == (
        NUM_VECTORS,
        DIMENSION,
    ), f"Expected shape ({NUM_VECTORS}, {DIMENSION}), got {vectors.shape}"
    assert ids.shape == (
        NUM_VECTORS,
    ), f"Expected shape ({NUM_VECTORS},), got {ids.shape}"
    assert (
        vectors.dtype == np.float32
    ), f"Expected dtype float32, got {vectors.dtype}"

    # Verify unit length
    vector_norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(
        vector_norms, 1.0, atol=1e-5
    ), "All vectors must have unit length (norm ~ 1.0)"


def main() -> None:
    """Execute dataset generation and save files."""
    print(f"Generating {NUM_VECTORS:,} vectors...")
    print(f"Dimension: {DIMENSION}")
    print(f"Clusters: {NUM_CLUSTERS}")

    # Determine project data directory (relative to project root)
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    vectors, ids, metadata = generate_dataset()

    # Define file paths
    vectors_path = data_dir / "vectors.npy"
    ids_path = data_dir / "ids.npy"
    metadata_path = data_dir / "metadata.json"

    # Save outputs
    np.save(vectors_path, vectors)
    print("Saved vectors to data/vectors.npy")

    np.save(ids_path, ids)
    print("Saved IDs to data/ids.npy")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to data/metadata.json")

    # Validate output
    validate_dataset(vectors, ids)

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()
