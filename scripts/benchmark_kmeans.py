"""Benchmark script for KMeans clustering on the 50,000-vector dataset."""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import numpy as np

from app.indexes.kmeans import KMeans


def run_benchmark(
    n_clusters: int = 100,
    max_iterations: int = 20,
    random_seed: int = 42,
) -> None:
    """Run KMeans training benchmark on the synthetic 50K dataset."""
    data_dir = project_root / "data"
    vectors_path = data_dir / "vectors.npy"

    if not vectors_path.exists():
        raise FileNotFoundError(
            "Dataset file data/vectors.npy not found. Please run `python scripts/generate_data.py` first."
        )

    # 1. Load dataset
    vectors = np.load(vectors_path)
    n_vectors, dimension = vectors.shape

    # 2. Instantiate KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        max_iterations=max_iterations,
        random_seed=random_seed,
    )

    # 3. Measure fit time
    start_time = time.perf_counter()
    kmeans.fit(vectors)
    end_time = time.perf_counter()

    training_time = end_time - start_time

    # 4. Calculate cluster statistics
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels, minlength=n_clusters)
    min_count = int(np.min(cluster_counts))
    max_count = int(np.max(cluster_counts))
    mean_count = float(np.mean(cluster_counts))
    all_clusters_assigned = bool(min_count > 0)

    # 5. Output benchmark results
    print("=" * 50)
    print("         KMeans Clustering Benchmark Results      ")
    print("=" * 50)
    print(f"Number of vectors       : {n_vectors:,}")
    print(f"Vector dimension        : {dimension}")
    print(f"Number of clusters      : {n_clusters}")
    print(f"Iterations performed    : {kmeans.n_iter_} / {max_iterations}")
    print(f"Training time           : {training_time:.4f} seconds")
    print(f"Cluster-size minimum    : {min_count}")
    print(f"Cluster-size maximum    : {max_count}")
    print(f"Cluster-size mean       : {mean_count:.1f}")
    print(f"All clusters non-empty  : {all_clusters_assigned}")
    print("=" * 50)
    print("\nCluster assignment counts (first 10 clusters):")
    for k in range(min(10, n_clusters)):
        print(f"  Cluster {k:2d}: {cluster_counts[k]} vectors")
    print("  ...")
    print("=" * 50)


if __name__ == "__main__":
    run_benchmark()
