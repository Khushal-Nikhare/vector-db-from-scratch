"""Benchmark script for ExactIndex brute-force search."""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running script directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import time
import numpy as np

from app.indexes.exact_index import ExactIndex


def run_benchmark(num_queries: int = 10, k: int = 10, random_seed: int = 42) -> None:
    """Run exact search benchmark on the synthetic dataset."""
    data_dir = project_root / "data"
    vectors_path = data_dir / "vectors.npy"
    ids_path = data_dir / "ids.npy"

    if not vectors_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            "Dataset files not found in data/. Please run `python scripts/generate_data.py` first."
        )

    # 1. Load dataset
    vectors = np.load(vectors_path)
    ids = np.load(ids_path)

    # 2. Build ExactIndex
    index = ExactIndex(vectors, ids)

    # 3. Generate reproducible benchmark queries from the existing dataset
    rng = np.random.default_rng(random_seed)
    query_indices = rng.choice(len(vectors), size=num_queries, replace=False)
    queries = vectors[query_indices]

    # 4. Measure top-k search times
    start_time = time.perf_counter()
    for query in queries:
        index.search(query, k=k)
    end_time = time.perf_counter()

    total_query_time = end_time - start_time
    avg_latency_ms = (total_query_time / num_queries) * 1000.0
    qps = num_queries / total_query_time if total_query_time > 0 else float("inf")

    # 5. Output benchmark results
    print("=" * 45)
    print("      ExactIndex Search Benchmark Results     ")
    print("=" * 45)
    print(f"Number of vectors     : {index.count:,}")
    print(f"Vector dimension      : {index.dimension}")
    print(f"Top-k                 : {k}")
    print(f"Number of queries     : {num_queries}")
    print(f"Total query time      : {total_query_time:.4f} seconds")
    print(f"Average query latency : {avg_latency_ms:.2f} ms")
    print(f"Queries per second    : {qps:.1f} QPS")
    print("=" * 45)


if __name__ == "__main__":
    run_benchmark()
