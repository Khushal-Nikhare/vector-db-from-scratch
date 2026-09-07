"""FastAPI application exposing REST API and interactive demo UI for Exact and IVF vector indexes."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.indexes.exact_index import ExactIndex
from app.indexes.ivf_flat import IVFFlat

# Global in-memory index references
exact_index: Optional[ExactIndex] = None
ivf_index: Optional[IVFFlat] = None
EXPECTED_DIMENSION = 128

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


def load_indexes() -> tuple[ExactIndex, IVFFlat]:
    """Load dataset from disk and initialize both in-memory indexes."""
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    vectors_path = data_dir / "vectors.npy"
    ids_path = data_dir / "ids.npy"

    if not vectors_path.exists() or not ids_path.exists():
        raise RuntimeError(
            "Dataset files not found in data/. Please run `python scripts/generate_data.py` first."
        )

    vectors = np.load(vectors_path)
    ids = np.load(ids_path)

    # Initialize ExactIndex
    exact_idx = ExactIndex(vectors, ids)

    # Initialize and train IVFFlat
    ivf_idx = IVFFlat(
        n_clusters=100,
        nprobe=5,
        max_iterations=20,
        random_seed=42,
    )
    ivf_idx.fit(vectors, ids)

    return exact_idx, ivf_idx


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load indices on startup."""
    global exact_index, ivf_index
    exact_index, ivf_index = load_indexes()
    yield
    # Cleanup on shutdown if needed
    exact_index = None
    ivf_index = None


app = FastAPI(
    title="Vector DB From Scratch API",
    description="REST API for Educational Vector Database using Exact and IVF-Flat Indexes",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static frontend directory
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------
# Request and Response Schemas
# ---------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"


class StatsResponse(BaseModel):
    exact_count: int
    ivf_count: int
    dimension: int
    ivf_clusters: int
    default_nprobe: int


class SampleResponse(BaseModel):
    id: int
    vector: list[float]


class SearchRequest(BaseModel):
    vector: list[float] = Field(..., description="Query vector of dimension 128")
    k: int = Field(default=10, ge=1, description="Number of top results to return")


class IVFSearchRequest(BaseModel):
    vector: list[float] = Field(..., description="Query vector of dimension 128")
    k: int = Field(default=10, ge=1, description="Number of top results to return")
    nprobe: Optional[int] = Field(default=None, ge=1, le=100, description="Number of centroid clusters to probe")


class SearchResultItem(BaseModel):
    id: int
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    count: int


class IVFSearchResponse(BaseModel):
    results: list[SearchResultItem]
    count: int
    nprobe: int
    candidates: int


class InsertRequest(BaseModel):
    id: int = Field(..., description="Unique integer ID for the vector")
    vector: list[float] = Field(..., description="Vector of dimension 128")


class InsertResponse(BaseModel):
    message: str = "Vector inserted"
    id: int


class DeleteResponse(BaseModel):
    message: str = "Vector deleted"
    id: int


# ---------------------------------------------------------
# Helper Validation Function
# ---------------------------------------------------------

def validate_vector_dim(vector: list[float], expected_dim: int = EXPECTED_DIMENSION) -> np.ndarray:
    """Validate vector length and cast to float32 numpy array."""
    if len(vector) != expected_dim:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Vector dimension must be exactly {expected_dim}, got {len(vector)}.",
        )
    return np.asarray(vector, dtype=np.float32)


# ---------------------------------------------------------
# Frontend Root
# ---------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    """Serve the single-page application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index HTML not found.")
    return FileResponse(str(index_file))


# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@app.get("/stats", response_model=StatsResponse)
def get_stats() -> StatsResponse:
    """Return current index statistics."""
    if exact_index is None or ivf_index is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Indexes not initialized.",
        )
    return StatsResponse(
        exact_count=exact_index.count,
        ivf_count=ivf_index.count,
        dimension=exact_index.dimension,
        ivf_clusters=ivf_index.n_clusters,
        default_nprobe=ivf_index.nprobe,
    )


@app.get("/sample", response_model=SampleResponse)
def get_sample() -> SampleResponse:
    """Return a randomly chosen existing vector from the loaded dataset."""
    if exact_index is None or exact_index.count == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Exact index not initialized or empty.",
        )
    rng = np.random.default_rng()
    random_pos = int(rng.integers(0, exact_index.count))
    sample_id = int(exact_index._ids[random_pos])
    sample_vector = [round(float(v), 6) for v in exact_index._vectors[random_pos]]
    return SampleResponse(id=sample_id, vector=sample_vector)


@app.post("/exact/search", response_model=SearchResponse)
def exact_search(request: SearchRequest) -> SearchResponse:
    """Perform exact brute-force cosine similarity search."""
    if exact_index is None:
        raise HTTPException(status_code=503, detail="Exact index not initialized.")

    query_vec = validate_vector_dim(request.vector, exact_index.dimension)
    try:
        top_ids, top_scores = exact_index.search(query_vec, k=request.k)
        results = [
            SearchResultItem(id=int(vid), score=round(float(score), 6))
            for vid, score in zip(top_ids, top_scores)
        ]
        return SearchResponse(results=results, count=len(results))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/ivf/search", response_model=IVFSearchResponse)
def ivf_search(request: IVFSearchRequest) -> IVFSearchResponse:
    """Perform approximate nearest neighbor search using IVF-Flat indexing."""
    if ivf_index is None:
        raise HTTPException(status_code=503, detail="IVF index not initialized.")

    query_vec = validate_vector_dim(request.vector, ivf_index.dimension)
    nprobe_val = request.nprobe if request.nprobe is not None else ivf_index.nprobe

    if nprobe_val < 1 or nprobe_val > ivf_index.n_clusters:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"nprobe must be between 1 and {ivf_index.n_clusters}, got {nprobe_val}.",
        )

    try:
        # 1. Determine candidate count using existing IVF centroid lists without redundant vector scans
        q_norm = np.linalg.norm(query_vec)
        q_normed = query_vec / q_norm if q_norm > 0 else query_vec
        selected_clusters = ivf_index._find_nearest_centroids(q_normed, nprobe_val)
        candidates_count = sum(len(ivf_index._inverted_lists[c]) for c in selected_clusters)

        # 2. Run IVF search
        top_ids, top_scores = ivf_index.search(query_vec, k=request.k, nprobe=nprobe_val)
        results = [
            SearchResultItem(id=int(vid), score=round(float(score), 6))
            for vid, score in zip(top_ids, top_scores)
        ]
        return IVFSearchResponse(
            results=results,
            count=len(results),
            nprobe=nprobe_val,
            candidates=candidates_count,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/exact/insert", response_model=InsertResponse)
def exact_insert(request: InsertRequest) -> InsertResponse:
    """Insert a single vector into ExactIndex."""
    if exact_index is None:
        raise HTTPException(status_code=503, detail="Exact index not initialized.")

    vec = validate_vector_dim(request.vector, exact_index.dimension)
    try:
        exact_index.insert(vec, ids=request.id)
        return InsertResponse(message="Vector inserted", id=request.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.post("/ivf/insert", response_model=InsertResponse)
def ivf_insert(request: InsertRequest) -> InsertResponse:
    """Insert a single vector into IVFFlat."""
    if ivf_index is None:
        raise HTTPException(status_code=503, detail="IVF index not initialized.")

    vec = validate_vector_dim(request.vector, ivf_index.dimension)
    try:
        ivf_index.insert(vec, ids=request.id)
        return InsertResponse(message="Vector inserted", id=request.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@app.delete("/exact/delete/{vector_id}", response_model=DeleteResponse)
def exact_delete(vector_id: int) -> DeleteResponse:
    """Delete a vector by ID from ExactIndex."""
    if exact_index is None:
        raise HTTPException(status_code=503, detail="Exact index not initialized.")

    if vector_id not in exact_index._id_set:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector ID {vector_id} not found in ExactIndex.",
        )

    exact_index.delete(vector_id)
    return DeleteResponse(message="Vector deleted", id=vector_id)


@app.delete("/ivf/delete/{vector_id}", response_model=DeleteResponse)
def ivf_delete(vector_id: int) -> DeleteResponse:
    """Delete a vector by ID from IVFFlat."""
    if ivf_index is None:
        raise HTTPException(status_code=503, detail="IVF index not initialized.")

    if vector_id not in ivf_index._id_to_pos:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Vector ID {vector_id} not found in IVFFlat.",
        )

    ivf_index.delete(vector_id)
    return DeleteResponse(message="Vector deleted", id=vector_id)
