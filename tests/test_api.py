"""Tests for FastAPI REST endpoints and static UI serving."""

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_index_html_serving(client):
    """Test that GET / serves the index.html page."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Vector Database From Scratch" in response.text


def test_sample_endpoint(client):
    """Test that GET /sample returns a valid 128-D vector from the dataset."""
    response = client.get("/sample")
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "vector" in data
    assert isinstance(data["id"], int)
    assert isinstance(data["vector"], list)
    assert len(data["vector"]) == 128
    assert all(isinstance(x, (int, float)) for x in data["vector"])


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_stats_endpoint(client):
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["exact_count"] >= 50000
    assert data["ivf_count"] >= 50000
    assert data["dimension"] == 128
    assert data["ivf_clusters"] == 100
    assert data["default_nprobe"] == 5


def test_exact_search_endpoint(client):
    query_vector = [0.1] * 128
    response = client.post("/exact/search", json={"vector": query_vector, "k": 5})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["count"] == 5
    assert len(data["results"]) == 5
    assert "id" in data["results"][0]
    assert "score" in data["results"][0]


def test_ivf_search_endpoint(client):
    query_vector = [0.1] * 128
    response = client.post("/ivf/search", json={"vector": query_vector, "k": 5, "nprobe": 2})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["count"] == 5
    assert data["nprobe"] == 2
    assert "candidates" in data
    assert data["candidates"] > 0


def test_search_invalid_dimension(client):
    response = client.post("/exact/search", json={"vector": [0.1] * 10, "k": 5})
    assert response.status_code == 400


def test_insert_and_delete_exact(client):
    test_id = 999991
    test_vec = [0.05] * 128

    # Insert
    resp = client.post("/exact/insert", json={"id": test_id, "vector": test_vec})
    assert resp.status_code == 200
    assert resp.json() == {"message": "Vector inserted", "id": test_id}

    # Duplicate insert should return 400
    resp_dup = client.post("/exact/insert", json={"id": test_id, "vector": test_vec})
    assert resp_dup.status_code == 400

    # Delete
    resp_del = client.delete(f"/exact/delete/{test_id}")
    assert resp_del.status_code == 200
    assert resp_del.json() == {"message": "Vector deleted", "id": test_id}

    # Delete non-existent returns 404
    resp_del_none = client.delete(f"/exact/delete/{test_id}")
    assert resp_del_none.status_code == 404


def test_insert_and_delete_ivf(client):
    test_id = 999992
    test_vec = [0.05] * 128

    # Insert
    resp = client.post("/ivf/insert", json={"id": test_id, "vector": test_vec})
    assert resp.status_code == 200
    assert resp.json() == {"message": "Vector inserted", "id": test_id}

    # Duplicate insert should return 400
    resp_dup = client.post("/ivf/insert", json={"id": test_id, "vector": test_vec})
    assert resp_dup.status_code == 400

    # Delete
    resp_del = client.delete(f"/ivf/delete/{test_id}")
    assert resp_del.status_code == 200
    assert resp_del.json() == {"message": "Vector deleted", "id": test_id}

    # Delete non-existent returns 404
    resp_del_none = client.delete(f"/ivf/delete/{test_id}")
    assert resp_del_none.status_code == 404
