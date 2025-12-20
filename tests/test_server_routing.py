import pytest
from fastapi.testclient import TestClient
from kupo.server import app


@pytest.fixture
def client_fixture():
    with TestClient(app) as client:
        yield client


def test_routing_kupo(
    client_fixture,
    pattern,
    spent,
    unspent,
    created_after,
    created_before,
    spent_after,
    spent_before,
):
    """Test forcing Kupo backend."""
    # Logic to skip invalid combinations (mirrors test_dbsync_comparison.py)
    if spent and unspent:
        pytest.skip("Cannot have both spent and unspent")
    if spent and (spent_after is not None or spent_before is not None):
        pytest.skip("Cannot have spent and spent_after or spent_before")
    if unspent and (spent_after is not None or spent_before is not None):
        pytest.skip("Cannot have unspent and spent_after or spent_before")
    if spent_after is not None and created_after is not None:
        pytest.skip("Cannot have spent_after and created_after")
    if spent_before is not None and created_before is not None:
        pytest.skip("Cannot have spent_before and created_before")

    params = {
        "created_after": created_after,
        "created_before": created_before,
        "spent_after": spent_after,
        "spent_before": spent_before,
    }
    if spent:
        params["spent"] = ""
    if unspent:
        params["unspent"] = ""

    # Filter out None params
    params = {k: v for k, v in params.items() if v is not None}

    response = client_fixture.get(
        f"/matches/{pattern}",
        headers={"X-Force-Backend": "kupo"},
        params=params,
    )

    assert response.status_code in [200, 202], f"Kupo failed: {response.text}"
    assert "X-Kupo-Source" in response.headers
    assert response.headers["X-Kupo-Source"] != "dbsync"
    assert response.headers["X-Kupo-Source"].startswith("http")


def test_routing_dbsync(
    client_fixture,
    pattern,
    spent,
    unspent,
    created_after,
    created_before,
    spent_after,
    spent_before,
):
    """Test forcing DBSync backend."""
    if spent and unspent:
        pytest.skip("Cannot have both spent and unspent")
    if spent and (spent_after is not None or spent_before is not None):
        pytest.skip("Cannot have spent and spent_after or spent_before")
    if unspent and (spent_after is not None or spent_before is not None):
        pytest.skip("Cannot have unspent and spent_after or spent_before")
    if spent_after is not None and created_after is not None:
        pytest.skip("Cannot have spent_after and created_after")
    if spent_before is not None and created_before is not None:
        pytest.skip("Cannot have spent_before and created_before")

    params = {
        "created_after": created_after,
        "created_before": created_before,
        "spent_after": spent_after,
        "spent_before": spent_before,
    }
    if spent:
        params["spent"] = ""
    if unspent:
        params["unspent"] = ""

    params = {k: v for k, v in params.items() if v is not None}

    response = client_fixture.get(
        f"/matches/{pattern}",
        headers={"X-Force-Backend": "dbsync"},
        params=params,
    )

    assert response.status_code == 200, f"DBSync failed: {response.text}"
    assert response.headers.get("X-Kupo-Source") == "dbsync"


def test_results_match(
    client_fixture,
    pattern,
    spent,
    unspent,
    created_after,
    created_before,
    spent_after,
    spent_before,
):
    """Test consistency between backends."""
    if spent and unspent:
        pytest.skip("Cannot have both spent and unspent")
    if spent and (spent_after is not None or spent_before is not None):
        pytest.skip("Cannot have spent and spent_after or spent_before")
    if unspent and (spent_after is not None or spent_before is not None):
        pytest.skip("Cannot have unspent and spent_after or spent_before")
    if spent_after is not None and created_after is not None:
        pytest.skip("Cannot have spent_after and created_after")
    if spent_before is not None and created_before is not None:
        pytest.skip("Cannot have spent_before and created_before")

    params = {
        "created_after": created_after,
        "created_before": created_before,
        "spent_after": spent_after,
        "spent_before": spent_before,
    }
    # For TestClient param handling with flags (keys without values)
    # requests/starlette usually handle None/Empty string specially.
    # We'll rely on presence.
    if spent:
        params["spent"] = ""
    if unspent:
        params["unspent"] = ""

    params = {k: v for k, v in params.items() if v is not None}

    # Get Kupo result
    res_kupo = client_fixture.get(
        f"/matches/{pattern}",
        headers={"X-Force-Backend": "kupo"},
        params=params,
    )

    # Get DBSync result
    res_dbsync = client_fixture.get(
        f"/matches/{pattern}",
        headers={"X-Force-Backend": "dbsync"},
        params=params,
    )

    # If Kupo fails (e.g. syncing), we might skip or fail.
    # But test expects valid server.
    assert res_kupo.status_code == 200
    assert res_dbsync.status_code == 200

    kupo_data = res_kupo.json()
    dbsync_data = res_dbsync.json()

    assert isinstance(kupo_data, list)
    assert isinstance(dbsync_data, list)

    # Sort by tx_hash and output_index
    def sort_key(m):
        return (m["transaction_id"], m["output_index"])

    kupo_data.sort(key=sort_key)
    dbsync_data.sort(key=sort_key)

    assert len(kupo_data) == len(dbsync_data), (
        f"Count mismatch for {pattern}: "
        f"Kupo={len(kupo_data)}, DBSync={len(dbsync_data)}"
    )

    for i, (k, d) in enumerate(zip(kupo_data, dbsync_data)):
        # Compare as dicts
        # Need to handle potential minor diffs if any (e.g. None vs null)
        # requests.json() handles null as None.
        assert k == d, f"Match {i} mismatch.\nKupo: {k}\nDBSync: {d}"
