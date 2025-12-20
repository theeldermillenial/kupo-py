import pytest
from kupo.models import ConnectionStatus, Health


def test_health(kupo_client):
    """Test the synchronous health endpoint."""
    health = kupo_client.health()
    assert isinstance(health, Health)
    assert isinstance(health.connection_status, ConnectionStatus)
    assert isinstance(health.version, str)
    assert isinstance(health.syncing, bool)

    # Optional: assert specific values if we expect a healthy connected instance
    # assert health.connection_status == ConnectionStatus.connected


@pytest.mark.asyncio
async def test_health_async(kupo_client):
    """Test the asynchronous health endpoint."""
    health = await kupo_client.health_async()
    assert isinstance(health, Health)
    assert isinstance(health.connection_status, ConnectionStatus)
    assert isinstance(health.version, str)
    assert isinstance(health.syncing, bool)
