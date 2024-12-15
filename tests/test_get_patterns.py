import pytest


def patterns(benchmark, kupo_client):
    benchmark(kupo_client.get_patterns)


@pytest.mark.asyncio
async def test_get_patterns(kupo_client):

    await kupo_client.get_patterns_async()
