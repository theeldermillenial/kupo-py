import os
import time

import pytest
from charli3_dendrite import MinswapCPPState
from dotenv import load_dotenv
from pycardano import Address

from kupo import KupoClient

load_dotenv()


@pytest.fixture
def kupo_client():

    url = (
        "http://localhost"
        if os.environ.get("KUPO_BASE_URL") is None
        else os.environ.get("KUPO_BASE_URL")
    )

    port = (
        1442
        if os.environ.get("KUPO_PORT") is None
        else int(os.environ.get("KUPO_PORT"))
    )

    return KupoClient(base_url=url, port=port)


@pytest.fixture(params=[True, False])
def unspent(request):
    return request.param


@pytest.fixture(params=[True, False])
def spent(request):
    return request.param


@pytest.fixture(params=[None, time.time() - 3600 * 24 * 2, time.time() - 3600 * 24])
def spent_after(request):
    return request.param


@pytest.fixture(params=[None, time.time() - 3600 * 24 * 2, time.time() - 3600 * 24])
def created_after(request):
    return request.param


@pytest.fixture(params=[None, time.time()])
def spent_before(request):
    return request.param


@pytest.fixture(params=[None, time.time()])
def created_before(request):
    return request.param


patterns: list[str] = []

address = Address.decode(MinswapCPPState.pool_selector().addresses[0])

patterns.append(address.payment_part.payload.hex() + "/*")


@pytest.fixture(params=patterns)
def pattern(request):
    return request.param
