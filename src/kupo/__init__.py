"""kupo-py: A Python client and server for Kupo.

Kupo is a lightweight Cardano blockchain indexer. This package provides:
- KupoClient: A client for interacting with Kupo's HTTP API
- Server mode: A load balancing proxy for multiple Kupo instances

Example:
    >>> from kupo import KupoClient
    >>> client = KupoClient("http://localhost", 1442)
    >>> health = client.health()
"""

from kupo.client import KupoClient

__all__ = ["KupoClient"]
