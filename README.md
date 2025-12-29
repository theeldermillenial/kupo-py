# kupo-py

<div align="center">

[![PyPI version](https://badge.fury.io/py/kupo-py.svg)](https://badge.fury.io/py/kupo-py)
[![GitHub Actions Status](https://github.com/theeldermillenial/kupo-py/actions/workflows/publish.yml/badge.svg)](https://github.com/theeldermillenial/kupo-py/actions)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://theeldermillenial.github.io/kupo-py/)
[![Downloads](https://static.pepy.tech/badge/kupo-py)](https://pepy.tech/project/kupo-py)
[![Monthly Downloads](https://static.pepy.tech/badge/kupo-py/month)](https://pepy.tech/project/kupo-py)

</div>

A Python client and server for interacting with [Kupo](https://cardanosolutions.github.io/kupo/), a lightweight and fast Cardano blockchain indexer. This package provides both a simple client to query Kupo's HTTP API and an optional server mode for load balancing across multiple Kupo instances.



## Documentation

Full documentation is available at [https://theeldermillenial.github.io/kupo-py/](https://theeldermillenial.github.io/kupo-py/).

## Features

- **Type-safe**: All responses use Pydantic models for validation and type hints
- **Client Mode**: Asynchronous and synchronous HTTP client for querying a Kupo instance
- **Server Mode**: FastAPI-based load balancer with health monitoring across multiple Kupo instances
  - **DBSync Fallback**: Optional fallback to Cardano DBSync when Kupo instances are unavailable
  - **Pattern Synchronization**: Automatically sync patterns across multiple Kupo instances in server mode

## Requirements

- Python 3.12 or higher

## Installation

```bash
# Basic installation (client mode only)
pip install kupo-py

# With server mode support (FastAPI, Uvicorn, Typer CLI)
pip install kupo-py[server]

# With DBSync fallback support
pip install kupo-py[dbsync]

# Full installation with all optional dependencies
pip install kupo-py[server,dbsync]
```

## Quick Start

### Setting up Kupo

Before using this client, you'll need to have a Kupo instance running. The easiest way to get started is using Docker:

```bash
docker run -d \
  --name kupo \
  -p 1442:1442 \
  cardanosolutions/kupo:latest
```

For more detailed setup instructions and configuration options, please refer to the [Kupo documentation](https://cardanosolutions.github.io/kupo/).

### Client Mode

Use `KupoClient` to interact with a single Kupo instance:

```python
from kupo import KupoClient
from kupo.client import Order, Limit
from kupo.models import Point

# Initialize the client
client = KupoClient("http://localhost", 1442)

# Check the health of the API
health = await client.health_async()
print(f"API Health: {health}")

# Get matches for an address
matches = await client.get_matches_async(
    pattern="addr1qxy2k...",  # See pattern matching syntax below
    policy_id="policy1...",
    asset_name="asset1",
    order=Order.DESC
)
for match in matches:
    print(f"Found match: {match}")
```

### Server Mode

Server mode provides a load-balancing proxy that sits in front of multiple Kupo instances. This is useful when running in a production environment to both load balance as well as to route traffick to kupo instances that are not syncing after a new pattern update. It also comes with a dbsync fallback, so that if you have dbsync and a kupo instance is not available it will query dbsync and return results in the standard kupo format.

To launch it:

```bash
# Start the server (requires kupo-py[server])
kupo serve --host 0.0.0.0 --port 1442 --workers 4
```

The server provides:

- **Round-robin load balancing** across healthy Kupo instances
- **Automatic health checks** to detect and skip unhealthy instances
- **Pattern synchronization** to keep patterns in sync across all instances
- **DBSync fallback** for `/matches` endpoints when all Kupo instances are unavailable

## Environment Variables

### Client Mode

| Variable | Description | Default |
|----------|-------------|---------|
| `KUPO_BASE_URL` | Base URL of the Kupo instance | None (required) |
| `KUPO_PORT` | Port number of the Kupo instance | None (required) |

### Server Mode

| Variable | Description | Default |
|----------|-------------|---------|
| `KUPO_PY_URL_1` | URL of first Kupo instance (e.g., `http://localhost:1442`) | None |
| `KUPO_PY_URL_2` | URL of second Kupo instance | None |
| `KUPO_PY_URL_N` | Additional Kupo instances (add as many as needed) | None |
| `KUPO_PY_ROLLBACK_SLOT` | Slot number for pattern rollback operations | `0` |
| `KUPO_PY_CHECK_INTERVAL` | Health check interval in seconds | `1` |
| `KUPO_PY_PATTERN_SYNC_INTERVAL` | Pattern synchronization interval in seconds | `60` |

### DBSync Fallback (Optional)

| Variable | Description |
|----------|-------------|
| `DBSYNC_DB_NAME` | PostgreSQL database name |
| `DBSYNC_HOST` | PostgreSQL host |
| `DBSYNC_PORT` | PostgreSQL port |
| `DBSYNC_USER` | PostgreSQL username |
| `DBSYNC_PASS` | PostgreSQL password |

You can use the provided `sample.env` file as a template for your configuration.

## Pattern Matching

Kupo uses a powerful pattern matching syntax to filter and match blockchain data. The pattern can be:

- A Cardano address
- A policy ID
- A script hash
- A datum hash
- A combination of these using the `*` wildcard

For example:

- `addr1qxy2k...` - Match a specific address
- `policy1...` - Match a specific policy ID
- `script1...` - Match a specific script
- `datum1...` - Match a specific datum
- `*/*` - Match all addresses (wildcard)

For detailed pattern matching syntax and examples, see the [Kupo Pattern Matching documentation](https://cardanosolutions.github.io/kupo/patterns/).

## Examples

### Querying Matches

```python
from kupo import KupoClient
from kupo.client import Order

client = KupoClient("http://localhost", 1442)

# Get all matches with filtering
matches = await client.get_all_matches_async(
    spent=True,  # Only get spent transactions
    created_after=1000000,  # Created after slot 1000000
    order=Order.ASC  # Oldest first
)

# Get matches for a specific address
matches = await client.get_matches_async(
    pattern="addr1qxy2k...",
    unspent=True,  # Only get unspent UTxOs
    policy_id="policy1..."  # Filter by policy ID
)
```

### Working with Scripts and Datums

```python
# Get a script by its hash
script = await client.get_script_by_hash_async("script1...")
print(f"Script: {script}")

# Get a datum by its hash
datum = await client.get_datum_by_hash_async("datum1...")
print(f"Datum: {datum}")

# Get metadata for a transaction
metadata = await client.get_metadata_by_tx_async("tx1...")
print(f"Transaction metadata: {metadata}")
```

### Managing Patterns

```python
from kupo.client import Limit
from kupo.models import Point

# Get all patterns
patterns = await client.get_patterns_async()
print(f"Active patterns: {patterns}")

# Add a new pattern
pattern = await client.add_pattern_async(
    pattern="addr1qxy2k...",
    rollback_to=Point(slot_no=1000000, header_hash="abcd1234..."),
    limit=Limit.SAFE
)

# Add multiple patterns at once
patterns = await client.bulk_add_pattern_async(
    patterns=["addr1qxy2k...", "addr1abc..."],
    rollback_to={"slot_no": 1000000}
)
```

### Server Mode Endpoints

When running in server mode, additional endpoints are available:

```python
import httpx

# Check the status of all backends
response = httpx.get("http://localhost:1442/router/status")
print(response.json())
# Returns: {"kupo_servers": [...], "dbsync_servers": [...]}

# Force a specific backend using headers
response = httpx.get(
    "http://localhost:1442/matches/addr1...",
    headers={"X-Force-Backend": "dbsync"}  # or "kupo"
)
```

## API Reference

For detailed API documentation, please refer to the [API Reference](api/client.md) section.

For upstream Kupo API documentation, see the [Kupo documentation](https://cardanosolutions.github.io/kupo/).

## Contributing

Contributions are welcome! Please follow these guidelines when contributing:

### Branch Naming

This project uses semantic versioning and automated version bumping based on branch names. When creating a new branch, please follow these naming conventions:

- For new features: `feat/your-feature-name` or `feature/your-feature-name`
  - This will trigger a minor version bump when merged to dev
- For bug fixes: `fix/your-fix-name` or `bugfix/your-fix-name`
  - This will trigger a patch version bump when merged to dev

### Development Workflow

1. Create a new branch following the naming convention above
2. Make your changes
3. Submit a pull request to the `dev` branch
4. Once merged, the version will be automatically bumped based on your branch name
5. When ready for release, create a pull request from `dev` to `master`

### Code Style

- Follow PEP 8 guidelines
- Use type hints
- Add docstrings for all public functions and classes
- Include tests for new functionality

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency. Before submitting a pull request:

1. Install pre-commit:
   ```bash
   pip install pre-commit
   ```

2. Install the project's pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. Run the pre-commit hooks on all files:
   ```bash
   pre-commit run --all-files
   ```

4. Make sure all hooks pass before submitting your pull request.

The pre-commit hooks will check for:

- Code formatting and linting (ruff)
- Type checking (mypy)
- And other quality checks

Please feel free to submit a Pull Request!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
