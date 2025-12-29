# Installation Guide

## Requirements

- Python 3.12 or higher
- pip or uv (Python package installers)

## Installation

### Basic Installation (Client Mode)

The simplest way to install kupo-py is using pip:

```bash
pip install kupo-py
```

Or using uv:

```bash
uv add kupo-py
```

### Optional Dependencies

kupo-py offers optional dependencies for extended functionality:

```bash
# Server mode: FastAPI load balancer with health monitoring
pip install kupo-py[server]

# DBSync fallback: Query Cardano DBSync when Kupo is unavailable
pip install kupo-py[dbsync]

# Full installation with all optional dependencies
pip install kupo-py[server,dbsync]
```

## Development Installation

For development purposes, clone the repository and install with dev dependencies:

```bash
git clone https://github.com/theeldermillenial/kupo-py.git
cd kupo-py

# Using uv (recommended)
uv sync --dev

# Or using pip
pip install -e ".[server,dbsync]"
```

## Environment Variables

kupo-py uses environment variables for configuration. You can set these in your shell or use a `.env` file.

A `sample.env` file is provided in the repository as a template.

### Client Mode

These variables configure the `KupoClient` when connecting to a single Kupo instance:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `KUPO_BASE_URL` | Base URL of your Kupo instance (e.g., `http://localhost`) | Yes* | None |
| `KUPO_PORT` | Port number of your Kupo instance | Yes* | None |

*Required only if not passed directly to `KupoClient()` constructor.

Example `.env` for client mode:

```bash
KUPO_BASE_URL=http://localhost
KUPO_PORT=1442
```

### Server Mode

These variables configure the load-balancing server when running `kupo serve`:

| Variable | Description | Default |
|----------|-------------|---------|
| `KUPO_PY_URL_1` | URL of the first Kupo instance (e.g., `http://localhost:1442`) | None |
| `KUPO_PY_URL_2` | URL of the second Kupo instance | None |
| `KUPO_PY_URL_N` | Additional Kupo instances (add as many as needed) | None |
| `KUPO_PY_ROLLBACK_SLOT` | Slot number used when adding patterns via sync | `0` |
| `KUPO_PY_CHECK_INTERVAL` | How often to check Kupo instance health (seconds) | `1` |
| `KUPO_PY_PATTERN_SYNC_INTERVAL` | How often to sync patterns across instances (seconds) | `60` |

Example `.env` for server mode:

```bash
KUPO_PY_ROLLBACK_SLOT=0
KUPO_PY_CHECK_INTERVAL=1
KUPO_PY_PATTERN_SYNC_INTERVAL=60
KUPO_PY_URL_1=http://localhost:1442
KUPO_PY_URL_2=http://localhost:1443
```

### DBSync Fallback (Optional)

If you have the `dbsync` optional dependency installed and want to use DBSync as a fallback when Kupo instances are unavailable:

| Variable | Description |
|----------|-------------|
| `DBSYNC_DB_NAME` | PostgreSQL database name (e.g., `cexplorer`) |
| `DBSYNC_HOST` | PostgreSQL host |
| `DBSYNC_PORT` | PostgreSQL port (typically `5432`) |
| `DBSYNC_USER` | PostgreSQL username |
| `DBSYNC_PASS` | PostgreSQL password |

Example `.env` for DBSync:

```bash
DBSYNC_DB_NAME=cexplorer
DBSYNC_HOST=localhost
DBSYNC_PORT=5432
DBSYNC_USER=postgres
DBSYNC_PASS=your_password_here
```

## Running the Server

After installing with server dependencies, you can start the load-balancing proxy:

```bash
# Basic usage
kupo serve

# With custom host and port
kupo serve --host 0.0.0.0 --port 1442

# With multiple workers for production
kupo serve --host 0.0.0.0 --port 1442 --workers 4
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Host address to bind to | `localhost` |
| `--port` | Port number to listen on | `1442` |
| `--workers` | Number of Uvicorn worker processes | `1` |

## Verifying Installation

You can verify your installation by checking the client:

```python
from kupo import KupoClient

# This should import without errors
client = KupoClient("http://localhost", 1442)

# If you have a Kupo instance running, test the connection
health = client.health()
print(health)
```

For server mode verification:

```bash
# Check if the CLI is available
kupo serve --help
```
