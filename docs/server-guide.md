# Server Mode Guide

Server mode transforms kupo-py into a load-balancing proxy that sits in front of multiple Kupo instances. This guide covers the architecture, configuration, and operation of server mode.

## Overview

When running in server mode, kupo-py provides:

- **Round-robin load balancing** across healthy Kupo instances
- **Automatic health monitoring** with configurable check intervals
- **Pattern synchronization** to keep all Kupo instances in sync
- **DBSync fallback** for `/matches` endpoints when all Kupo instances are unavailable
- **Transparent proxying** of all Kupo API endpoints

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         kupo-py Server              │
                    │                                     │
  Client Request    │  ┌─────────────────────────────┐   │
────────────────────┼─▶│     FastAPI Application     │   │
                    │  └──────────────┬──────────────┘   │
                    │                 │                   │
                    │     ┌───────────┴───────────┐      │
                    │     ▼                       ▼      │
                    │  ┌──────┐               ┌──────┐   │
                    │  │Health│               │Round │   │
                    │  │Check │               │Robin │   │
                    │  └──┬───┘               └──┬───┘   │
                    │     │                      │       │
                    └─────┼──────────────────────┼───────┘
                          │                      │
          ┌───────────────┼──────────────────────┼───────────────┐
          │               │                      │               │
          ▼               ▼                      ▼               ▼
    ┌──────────┐    ┌──────────┐          ┌──────────┐    ┌──────────┐
    │  Kupo 1  │    │  Kupo 2  │   ...    │  Kupo N  │    │  DBSync  │
    │ (primary)│    │(secondary│          │          │    │(fallback)│
    └──────────┘    └──────────┘          └──────────┘    └──────────┘
```

## Installation

Server mode requires the `server` optional dependency:

```bash
pip install kupo-py[server]

# Or with DBSync fallback support
pip install kupo-py[server,dbsync]
```

## Configuration

### Environment Variables

Configure server mode using environment variables. Create a `.env` file or set them in your environment:

```bash
# Kupo instances (add as many as needed)
KUPO_PY_URL_1=http://kupo1.example.com:1442
KUPO_PY_URL_2=http://kupo2.example.com:1442
KUPO_PY_URL_3=http://kupo3.example.com:1442

# Health check interval (seconds)
KUPO_PY_CHECK_INTERVAL=1

# Pattern synchronization interval (seconds)
KUPO_PY_PATTERN_SYNC_INTERVAL=60

# Slot number for pattern rollback operations
KUPO_PY_ROLLBACK_SLOT=0
```

### DBSync Fallback (Optional)

If you have access to a Cardano DBSync database, you can configure it as a fallback:

```bash
DBSYNC_DB_NAME=cexplorer
DBSYNC_HOST=dbsync.example.com
DBSYNC_PORT=5432
DBSYNC_USER=postgres
DBSYNC_PASS=your_password
```

When all Kupo instances are unavailable or syncing, requests to `/matches` endpoints will automatically fall back to DBSync.

## Starting the Server

Use the `kupo serve` CLI command:

```bash
# Basic usage (binds to localhost:1442)
kupo serve

# Specify host and port
kupo serve --host 0.0.0.0 --port 8080

# Run with multiple workers for production
kupo serve --host 0.0.0.0 --port 1442 --workers 4
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--host` | Host address to bind to | `localhost` |
| `--port` | Port number to listen on | `1442` |
| `--workers` | Number of Uvicorn worker processes | `1` |

## How It Works

### Health Monitoring

The server periodically checks the health of each Kupo instance by calling their `/health` endpoint. Instances are considered healthy when:

1. They respond to health checks
2. They are not currently syncing (`syncing: false`)

Unhealthy or syncing instances are temporarily removed from the load balancing pool.

### Load Balancing

Requests are distributed using a round-robin algorithm:

1. The server maintains a cycle of all configured Kupo instances
2. For each incoming request, it selects the next healthy instance
3. If no healthy Kupo instances are available, it falls back to DBSync (if configured)
4. If no backends are available, it returns a 503 error

### Pattern Synchronization

When running multiple Kupo instances, patterns must be kept in sync. The server handles this automatically:

1. Periodically fetches patterns from all instances
2. Computes the union of all patterns
3. Adds missing patterns to instances that don't have them
4. Uses `KUPO_PY_ROLLBACK_SLOT` for rollback operations

!!! note "Sync Requirements"
    Pattern synchronization only runs when all Kupo instances are healthy and not syncing.

### Pattern Update Protection

When updating patterns via the API, the server enforces safety rules:

- Pattern updates require at least 2 healthy Kupo instances
- This prevents pattern loss if one instance fails during an update

## API Endpoints

### Proxied Endpoints

All standard Kupo endpoints are proxied transparently:

| Endpoint | Description |
|----------|-------------|
| `GET /matches` | Get all matches |
| `GET /matches/{pattern}` | Get matches for a pattern |
| `GET /patterns` | Get all patterns |
| `PUT /patterns` | Add multiple patterns |
| `PUT /patterns/{pattern}` | Add a single pattern |
| `GET /health` | Health check |
| `GET /datums/{hash}` | Get datum by hash |
| `GET /scripts/{hash}` | Get script by hash |
| `GET /metadata/{tx}` | Get transaction metadata |

### Server-Specific Endpoints

#### GET /router/status

Returns the status of all backend servers:

```bash
curl http://localhost:1442/router/status
```

Response:

```json
{
  "kupo_servers": [
    {
      "url": "http://localhost:1442",
      "health": {
        "connection_status": "connected",
        "most_recent_checkpoint": 120000000,
        "most_recent_node_tip": 120000000,
        "syncing": false,
        "version": "2.9.0"
      }
    },
    {
      "url": "http://localhost:1443",
      "health": null
    }
  ],
  "dbsync_servers": [
    "postgresql://postgres:***@localhost:5432/cexplorer"
  ]
}
```

### Request Headers

#### X-Force-Backend

Force a request to use a specific backend:

```bash
# Force DBSync
curl -H "X-Force-Backend: dbsync" http://localhost:1442/matches/addr1...

# Force Kupo (skip DBSync fallback)
curl -H "X-Force-Backend: kupo" http://localhost:1442/matches/addr1...
```

### Response Headers

#### X-Kupo-Source

Every response includes this header indicating which backend handled the request:

```bash
curl -v http://localhost:1442/matches/addr1...
# Response header: X-Kupo-Source: http://localhost:1442
# Or: X-Kupo-Source: dbsync
```

## DBSync Fallback Details

The DBSync fallback provides a subset of Kupo functionality:

### Supported Endpoints

| Endpoint | DBSync Support |
|----------|---------------|
| `GET /matches` | ✅ Full support |
| `GET /matches/{pattern}` | ✅ Full support |
| `GET /patterns` | ❌ Not supported |
| `PUT /patterns` | ❌ Not supported |
| `GET /health` | ❌ Not supported |
| `GET /datums/{hash}` | ❌ Not supported |
| `GET /scripts/{hash}` | ❌ Not supported |
| `GET /metadata/{tx}` | ❌ Not supported |

### Supported Query Parameters

All standard match query parameters are supported:

- `spent` / `unspent`
- `created_after` / `created_before`
- `spent_after` / `spent_before`
- `policy_id` / `asset_name`
- `transaction_id` / `output_index`
- `order`
- `resolve_hashes`

### Pattern Format

DBSync supports the same pattern formats as Kupo:

- Full addresses: `addr1qxy2k...`
- Credential patterns: `{payment_cred}/{stake_cred}`
- Wildcards: `*`, `*/*`

## Production Deployment

### Recommended Configuration

For production deployments:

```bash
# Run at least 2 Kupo instances for redundancy
KUPO_PY_URL_1=http://kupo1:1442
KUPO_PY_URL_2=http://kupo2:1442

# More frequent health checks
KUPO_PY_CHECK_INTERVAL=1

# Regular pattern sync
KUPO_PY_PATTERN_SYNC_INTERVAL=60

# DBSync fallback for high availability
DBSYNC_DB_NAME=cexplorer
DBSYNC_HOST=dbsync:5432
DBSYNC_USER=readonly_user
DBSYNC_PASS=secure_password
```

### Running with Multiple Workers

For high-traffic deployments, use multiple workers:

```bash
kupo serve --host 0.0.0.0 --port 1442 --workers 4
```

!!! warning "Worker Considerations"
    Each worker maintains its own health check loop. With many workers, consider increasing `KUPO_PY_CHECK_INTERVAL` to reduce load on Kupo instances.

### Docker Deployment

Example `docker-compose.yml`:

```yaml
version: '3.8'
services:
  kupo-router:
    image: python:3.12-slim
    command: kupo serve --host 0.0.0.0 --port 1442 --workers 2
    ports:
      - "1442:1442"
    environment:
      - KUPO_PY_URL_1=http://kupo1:1442
      - KUPO_PY_URL_2=http://kupo2:1442
      - KUPO_PY_CHECK_INTERVAL=1
      - KUPO_PY_PATTERN_SYNC_INTERVAL=60
    depends_on:
      - kupo1
      - kupo2
```

## Troubleshooting

### No Available Server (503)

**Cause**: All Kupo instances are unhealthy or syncing, and no DBSync fallback is configured.

**Solutions**:

1. Check `/router/status` to see instance health
2. Verify Kupo instances are running and accessible
3. Wait for Kupo instances to finish syncing
4. Configure DBSync fallback for higher availability

### Pattern Sync Not Working

**Cause**: Pattern sync only runs when all instances are healthy.

**Solutions**:

1. Check that all instances are healthy via `/router/status`
2. Ensure no instances are syncing
3. Check server logs for sync errors

### DBSync Fallback Not Working

**Cause**: DBSync dependency not installed or misconfigured.

**Solutions**:

1. Install with `pip install kupo-py[dbsync]`
2. Verify database credentials in environment variables
3. Check server logs for connection errors

