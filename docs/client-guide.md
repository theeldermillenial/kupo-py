# Client Mode Guide

This guide provides a deep dive into using kupo-py's client mode to interact with a Kupo instance. Client mode is the core functionality of kupo-py, providing both synchronous and asynchronous methods to query the Kupo API.

One of the main advantages of using kupo-py is the integration with other native Python tooling such as pycardano. In many cases you can pass in pycardano types (pycardano.Address, pycardano.ScriptHash) into the client methods.

## Overview

The `KupoClient` class provides a complete interface to Kupo's HTTP API:

- **Match queries** - Find UTxOs by address, pattern, policy, or transaction
- **Pattern management** - Add and list indexing patterns
- **Data retrieval** - Fetch scripts, datums, and metadata
- **Health checks** - Monitor Kupo instance status

## Installation

Client mode is included in the base installation:

```bash
pip install kupo-py
```

## Quick Start

```python
from kupo import KupoClient

# Create a client
client = KupoClient("http://localhost", 1442)

# Check health
health = client.health()
print(f"Kupo version: {health.version}")
print(f"Synced: {not health.syncing}")

# Query matches
matches = client.get_matches("addr1qxy2k...")
for match in matches:
    print(f"{match.transaction_id}#{match.output_index}: {match.value.coins} lovelace")
```

## Configuration

### Direct Configuration

Pass the base URL and port directly to the constructor:

```python
from kupo import KupoClient

client = KupoClient(base_url="http://localhost", port=1442)
```

### Environment Variables

The client can read configuration from environment variables:

| Variable | Description |
|----------|-------------|
| `KUPO_BASE_URL` | Base URL of the Kupo instance |
| `KUPO_PORT` | Port number |

```python
import os
os.environ["KUPO_BASE_URL"] = "http://localhost"
os.environ["KUPO_PORT"] = "1442"

# Client will use environment variables
client = KupoClient()
```

You can also use a `.env` file - the client uses `python-dotenv` to load it automatically.

## Synchronous vs Asynchronous

Every method has both synchronous and asynchronous versions:

| Sync Method | Async Method |
|-------------|--------------|
| `get_matches()` | `get_matches_async()` |
| `get_all_matches()` | `get_all_matches_async()` |
| `get_patterns()` | `get_patterns_async()` |
| `get_pattern()` | `get_pattern_async()` |
| `add_pattern()` | `add_pattern_async()` |
| `bulk_add_pattern()` | `bulk_add_pattern_async()` |
| `get_script_by_hash()` | `get_script_by_hash_async()` |
| `get_datum_by_hash()` | `get_datum_by_hash_async()` |
| `get_metadata_by_tx()` | `get_metadata_by_tx_async()` |
| `health()` | `health_async()` |

### Synchronous Usage

Synchronous methods use the `requests` library:

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

# All calls block until complete
health = client.health()
matches = client.get_matches("addr1...")
patterns = client.get_patterns()
```

### Asynchronous Usage

Asynchronous methods use `aiohttp` for non-blocking I/O:

```python
import asyncio
from kupo import KupoClient

async def main():
    client = KupoClient("http://localhost", 1442)
    
    # Use await for async methods
    health = await client.health_async()
    matches = await client.get_matches_async("addr1...")
    
    # Run multiple queries concurrently
    results = await asyncio.gather(
        client.get_matches_async("addr1..."),
        client.get_matches_async("addr2..."),
        client.get_matches_async("addr3...")
    )

asyncio.run(main())
```

## Querying Matches

Matches are UTxOs that Kupo has indexed based on configured patterns.

### Basic Match Query

```python
# Get matches for a specific pattern
matches = client.get_matches("addr1qxy2k...")

# Access match properties
for match in matches:
    print(f"Transaction: {match.transaction_id}")
    print(f"Output index: {match.output_index}")
    print(f"Address: {match.address}")
    print(f"Value: {match.value.coins} lovelace")
    print(f"Created at slot: {match.created_at.slot_no}")
    
    if match.spent_at:
        print(f"Spent at slot: {match.spent_at.slot_no}")
    
    if match.value.assets:
        for asset_id, quantity in match.value.assets.items():
            print(f"Asset {asset_id}: {quantity}")
```

### Query All Matches

Query across all indexed patterns:

```python
# Get all matches (no pattern filter)
all_matches = client.get_all_matches()

# With filters
recent_unspent = client.get_all_matches(
    unspent=True,
    created_after=120000000
)
```

### Filtering Options

#### By Spent Status

```python
# Only unspent UTxOs
unspent = client.get_matches("addr1...", unspent=True)

# Only spent UTxOs
spent = client.get_matches("addr1...", spent=True)
```

!!! warning "Mutually Exclusive"
    You cannot use both `spent=True` and `unspent=True` in the same query.

#### By Creation Time

```python
# Created after a specific slot
matches = client.get_matches(
    "addr1...",
    created_after=120000000
)

# Created before a specific slot
matches = client.get_matches(
    "addr1...",
    created_before=130000000
)

# Created within a range
matches = client.get_matches(
    "addr1...",
    created_after=120000000,
    created_before=130000000
)
```

#### By Spent Time

```python
# Spent after a specific slot
matches = client.get_matches(
    "addr1...",
    spent=True,
    spent_after=120000000
)

# Spent before a specific slot
matches = client.get_matches(
    "addr1...",
    spent=True,
    spent_before=130000000
)
```

!!! note "Spent Time Filters"
    `spent_after` and `spent_before` implicitly filter for spent UTxOs. You don't need to also set `spent=True`.

#### By Assets

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

# Filter by policy ID
matches = client.get_matches(
    "addr1...",
    policy_id="8a1cfae21368b8bebbbed9800fec304e95cce39a2a57dc35e2e3ebaa"
)

# Filter by policy ID and asset name
matches = client.get_matches(
    "addr1...",
    policy_id="8a1cfae21368b8bebbbed9800fec304e95cce39a2a57dc35e2e3ebaa",
    asset_name="4d494c4b"  # Hex-encoded
)
```

#### By Transaction

```python
# Filter by transaction ID
matches = client.get_matches(
    "*",  # Wildcard pattern
    transaction_id="35d8a1c8a66b78f0b9b3e9f2d1c4a5b6c7d8e9f0..."
)

# Get a specific UTxO
matches = client.get_matches(
    "*",
    transaction_id="35d8a1c8...",
    output_index=0
)
```

### Ordering Results

```python
from kupo.client import Order

# Newest first (default)
matches = client.get_matches("addr1...", order=Order.DESC)

# Oldest first
matches = client.get_matches("addr1...", order=Order.ASC)
```

### Resolving Hashes

By default, only hashes are returned for datums and scripts. Use `resolve_hashes=True` to include full values:

```python
matches = client.get_matches("addr1...", resolve_hashes=True)

for match in matches:
    if match.datum_hash:
        print(f"Datum hash: {match.datum_hash}")
        print(f"Datum (resolved): {match.datum}")  # Full datum if available
    
    if match.script_hash:
        print(f"Script hash: {match.script_hash}")
        if match.script:
            print(f"Script language: {match.script.language}")
            print(f"Script bytes: {match.script.script}")
```

## Pattern Management

Patterns determine what UTxOs Kupo indexes. For complete pattern syntax, see the [Kupo Pattern documentation](https://cardanosolutions.github.io/kupo/#section/Patterns).

### Listing Patterns

```python
# Get all patterns
patterns = client.get_patterns()
for pattern in patterns:
    print(pattern.root)

# Check if a specific pattern exists
patterns = client.get_pattern("addr1qxy2k...")
exists = len(patterns) > 0
```

### Adding Patterns

```python
from kupo.client import Limit
from kupo.models import Point

# Add a pattern (rollback to slot 0)
result = client.add_pattern(
    pattern="addr1qxy2k...",
    rollback_to=0
)

# Add with a specific rollback point
result = client.add_pattern(
    pattern="addr1qxy2k...",
    rollback_to=Point(
        slot_no=120000000,
        header_hash="9d09c4de32c70c9c2e3c9d8f3b4a5c6d7e8f9a0b..."
    )
)

# Allow rollback beyond the safe zone
result = client.add_pattern(
    pattern="addr1qxy2k...",
    rollback_to=0,
    limit=Limit.UNSAFE
)
```

### Bulk Pattern Operations

```python
# Add multiple patterns at once
result = client.bulk_add_pattern(
    patterns=[
        "addr1qxy2k...",
        "addr1abc...",
        "addr1def..."
    ],
    rollback_to=0
)
```

### Understanding Rollback

When you add a new pattern, Kupo needs to re-index from a specific point to find matching UTxOs. The `rollback_to` parameter specifies where to start:

- **Slot number**: Kupo rolls back to this slot and re-indexes forward
- **Point object**: Includes header hash for verification
- **Safe zone**: By default, rollback is limited to ~36 hours (129,600 slots)

!!! warning "Unsafe Rollback"
    Using `Limit.UNSAFE` allows rollback beyond the safe zone, but may result in temporarily inconsistent data while re-indexing.

## Data Retrieval

### Scripts

```python
script = client.get_script_by_hash(
    "8a1cfae21368b8bebbbed9800fec304e95cce39a2a57dc35e2e3ebaa"
)

print(f"Language: {script.language}")  # native, plutus:v1, plutus:v2, plutus:v3
print(f"Script bytes: {script.script}")
```

### Datums

```python
datum = client.get_datum_by_hash(
    "cd6a5d31bf9d309706b92ad83402e682fdab9fc889b1b63565ee3de14e09dedf"
)

print(f"Datum CBOR: {datum.datum}")
```

### Transaction Metadata

```python
metadata = client.get_metadata_by_tx(
    "35d8a1c8a66b78f0b9b3e9f2d1c4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2"
)

print(f"Metadata hash: {metadata.hash}")
print(f"Raw CBOR: {metadata.raw}")
print(f"Decoded schema: {metadata.schema_}")
```

## Health Checks

```python
health = client.health()

print(f"Connection: {health.connection_status}")  # connected/disconnected
print(f"Version: {health.version}")
print(f"Syncing: {health.syncing}")
print(f"Latest checkpoint: {health.most_recent_checkpoint}")
print(f"Node tip: {health.most_recent_node_tip}")
print(f"Network sync: {health.network_synchronization}")
print(f"Indexes: {health.configuration.indexes}")  # deferred/installed
```

### Checking Sync Status

```python
health = client.health()

if health.syncing:
    progress = health.network_synchronization or 0
    print(f"Syncing: {progress:.2%} complete")
else:
    print("Fully synchronized")
```

## PyCardano Integration

kupo-py integrates with [PyCardano](https://pycardano.readthedocs.io/) types:

```python
from kupo import KupoClient
from pycardano import Address, ScriptHash, TransactionId, AssetName

client = KupoClient("http://localhost", 1442)

# Use PyCardano Address
addr = Address.from_primitive("addr1qxy2k...")
matches = client.get_matches(pattern=addr)

# Use ScriptHash for policy ID
policy = ScriptHash.from_primitive(
    "8a1cfae21368b8bebbbed9800fec304e95cce39a2a57dc35e2e3ebaa"
)
matches = client.get_matches("addr1...", policy_id=policy)

# Use AssetName
asset = AssetName(b"MILK")
matches = client.get_matches(
    "addr1...",
    policy_id=policy,
    asset_name=asset
)

# Use TransactionId
tx_id = TransactionId.from_primitive("35d8a1c8...")
matches = client.get_matches("*", transaction_id=tx_id)
```

## Timeout Configuration

All methods accept a `timeout` parameter (default: 300 seconds):

```python
# Quick health check with short timeout
health = client.health(timeout=5)

# Long-running query with extended timeout
matches = client.get_all_matches(timeout=600)
```

## Error Handling

### Synchronous Errors

```python
import requests
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

try:
    matches = client.get_matches("addr1...")
except requests.exceptions.ConnectionError:
    print("Could not connect to Kupo")
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e.response.status_code}")
    print(f"Response: {e.response.text}")
```

### Asynchronous Errors

```python
import aiohttp
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

try:
    matches = await client.get_matches_async("addr1...")
except aiohttp.ClientConnectionError:
    print("Could not connect to Kupo")
except asyncio.TimeoutError:
    print("Request timed out")
except aiohttp.ClientResponseError as e:
    print(f"HTTP error: {e.status}")
```

## Response Types

### MatchResponse

Returned by `get_matches()` and `get_all_matches()`. Behaves like a list:

```python
matches = client.get_matches("addr1...")

# List-like operations
print(f"Found {len(matches)} matches")
first_match = matches[0]

for match in matches:
    print(match.transaction_id)
```

### PatternResponse

Returned by `get_patterns()`, `get_pattern()`, `add_pattern()`, and `bulk_add_pattern()`:

```python
patterns = client.get_patterns()

print(f"Found {len(patterns)} patterns")

for pattern in patterns:
    print(pattern.root)
```

## Best Practices

### Connection Reuse

For multiple queries, reuse the same client instance:

```python
# Good - reuse client
client = KupoClient("http://localhost", 1442)
for addr in addresses:
    matches = client.get_matches(addr)

# Avoid - creating new client for each query
for addr in addresses:
    client = KupoClient("http://localhost", 1442)  # Unnecessary
    matches = client.get_matches(addr)
```

### Async for High Throughput

Use async methods with `asyncio.gather()` for parallel queries:

```python
import asyncio

async def query_multiple_addresses(addresses):
    client = KupoClient("http://localhost", 1442)
    
    tasks = [client.get_matches_async(addr) for addr in addresses]
    results = await asyncio.gather(*tasks)
    
    return dict(zip(addresses, results))
```

### Filter Early

Apply filters in the query rather than filtering results in Python:

```python
# Good - filter in query
unspent = client.get_matches(
    "addr1...",
    unspent=True,
    created_after=120000000
)

# Avoid - filtering after fetch
all_matches = client.get_matches("addr1...")
unspent = [m for m in all_matches if m.spent_at is None]
```

