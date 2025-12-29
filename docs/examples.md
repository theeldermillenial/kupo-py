# Examples

This page provides examples for common use cases with kupo-py.

## Basic Usage

### Synchronous Client

```python
from kupo import KupoClient

# Initialize the client
client = KupoClient(base_url="http://localhost", port=1442)

# Get matches for an address
matches = client.get_matches("addr1...")

# Get all patterns
patterns = client.get_patterns()

# Check API health
health = client.health()
print(f"Connected: {health.connection_status}")
print(f"Syncing: {health.syncing}")
```

### Asynchronous Client

```python
import asyncio
from kupo import KupoClient

async def main():
    client = KupoClient(base_url="http://localhost", port=1442)

    # Get matches for an address
    matches = await client.get_matches_async("addr1...")
    
    # Get all patterns
    patterns = await client.get_patterns_async()
    
    # Check API health
    health = await client.health_async()
    print(f"Connected: {health.connection_status}")

asyncio.run(main())
```

### Using Environment Variables

```python
from kupo import KupoClient

# The client will automatically use KUPO_BASE_URL and KUPO_PORT from environment
client = KupoClient()

# Or specify them explicitly
client = KupoClient(
    base_url="http://localhost",
    port=1442
)
```

## Querying Matches

### Basic Match Query

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

# Get matches for a specific address
matches = client.get_matches("addr1qxy2k...")

# Iterate over results
for match in matches:
    print(f"TX: {match.transaction_id}")
    print(f"Output Index: {match.output_index}")
    print(f"Value: {match.value.coins} lovelace")
    print(f"Created at slot: {match.created_at.slot_no}")
```

### Filtering Matches

```python
from kupo import KupoClient
from kupo.client import Order

client = KupoClient("http://localhost", 1442)

# Get only unspent UTxOs
unspent = client.get_matches(
    pattern="addr1qxy2k...",
    unspent=True
)

# Get only spent UTxOs
spent = client.get_matches(
    pattern="addr1qxy2k...",
    spent=True
)

# Filter by creation time (slot number)
recent = client.get_matches(
    pattern="addr1qxy2k...",
    created_after=120000000,  # After this slot
    created_before=130000000  # Before this slot
)

# Filter by when UTxOs were spent
spent_recently = client.get_matches(
    pattern="addr1qxy2k...",
    spent=True,
    spent_after=120000000
)

# Order results (oldest first)
ordered = client.get_matches(
    pattern="addr1qxy2k...",
    order=Order.ASC
)

# Order results (newest first - default)
ordered = client.get_matches(
    pattern="addr1qxy2k...",
    order=Order.DESC
)
```

### Filtering by Assets

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

# Filter by policy ID
matches = client.get_matches(
    pattern="addr1qxy2k...",
    policy_id="8a1cfae21368b8bebbbed9800fec304e95cce39a2a57dc35e2e3ebaa"
)

# Filter by policy ID and asset name
matches = client.get_matches(
    pattern="addr1qxy2k...",
    policy_id="8a1cfae21368b8bebbbed9800fec304e95cce39a2a57dc35e2e3ebaa",
    asset_name="4d494c4b"  # Hex-encoded asset name
)

# Access asset values from results
for match in matches:
    if match.value.assets:
        for asset_id, quantity in match.value.assets.items():
            print(f"Asset: {asset_id}, Quantity: {quantity}")
```

### Filtering by Transaction

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

# Get a specific UTxO by transaction ID and output index
matches = client.get_matches(
    pattern="*",  # Wildcard pattern
    transaction_id="35d8a1c8a66b78f0b9b3e9f2d1c4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2",
    output_index=0
)
```

### Get All Matches (No Pattern Filter)

```python
from kupo import KupoClient
from kupo.client import Order

client = KupoClient("http://localhost", 1442)

# Get all matches across all patterns
all_matches = client.get_all_matches(
    unspent=True,
    order=Order.DESC
)

# With async
all_matches = await client.get_all_matches_async(
    created_after=120000000,
    order=Order.ASC
)
```

### Resolving Datum and Script Hashes

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

# Include resolved datums and scripts in the response
matches = client.get_matches(
    pattern="addr1qxy2k...",
    resolve_hashes=True
)

for match in matches:
    if match.datum:
        print(f"Datum: {match.datum}")
    if match.script:
        print(f"Script language: {match.script.language}")
        print(f"Script bytes: {match.script.script}")
```

## Working with Scripts and Datums

### Get Script by Hash

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

script = client.get_script_by_hash(
    "8a1cfae21368b8bebbbed9800fec304e95cce39a2a57dc35e2e3ebaa"
)

print(f"Language: {script.language}")  # e.g., "plutus:v2"
print(f"Script: {script.script}")  # Hex-encoded script bytes
```

### Get Datum by Hash

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

datum = client.get_datum_by_hash(
    "cd6a5d31bf9d309706b92ad83402e682fdab9fc889b1b63565ee3de14e09dedf"
)

print(f"Datum: {datum.datum}")  # CBOR-encoded datum
```

### Get Transaction Metadata

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

metadata = client.get_metadata_by_tx(
    "35d8a1c8a66b78f0b9b3e9f2d1c4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2"
)

print(f"Hash: {metadata.hash}")
print(f"Raw CBOR: {metadata.raw}")
print(f"Schema: {metadata.schema_}")
```

## Managing Patterns

Patterns determine what UTxOs Kupo indexes. For detailed information on pattern syntax and formats, see the [Kupo Pattern Matching documentation](https://cardanosolutions.github.io/kupo/#section/Patterns).

### Get All Patterns

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

patterns = client.get_patterns()

for pattern in patterns:
    print(f"Pattern: {pattern.root}")
```

### Get Specific Pattern

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

# Check if a specific pattern exists
patterns = client.get_pattern("addr1qxy2k...")

if len(patterns) > 0:
    print("Pattern is registered")
```

### Add a Pattern

```python
from kupo import KupoClient
from kupo.client import Limit
from kupo.models import Point

client = KupoClient("http://localhost", 1442)

# Add a pattern with rollback to a specific slot
result = client.add_pattern(
    pattern="addr1qxy2k...",
    rollback_to=1000000  # Slot number
)

# Add with a Point object (includes header hash for verification)
result = client.add_pattern(
    pattern="addr1qxy2k...",
    rollback_to=Point(
        slot_no=1000000,
        header_hash="9d09c4de32c70c9c2e3c9d8f3b4a5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b"
    )
)

# Allow rollback beyond the safe zone (36 hours)
result = client.add_pattern(
    pattern="addr1qxy2k...",
    rollback_to=1000000,
    limit=Limit.UNSAFE
)
```

### Add Multiple Patterns

```python
from kupo import KupoClient

client = KupoClient("http://localhost", 1442)

# Add multiple patterns at once
result = client.bulk_add_pattern(
    patterns=[
        "addr1qxy2k...",
        "addr1abc...",
        "addr1def..."
    ],
    rollback_to=1000000
)

print(f"Added {len(result)} patterns")
```

## Using PyCardano Types

kupo-py integrates with PyCardano for type-safe address and hash handling:

```python
from kupo import KupoClient
from pycardano import Address, ScriptHash, TransactionId

client = KupoClient("http://localhost", 1442)

# Use PyCardano Address
addr = Address.from_primitive(
    "addr1qxy2k..."
)
matches = client.get_matches(pattern=addr)

# Use ScriptHash for policy ID
policy = ScriptHash.from_primitive(
    "8a1cfae21368b8bebbbed9800fec304e95cce39a2a57dc35e2e3ebaa"
)
matches = client.get_matches(
    pattern="addr1qxy2k...",
    policy_id=policy
)

# Use TransactionId
tx_id = TransactionId.from_primitive(
    "35d8a1c8a66b78f0b9b3e9f2d1c4a5b6c7d8e9f0a1b2c3d4e5f6a7b8c9d0e1f2"
)
matches = client.get_matches(
    pattern="*",
    transaction_id=tx_id
)
```

## Utility Functions

kupo-py provides utility functions for working with Cardano slots and timestamps:

```python
from kupo.utils import slot_to_timestamp, timestamp_to_slot

# Convert a slot number to a POSIX timestamp
slot = 120000000
timestamp = slot_to_timestamp(slot)
print(f"Slot {slot} = timestamp {timestamp}")

# Convert a POSIX timestamp to a slot number
import time
current_time = int(time.time())
slot = timestamp_to_slot(current_time)
print(f"Current slot: {slot}")
```

## Server Mode

When running kupo-py in server mode, you can interact with the load balancer:

### Check Router Status

```python
import httpx

# Get status of all backend servers
response = httpx.get("http://localhost:1442/router/status")
status = response.json()

print("Kupo servers:")
for server in status["kupo_servers"]:
    print(f"  {server['url']}: {server['health']}")

print("DBSync servers:")
for url in status["dbsync_servers"]:
    print(f"  {url}")
```

### Force a Specific Backend

```python
import httpx

# Force request to go to DBSync
response = httpx.get(
    "http://localhost:1442/matches/addr1qxy2k...",
    headers={"X-Force-Backend": "dbsync"}
)

# Check which backend handled the request
source = response.headers.get("X-Kupo-Source")
print(f"Request handled by: {source}")

# Force request to go to Kupo (skip DBSync)
response = httpx.get(
    "http://localhost:1442/matches/addr1qxy2k...",
    headers={"X-Force-Backend": "kupo"}
)
```

## Error Handling

```python
from kupo import KupoClient
import requests

client = KupoClient("http://localhost", 1442)

try:
    matches = client.get_matches("addr1qxy2k...")
except requests.exceptions.ConnectionError:
    print("Could not connect to Kupo instance")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error: {e.response.status_code}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Async Error Handling

```python
from kupo import KupoClient
import aiohttp

client = KupoClient("http://localhost", 1442)

try:
    matches = await client.get_matches_async("addr1qxy2k...")
except aiohttp.ClientConnectionError:
    print("Could not connect to Kupo instance")
except aiohttp.ClientResponseError as e:
    print(f"HTTP error: {e.status}")
except Exception as e:
    print(f"Unexpected error: {e}")
```
