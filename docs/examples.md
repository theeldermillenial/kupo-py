# Examples

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
```

### Asynchronous Client

```python
import asyncio
from kupo import KupoClient

async def main():
    async with KupoClient(base_url="http://localhost", port=1442) as client:
        # Get matches for an address
        matches = await client.get_matches_async("addr1...")
        
        # Get all patterns
        patterns = await client.get_patterns_async()
        
        # Check API health
        health = await client.health_async()

# Run the async code
asyncio.run(main())
```

## Using Environment Variables

```python
from kupo import KupoClient
import os

# The client will automatically use KUPO_BASE_URL and KUPO_PORT from environment
client = KupoClient()

# Or specify them explicitly
client = KupoClient(
    base_url=os.getenv("KUPO_BASE_URL", "http://localhost"),
    port=int(os.getenv("KUPO_PORT", "1442"))
)
``` 