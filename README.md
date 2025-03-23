# kupo-py

[![PyPI version](https://badge.fury.io/py/kupo-py.svg)](https://badge.fury.io/py/kupo-py)
[![GitHub Actions Status](https://github.com/theeldermillenial/kupo-py/actions/workflows/publish.yml/badge.svg)](https://github.com/theeldermillenial/kupo-py/actions)
[![Downloads](https://static.pepy.tech/badge/kupo-py)](https://pepy.tech/project/kupo-py)
[![Monthly Downloads](https://static.pepy.tech/badge/kupo-py/month)](https://pepy.tech/project/kupo-py)

A Python client for interacting with [Kupo](https://cardanosolutions.github.io/kupo/), a lightweight and fast Cardano blockchain indexer. This client provides a simple and intuitive interface to query and interact with Kupo's HTTP API.

## Features

- Asynchronous HTTP client for optimal performance
- Type-safe responses using Pydantic models
- Comprehensive API coverage
- Easy-to-use interface for common Kupo operations

## Installation

```bash
pip install kupo-py
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

### Using the Client

```python
from kupo import KupoClient
from kupo.models import Order, Limit, Point

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

#### Pattern Matching

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
- `addr1*` - Match all addresses starting with "addr1"

For detailed pattern matching syntax and examples, see the [Kupo Pattern Matching documentation](https://cardanosolutions.github.io/kupo/patterns/).

## Examples

### Querying Matches

```python
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
# Get all patterns
patterns = await client.get_patterns_async()
print(f"Active patterns: {patterns}")

# Add a new pattern
pattern = await client.add_pattern_async(
    pattern="addr1qxy2k...",
    rollback_to=Point(slot_no=1000000),  # Rollback to specific slot
    limit=Limit.SAFE
)

# Add multiple patterns at once
patterns = await client.bulk_add_pattern_async(
    patterns=["addr1qxy2k...", "addr1qxy2k..."],
    rollback_to={"slot_no": 1000000}
)
```

## API Reference

For detailed API documentation, please refer to the [Kupo documentation](https://cardanosolutions.github.io/kupo/).

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
- Code formatting (black)
- Import sorting (isort)
- Type checking (mypy)
- Linting (ruff)
- And other quality checks

Please feel free to submit a Pull Request!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
