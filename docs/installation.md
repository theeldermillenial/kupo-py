# Installation Guide

## Requirements

- Python 3.10 or higher
- pip (Python package installer)

## Installation

The simplest way to install kupo-py is using pip:

```bash
pip install kupo-py
```

## Development Installation

For development purposes, you can clone the repository and install in editable mode:

```bash
git clone https://github.com/yourusername/kupo-py.git
cd kupo-py
pip install -e .
```

## Environment Variables

kupo-py can be configured using environment variables:

- `KUPO_BASE_URL`: The base URL of your Kupo instance
- `KUPO_PORT`: The port number of your Kupo instance

You can set these variables in your environment or use a `.env` file:

```bash
KUPO_BASE_URL=http://localhost
KUPO_PORT=1442
``` 