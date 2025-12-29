# Models API Reference

This page documents all the Pydantic data models used in the library. These models are auto-generated from the [Kupo OpenAPI specification](https://cardanosolutions.github.io/kupo/api/nightly.yaml).

## Core Models

### Match

Represents a UTxO match returned by Kupo.

::: kupo.models.Match
    options:
      show_root_heading: false

### Value

Represents the value (ADA and native assets) in a UTxO.

::: kupo.models.Value
    options:
      show_root_heading: false

### Pattern

A matching pattern for addresses, assets, or transactions.

::: kupo.models.Pattern
    options:
      show_root_heading: false

### Point

A point on the blockchain, identified by slot number and block header hash.

::: kupo.models.Point
    options:
      show_root_heading: false

## Block Reference Models

### CreatedAt

Block reference indicating when a transaction was included in the ledger.

::: kupo.models.CreatedAt
    options:
      show_root_heading: false

### SpentAt

Block reference indicating when a UTxO was spent, including spending transaction details.

::: kupo.models.SpentAt
    options:
      show_root_heading: false

## Script and Datum Models

### Script

A native or Plutus script.

::: kupo.models.Script
    options:
      show_root_heading: false

### Datum

A serialized Plutus datum.

::: kupo.models.Datum
    options:
      show_root_heading: false

### Metadata

Transaction metadata with raw CBOR and decoded schema.

::: kupo.models.Metadata
    options:
      show_root_heading: false

## Enums

### Language

Script language types.

::: kupo.models.Language
    options:
      show_root_heading: false

### DatumType

Categorizes the type of datum in an output (hash reference or inline).

::: kupo.models.DatumType
    options:
      show_root_heading: false

## Health and Status Models

### Health

Overview of the Kupo server and connection status.

::: kupo.models.Health
    options:
      show_root_heading: false

### ConnectionStatus

Condition of the connection with the underlying Cardano node.

::: kupo.models.ConnectionStatus
    options:
      show_root_heading: false

### Configuration

Summary of Kupo configuration parameters.

::: kupo.models.Configuration
    options:
      show_root_heading: false

### Indexes

Database query index behavior.

::: kupo.models.Indexes
    options:
      show_root_heading: false

## Pattern Management Models

### RollbackTo

Specifies a point to rollback synchronization to when adding patterns.

::: kupo.models.RollbackTo
    options:
      show_root_heading: false

### ForcedRollback

Request body for forcing a rollback operation.

::: kupo.models.ForcedRollback
    options:
      show_root_heading: false

## Server Mode Models

These models are used when running kupo-py in server mode for load balancing.

### KupoServer

Configuration and health status for a Kupo server instance.

::: kupo.models.KupoServer
    options:
      show_root_heading: false

### DbsyncServer

Configuration for a DBSync server instance used as fallback.

::: kupo.models.DbsyncServer
    options:
      show_root_heading: false

### RoundRobin

Round-robin server selector for load balancing across Kupo and DBSync instances.

::: kupo.models.RoundRobin
    options:
      show_root_heading: false

## Metadata Schema Models

These models represent the structure of transaction metadata.

### MetadatumInt

::: kupo.models.MetadatumInt
    options:
      show_root_heading: false

### MetadatumString

::: kupo.models.MetadatumString
    options:
      show_root_heading: false

### MetadatumBytes

::: kupo.models.MetadatumBytes
    options:
      show_root_heading: false

### MetadatumList

::: kupo.models.MetadatumList
    options:
      show_root_heading: false

### MetadatumMap

::: kupo.models.MetadatumMap
    options:
      show_root_heading: false
