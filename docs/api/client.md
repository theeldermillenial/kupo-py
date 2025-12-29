# Client API Reference

This page documents the client module, including the main client class, enums, and response models.

## KupoClient

The main client class for interacting with a Kupo instance.

::: kupo.client.KupoClient
    options:
      show_root_heading: false
      show_source: true
      members_order: source

## Enums

### Order

Controls the ordering of results by time.

::: kupo.client.Order
    options:
      show_root_heading: false
      members_order: source

### Limit

Controls rollback behavior when adding patterns.

::: kupo.client.Limit
    options:
      show_root_heading: false
      members_order: source

## Query Models

### MatchQuery

Validator model for match query parameters. This is used internally but can also be passed directly to match methods.

::: kupo.client.MatchQuery
    options:
      show_root_heading: false
      show_source: true

## Response Models

### MatchResponse

Response model for endpoints that return matches. Behaves like a list of `Match` objects.

::: kupo.client.MatchResponse
    options:
      show_root_heading: false
      show_source: true

### PatternResponse

Response model for endpoints that return patterns. Behaves like a list of `Pattern` objects.

::: kupo.client.PatternResponse
    options:
      show_root_heading: false
      show_source: true
