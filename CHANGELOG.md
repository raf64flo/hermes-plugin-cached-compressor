# Changelog

## 0.2.0

- Fall back to the built-in compressor on routes that do not report cache reads, instead of replaying
  the full prefix at input price (`cacheability.py`).
- Declare the `pre_api_request` hook in `plugin.yaml`, so `hermes plugins validate` passes.
- English README, MIT licence file, CI workflow.

## 0.1.0

- Context engine that summarizes from the session prefix and its own model: payload capture through
  `pre_api_request`, window alignment by index then by sequence, session-scoped payload slot,
  cache-read telemetry (`cache_read_tokens`, `cache_read_ratio`).
