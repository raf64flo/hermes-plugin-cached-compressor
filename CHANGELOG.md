# Changelog

## Unreleased

- `plugin.yaml`: the version gate now uses `requires_hermes: ">=0.23.0"`, the field Hermes actually
  reads. The previous `hermes.min_version` block was never loaded (verified against the manifest
  parser: the `hermes` key is reserved but unconsumed for native `plugin.yaml` files), so the gate
  did not exist; on an older core the plugin loaded and the `register()` contract check was the
  only protection.
- README: "Should you install it?" decision section, built on the measured break-even; the tested
  range (0.23.0 up to `main`, CI matrix) is stated where the version gate is documented.
- Release: tags `v0.3.0` and `v0.4.0` added; changelog sections dated.
- Cacheability guard hardening: `post_api_request` now feeds the guard from the main loop's usage,
  so a backend that never serves from cache is recognised before the first compaction, at zero cost
  in calls (the plugin's own summary calls go through the auxiliary path, no feedback loop).
- `min_cache_ratio` setting: a route whose observed hit rate over its last eight large measurements
  is below the threshold is denied the prefix mode even without strikes. Unset by default.
- `force_builtin` setting: permanent fallback to the built-in compressor, for backends known not to
  cache. Unset by default; a clear log line marks the forced fallback.
- README: new "Tuning" section; test count and CI policy brought up to date.
- `probes/`: the report probe honors `retry-after` on a 429, and `cache_write_sweep.py` measures
  the cache write window. Measured on the reference router (report in
  `docs/results/cache-write-sweep-20261110.json`): the prefix is readable on the second call of the
  same turn (97 % with no idle), and a 10 s idle makes the hit *worse* (0, 93, 0, 0, 0, 93 %), so
  the window is short and irregular. Verdict: no delay knob in the compaction path, and the README
  says so next to the two settings that do exist.

## 0.4.0 (2026-11-09)

- State: `on_session_end` releases the session's slot. It used to live until LRU eviction, holding a
  whole conversation in memory after the session had ended.
- State: the budget is now counted in bytes as well as in sessions (64 MiB, 8 sessions). A single
  oversized transcript is kept rather than evicted, since dropping it would free nothing useful.
- State: messages are copied one level deep on capture, so the build-in pruner rewriting `content` in
  place no longer changes what the slot returns. `tool_calls` stay shared, documented.
- `contract.py`: the private Hermes symbols the plugin needs are declared, and `register()` refuses to
  install the engine when one is missing or has changed signature, keeping the built-in compressor with
  an explicit message. `tests/test_contract.py` fails at the first version that moves, and CI runs the
  suite on the declared minimum plus `main` (allowed to fail).
- Usage: cache reads are read from every field in use (`prompt_tokens_details.cached_tokens`,
  `prompt_cache_hit_tokens`, `cache_read_input_tokens`, `cached_content_token_count`), so a DeepSeek,
  Anthropic or Gemini backend is recognised instead of being taken for a route without cache.
- `probes/cache_report_check.py` and `docs/PROVIDERS.md`: check a backend in a minute, and know whether
  the gain is money or prefill time.

## 0.3.0 (2026-11-09)

- `bench.py --aux-model`: run the built-in arm on a cheaper model, as `auxiliary.compression` does in a
  deployment, while the plugin keeps the session model. The run records the provider's published
  prices, and `report.py` prints the break-even ratio below which the plugin still saves money.
- `bench.py --repeats` now gives each repeat its own transcript. Before, the second repeat found both
  arms already cached, which flattered the built-in.
- `report.py`: median, 95 % bootstrap interval and served / not-served split per size, and separate
  tables per routing configuration.
- `docs/BENCHMARK.md`: new section on a realistic auxiliary model. Against `mimo-v2.6-flash` the
  plugin costs 1.9 to 2.6 times more and scores better; the break-even sits at an auxiliary price
  ratio of 0.7 to 0.8, so the cost saving needs a session model cheaper than its auxiliary.

## 0.2.2 (2026-11-09)

- Docs: the cache-read multiplier was stated as 10 % of the input price; the bills say 2 %
  (`docs/BENCHMARK.md`, `README.md`), so the worked example drops from 0.092 to 0.022.
- Docs: five rows served per size, not six; seven pinned passes per size, not six; the built-in's
  14 039 characters come from tool-result pruning, not from the `_SUMMARY_INPUT_MAX_CHARS` bound.
- Docs: three claims corrected against the record (the built-in is not always posted cold, the losing
  run was served at 59 %, the judge's behaviour does not track transcript size), plus the cost spread
  and median beside the mean.
- The cache-less fallback message is now English, as the README already quoted it.
- `bench/report.py`: the `pass` column shows date and time, not the date alone.

## 0.2.1 (2026-11-09)

- Head-to-head benchmark against the built-in compressor (`bench/bench.py`, 8 runs): same model, same
  transport, billed cost, blind rubric judge, warm-cache guard.
- Input-only variant (`bench/input_only.py`), output pinned at 256 tokens for both engines, so the
  bill isolates the prompt from the length of the summary.
- `bench/report.py` regenerates the published tables from the raw records.
- Protocol, results and caveats: `docs/BENCHMARK.md`.
- README: install commands point at the published repository.

## 0.2.0 (2026-11-09)

- Fall back to the built-in compressor on routes that do not report cache reads, instead of replaying
  the full prefix at input price (`cacheability.py`).
- Declare the `pre_api_request` hook in `plugin.yaml`, so `hermes plugins validate` passes.
- English README, MIT licence file, CI workflow.

## 0.1.0 (2026-11-09)

- Context engine that summarizes from the session prefix and its own model: payload capture through
  `pre_api_request`, window alignment by index then by sequence, session-scoped payload slot,
  cache-read telemetry (`cache_read_tokens`, `cache_read_ratio`).
