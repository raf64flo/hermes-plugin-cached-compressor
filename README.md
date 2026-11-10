# cached-compressor

A Hermes context engine that compacts a session at the provider's cache-read price, using the
session's own model instead of a fixed auxiliary one.

## The problem

Hermes' built-in compressor serializes a window of the conversation into a fresh text prompt and
sends it to the configured auxiliary model. The main provider never sees that request, so:

- the summary call is billed as a brand new prompt, at full input price;
- the main loop's prefix cache is left untouched, so the next turn repays prefill;
- the summary is written by whatever model sits in `auxiliary.compression`, which may not be the
  model that actually holds the conversation.

## The idea

The session prefix is already cached at the provider. Instead of sending a re-serialized copy of the
conversation, send the real prefix, truncated at the compaction boundary, followed by a summary
instruction. Only the output tokens are billed at full price; the prefix is billed as a cache read.

The prefix must be byte-identical to what the main loop sent. Rebuilding a transcript by hand breaks
the cache: stripping content, canonicalizing `tool_calls` arguments, or prepending the system prompt
at send time is enough to miss. The plugin therefore captures the payload that was just sent, through
the `pre_api_request` hook, and reuses it.

## Should you install it?

Three questions, all answerable with the numbers further down:

1. Does your backend report cache reads? The one-minute check is
   `python probes/cache_report_check.py`. If it reports nothing, the prefix mode can only cost,
   and the plugin steps aside on its own, so there is nothing to gain from installing.
2. What is `R`, the session model's price over the auxiliary model's? The measured break-even is
   `R_max = 0.7-0.8`, so the plugin saves money only when the session model is cheaper than its
   auxiliary, which is rare:
   - `R ~ 1` (both engines on the same model): the plugin saves 32 % of billed cost and 45 % of
     latency, and the summary is about as good (19.6/25 against 19.9/25 on the rubric judge).
   - `R > 1` (a cheaper auxiliary, the usual deployment): the plugin costs 1.9 to 2.6 times more
     but scores 24.2/25 against 15.2/25. It buys a better summary, not a lighter bill.
3. How long is the transcript? Cache amortisation grows with size: with output pinned, the
   summary call costs 0.40 of the built-in's at 20k and 0.78 at 100k when the prefix is served,
   and 2.7 to 3.0 times as much when it is not. On short sessions the built-in's small prompt
   wins by default.

Net: same-priced model and long sessions, install; cheap auxiliary, only for the summary quality;
no cache reporting, do not.

## Measured results

Provider with prefix-cache reporting, 20 987-token history, 162 messages, 12-message protected tail,
payload built by Hermes' own assembly path (`_iteration_summary_api_messages` then
`_build_api_kwargs_for_mode`):

    request                         prompt   cached    hit
    W  writer, real path            20 987        0   0.0 %   cold, expected
    A  reader, real path            19 690   19 584  99.5 %   design validated
    B  reader, hand-rebuilt         19 803        0   0.0 %   why rebuilding is banned

    effective A = 106 + 0.02 x 19 584 = 498, against 19 690 + 2 656 = 22 346 tokens at full price, ratio 0.022
    (0.02 is this provider's cache-read multiplier, derived from its own bills; substitute your own)

Full round trip on a real session, through `CachedCompressor._call_summary_llm`:

    writer  prompt=20987  cached=20864  hit=99.4%
    reader  prompt=21267  cached=21248  hit=99.91%  cost=0.00166 USD

Real compaction of a 161-message session: 161 down to 103 messages, one auxiliary call,
11 351 prompt tokens of which 11 264 were cache reads (99.2 %).

## Head-to-head benchmark

Same model, same transport, both engines running their real `compress()`: eight runs over ~20 000 and
~100 000-token transcripts, billed cost read from the router's own `usage` field.

    arm          cost USD        latency      input tokens   cache hit   summary size
    built-in     0.0036-0.0092   31-148 s     5 863-8 157    0 %         baseline
    plugin       0.0027-0.0069   27.6-79.2 s  26 017-44 585   0-96 %      half the size

    mean over the eight runs: -32 % billed cost, -45 % latency
    the four runs where the provider served the prefix: -48 % billed cost

Read that table with its two caveats. The plugin sends 3.2 to 7.4 times more input tokens and pays
for them at the cache price, so the gain depends on the provider actually serving the prefix from
cache. The one losing run was served at 59 % and cost 15.6 % more; the two runs where nothing was
served at all still ended 27 % and 2 % cheaper, because the built-in wrote two to three times more
output. And its summary comes out about half as long, which the rubric judge cannot score apart from
the built-in's: 19.6/25 against 19.9/25, with half the verdicts flipping when the labels are swapped.

A second experiment pins the output at 256 tokens for both engines, so the bill isolates the prompt:
the plugin's call then costs 0.40 of the built-in's at 20k and 0.78 at 100k when the prefix is
served, and 2.7 to 3.0 times as much when it is not.

Both of those hand the built-in arm the session model, which isolates the cache. A real deployment
does not: the built-in calls `auxiliary.compression`, usually a cheaper model, and the picture
inverts. With `mimo-v2.6-flash` as the auxiliary (0.14/0.28 USD per million against 0.15/0.6), the
plugin costs 1.9 to 2.6 times more and scores 24.2/25 against 15.2/25. The break-even is an auxiliary
price ratio of 0.7 to 0.8, so the plugin saves money only when the session model is cheaper than its
auxiliary, which is rare. Against a cheap auxiliary it buys a better summary, not a cheaper bill.

Protocol, per-run tables, why the hit is capped near 96 %, the summary-input coverage of both
engines, and the reproduction commands: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## When it does not help

The design assumes the provider reports cache reads in `usage.prompt_tokens_details.cached_tokens`.
Some backends return no such field at all. There, the prefix is billed at full price, and resending
68 000 tokens costs more than the built-in compressor's serialized window.

The plugin measures instead of assuming. A route that comes back twice without any cache read, on a
prompt large enough to have been cached, is marked as cache-less for the rest of the process, and
compaction falls back to the built-in path. One observation is not enough: the provider may simply
never have seen that exact prefix. Cacheless behaviour is logged:

    cached-compressor: route without confirmed cache, falling back to the built-in compressor

## Privacy and memory

The plugin keeps the last request payload per session, because that byte-identical copy is what makes
the cache hit possible. What that implies:

- Only the `messages` and `tools` of the last request are kept, for at most 8 sessions and 64 MiB of
  estimated bytes; past either limit the oldest session is dropped. `on_session_end` clears the slot
  of the session that closes.
- No message content, no tool output, no summary text reaches the logs or the telemetry. The plugin
  logs routes (`provider=... model=...`) and counters (`prompt=... cached=...`); its telemetry entry
  carries `cache_read_ratio` and nothing else.
- Messages are copied shallowly on capture, so a later in-place edit of the live history (the built-in
  pruner rewrites the `content` of old tool results) does not change what the slot returns. Nested
  values such as `tool_calls` stay shared with the live history: deep-copying a session transcript
  costs more than the guarantee is worth.
- The slot is process-wide by design: two imports of the package (context engine and ordinary plugin)
  resolve to the same `sys.modules` holder, so the hook and the engine never write to different slots.

`tests/test_state_privacy.py` holds those four properties: no content in the logs, release at session
end, byte budget, and one session cannot read another's thread.

## Requirements

- a provider with prefix caching, and a usage field reporting the cache read. `docs/PROVIDERS.md` lists
  the fields read, the one-minute check, and what changes on a local backend;
- Hermes `0.23.0` or later, with `plugins.isolation: in_process` (the default). The `host` isolation
  mode is incompatible: the context engine subclasses Hermes' own compressor class, which cannot be
  imported across an isolation boundary.

### Compatibility policy

The plugin subclasses Hermes' compressor and posts its summary through `call_llm`, so it depends on
symbols Hermes does not publish (`contract.py` lists them). Rather than hope, `register()` checks their
presence and signature at load time and keeps the built-in compressor with an explicit message if one
is missing, and `tests/test_contract.py` fails at the first version that moves. CI runs the suite
against the declared minimum and against `main`, the latter allowed to fail. Upstreaming a public seam
for context engines is the only real fix; until then, pin Hermes to a version you have tested.

## Install

From a git checkout:

    git clone https://github.com/raf64flo/hermes-plugin-cached-compressor
    ln -s "$PWD/hermes-plugin-cached-compressor" ~/.hermes/plugins/cached-compressor

Or let Hermes clone it:

    hermes plugins install raf64flo/hermes-plugin-cached-compressor

Then enable both entry points. They are both required: the engine does the compaction, the plugin
registers the hook that captures the payload.

    hermes config set context.engine cached-compressor
    hermes plugins enable cached-compressor

Check it:

    hermes config get context.engine     # cached-compressor
    hermes plugins list                  # cached-compressor enabled

## Tuning

Nothing is tuned by default; the base policy is: unknown route runs the prefix mode, and the
observation decides. One observation is always on; two settings are optional, read from
`plugins.entries.cached-compressor.settings` at load time, and ignored when unreadable:

- **Free pre-diagnostic.** The plugin also registers `post_api_request`, which receives the main
  loop's usage with the cache read already normalized (`cache_read_tokens` covers the OpenAI,
  Anthropic and DeepSeek spellings). A large prompt answered with zero cache read is a strike on the
  route, at zero cost in calls, so a backend that does not cache is recognised before the first
  compaction instead of paying one full-price transcript to find out. The plugin's own summary calls
  go through the auxiliary path, so there is no feedback loop.
- `min_cache_ratio: 0.9` — profitability threshold, in `0..1`. When set, a route whose observed
  hit rate over its last eight large measurements is below the threshold is denied the prefix mode
  even without strikes. No threshold (the default) keeps the base policy: strikes only.
- `force_builtin: true` — permanent fallback to the built-in compressor, for backends known not to
  cache (a local vLLM, for instance): the prefix can only cost, and the built-in prompt is small.

    hermes config set plugins.entries.cached-compressor.settings.min_cache_ratio 0.9
    hermes config set plugins.entries.cached-compressor.settings.force_builtin true

There is no sleep or delay setting, and that is a measured decision, not an omission: the write
window probe (`probes/cache_write_sweep.py`, 0/2/5/10 s) shows the prefix is readable on the second
call of the same turn and that idling ten seconds makes the hit *worse*, not better. Details and
the table: `docs/PROVIDERS.md`, "Cache write window".

## Design

Two entry points in one package, both through `register(ctx)`:

- context engine (`context.engine: cached-compressor`): `CachedCompressor` subclasses the built-in
  compressor and changes only the summary request;
- regular plugin (`plugins.enabled`): registers `pre_api_request`, which receives the final wire
  payload after assembly, sanitizers and the cache plan (`agent/turn_api_request.py`), and
  `post_api_request`, which feeds the route cacheability guard from the main loop's usage.

`state.PayloadSlot` keeps the last observed payload with its route (provider, model, base URL,
API mode), per session, so the summary runs with the model that actually answered in that session.
Session rotation during compaction happens after the summary call, so the captured thread still
matches.

Because the package is loaded twice, under two module names, the payload slot and the route cache
tracker live in a `sys.modules` sentinel module (`state._shared_holder`). Otherwise the hook would
write to a slot the engine never reads.

## Tests

    HERMES_SOURCE=~/.hermes/hermes-agent python -m pytest tests -q

83 tests, no network, no API key. `HERMES_SOURCE` points at a Hermes source checkout (the plugin
imports `agent.*`). CI runs the suite against the declared minimum (`v0.23.0`), the current release
(`v0.23.1`), and `main`, the latter allowed to fail. The bench probes' guards (anonymizer canary,
leak check, self-tests) run in the same suite.

Catalog validation is a manual pre-release check, because the CLI needs a fully installed Hermes:

    hermes plugins validate .

Live probes, which do need a provider key, are in `probes/`:

    HERMES_API_KEY=... python probes/cache_hit_live.py     # writer + reader, real assembly path
    HERMES_API_KEY=... python probes/compress_live.py      # full compress(), slot and telemetry

Two experiments have their own entry points, both with a network-free `--self-test`:

    BENCH_API_KEY=... python bench/fact_recall.py --tokens 20000   # which facts survive compaction
    BENCH_API_KEY=... python bench/hetero_bench.py --sessions 4    # real, masked local transcripts
    python bench/anonymize.py --canary                             # the masking guard, no network

## Known limitations

- The plugin relies on parts of Hermes that are not a public plugin surface: the compressor base
  class, `agent.chat_completion_helpers._iteration_summary_api_messages`,
  `agent.auxiliary_client.call_llm`. `hermes plugins validate` reports them as warnings. A Hermes
  upgrade can break it: `requires_hermes: ">=0.23.0"` in `plugin.yaml` makes Hermes itself refuse
  to load the plugin on an older core, and the `register()` contract check keeps compaction working
  (built-in compressor, explicit log line) if a private symbol has moved.
- Cache accounting is only as good as the provider's usage payload. On providers that do not report
  cache reads, the plugin steps aside (see above).

Tested range: 0.23.0 (declared minimum) and 0.23.1 (both in CI), up to the current `main` (CI,
allowed to fail). There is no manifest field for a tested maximum, so the CI matrix is the record.
0.23.1 restructured `ContextCompressor` into mixins (`SummaryDispatchMixin`, `MicroCompactionMixin`
and others) without moving `compress(messages)`, and kept the `pre_api_request` / `post_api_request`
payload keys the plugin reads, `usage.cache_read_tokens` included.
