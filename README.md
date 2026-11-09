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

## Measured results

Provider with prefix-cache reporting, 20 987-token history, 162 messages, 12-message protected tail,
payload built by Hermes' own assembly path (`_iteration_summary_api_messages` then
`_build_api_kwargs_for_mode`):

    request                         prompt   cached    hit
    W  writer, real path            20 987        0   0.0 %   cold, expected
    A  reader, real path            19 690   19 584  99.5 %   design validated
    B  reader, hand-rebuilt         19 803        0   0.0 %   why rebuilding is banned

    effective A = 106 + 0.1 x 19 584 = 2 064 against 22 346 at full price, ratio 0.092
    (0.1 is this provider's cache-read multiplier; substitute your own)

Full round trip on a real session, through `CachedCompressor._call_summary_llm`:

    writer  prompt=20987  cached=20864  hit=99.4%
    reader  prompt=21267  cached=21248  hit=99.91%  cost=0.00166 USD

Real compaction of a 161-message session: 161 down to 103 messages, one auxiliary call,
11 351 prompt tokens of which 11 264 were cache reads (98.95 %).

## When it does not help

The design assumes the provider reports cache reads in `usage.prompt_tokens_details.cached_tokens`.
Some backends return no such field at all. There, the prefix is billed at full price, and resending
68 000 tokens costs more than the built-in compressor's serialized window.

The plugin measures instead of assuming. A route that comes back twice without any cache read, on a
prompt large enough to have been cached, is marked as cache-less for the rest of the process, and
compaction falls back to the built-in path. One observation is not enough: the provider may simply
never have seen that exact prefix. Cacheless behaviour is logged:

    cached-compressor: route without confirmed cache, falling back to the built-in compressor

## Requirements

- a provider with prefix caching, and `cached_tokens` in its usage payload;
- `plugins.isolation: in_process` (the default). The `host` isolation mode is incompatible: the
  context engine subclasses Hermes' own compressor class, which cannot be imported across an
  isolation boundary;
- Hermes 0.23.0 or newer.

## Install

From a git checkout:

    git clone https://github.com/<owner>/hermes-plugin-cached-compressor
    ln -s "$PWD/hermes-plugin-cached-compressor" ~/.hermes/plugins/cached-compressor

Or let Hermes clone it:

    hermes plugins install <owner>/hermes-plugin-cached-compressor

Then enable both entry points. They are both required: the engine does the compaction, the plugin
registers the hook that captures the payload.

    hermes config set context.engine cached-compressor
    hermes plugins enable cached-compressor

Check it:

    hermes config get context.engine     # cached-compressor
    hermes plugins list                  # cached-compressor enabled

## Design

Two entry points in one package, both through `register(ctx)`:

- context engine (`context.engine: cached-compressor`): `CachedCompressor` subclasses the built-in
  compressor and changes only the summary request;
- regular plugin (`plugins.enabled`): registers `pre_api_request`, which receives the final wire
  payload after assembly, sanitizers and the cache plan (`agent/turn_api_request.py`).

`state.PayloadSlot` keeps the last observed payload with its route (provider, model, base URL,
API mode), per session, so the summary runs with the model that actually answered in that session.
Session rotation during compaction happens after the summary call, so the captured thread still
matches.

Because the package is loaded twice, under two module names, the payload slot and the route cache
tracker live in a `sys.modules` sentinel module (`state._shared_holder`). Otherwise the hook would
write to a slot the engine never reads.

## Tests

    HERMES_SOURCE=~/.hermes/hermes-agent python -m pytest tests -q

48 tests, no network, no API key. `HERMES_SOURCE` points at a Hermes source checkout (the plugin
imports `agent.*`).

Live probes, which do need a provider key, are in `probes/`:

    HERMES_API_KEY=... python probes/cache_hit_live.py     # writer + reader, real assembly path
    HERMES_API_KEY=... python probes/compress_live.py      # full compress(), slot and telemetry

## Known limitations

- The plugin relies on parts of Hermes that are not a public plugin surface: the compressor base
  class, `agent.chat_completion_helpers._iteration_summary_api_messages`,
  `agent.auxiliary_client.call_llm`. `hermes plugins validate` reports them as warnings. A Hermes
  upgrade can break it; `hermes.min_version` is declared, and the fallback path keeps compaction
  working if the contract is not met.
- Cache accounting is only as good as the provider's usage payload. On providers that do not report
  cache reads, the plugin steps aside (see above).
