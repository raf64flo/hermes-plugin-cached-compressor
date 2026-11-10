# Provider compatibility

The plugin only makes sense on a route that reports cache reads. This page says which field each
family uses, how to check your own backend in a minute, and when the gain is time rather than money.

## The one-minute check

    BENCH_API_KEY=... python probes/cache_report_check.py --base URL --model ID --tokens 4000

It posts the same prompt four times, four seconds apart, and prints the usage fields it sees. On the
router used for the published benchmark:

    appel 1: prompt=3704 champs={'prompt_tokens_details.cached_tokens': 0} hit=0.0% prefill=0.80s
    appel 2: prompt=3704 champs={'prompt_tokens_details.cached_tokens': 3584} hit=96.8% prefill=0.90s
    appel 3: prompt=3704 champs={'prompt_tokens_details.cached_tokens': 3584} hit=96.8% prefill=0.54s

The first call always misses: the cache write lands after the response. A backend that never reports a
field, or never reports a hit, is treated by the plugin as a route without cache: after two silent
measurements it falls back to the built-in compressor for the rest of the process.

## Fields read

`__init__._cached_tokens` tries these in order, at the root of `usage` and inside
`usage.prompt_tokens_details`:

| family | field | notes |
|---|---|---|
| OpenAI, vLLM, most OpenAI-compatible routers | `prompt_tokens_details.cached_tokens` | vLLM needs `--enable-prefix-caching` |
| DeepSeek | `prompt_cache_hit_tokens` | cache on by default, no configuration |
| Anthropic | `cache_read_input_tokens` | requires explicit `cache_control` breakpoints |
| Gemini | `cached_content_token_count` / `cachedContentTokenCount` | explicit cached content API |

Anything else is a mute route for this plugin, even if the backend caches internally.

## What the saving is made of

- **Billed route (per token):** the gain is money. The prefix is billed at the provider's cache price,
  which the published benchmark measures at 2 % of the input price on the reference router. Whether the
  bill actually drops depends on the model the built-in would have used: see the break-even ratio in
  `docs/BENCHMARK.md`, section 2.
- **Local inference (vLLM, llama.cpp, Ollama, on your own GPU):** nothing is billed per token, so the
  gain is prefill time, not cost. A 40 000-token prefix re-read from the KV cache saves the prefill of
  those tokens, which is the difference between a compaction that stalls the session for a minute and
  one that does not. The plugin still helps there, and for a different reason: it does not fall back to
  a cheaper auxiliary model, it summarizes with the session model.

## Backends that report nothing

Ollama does not report cache reads in the OpenAI-compatible usage block. llama.cpp reports them only
when prompt caching is enabled and the request matches the previous one byte for byte, which is what
the plugin does by design, but the field is not always surfaced.

Both end up on the fallback path. That is deliberate: without a billed cache read there is nothing to
save on a metered route, and paying the full price for a longer prefix is strictly worse than the
built-in's serialized window.

## Cache write window

Compaction happens right after the main call of the same turn, so the only question is whether the
prefix written by that call is readable seconds later. Measured with `probes/cache_write_sweep.py`
on the reference router (`deepseek-v4.1-flash`, 24 identical 3 704-token posts, 2026-11-10; report
in `docs/results/cache-write-sweep-20261110.json`):

| idle before each post | hit ratios | mean of the last three |
|---|---|---|
| 0 s | 0, 97, 97, 97, 97, 97 % | 97 % |
| 2 s | 97, 97, 97, 97, 97, 97 % | 97 % |
| 5 s | 0, 97, 97, 97, 97, 97 % | 97 % |
| 10 s | 0, 93, 0, 0, 0, 93 % | 31 % |

The write lands within a second of the first response (0 s series, second post served at 97 %), so
there is no window to wait for. The 10 s series says the entry's lifetime is on the order of
seconds and irregular: idle time does not make the read more likely, it makes it less. Hence no
sleep knob in the compaction path, sleeping cannot buy a hit the cache will not serve and it takes
a fixed cost on every turn. The 2 s series opening warm is the previous series' prefix still cached,
not a faster write.
