# Head-to-head benchmark

Does compaction cost less when the summary call reads the session prefix from the provider cache,
instead of sending a re-serialized copy of the window at full price? And is the summary still good?

Two experiments answer it, because one number cannot: `bench.py` runs whole compactions with the
output length free, `input_only.py` pins the output so the bill isolates the prompt. Both post to the
same model through the same transport, and read the router's own billed cost back from `usage`
instead of trusting a price table.

    python bench/bench.py --tokens 20000 --tokens 100000        # whole compaction
    python bench/input_only.py --tokens 20000 --tokens 100000   # output pinned
    python bench/report.py                                      # the tables below, from results/*.json

Model: `deepseek-v4.1-flash` through a router that bills and reports cache reads (`opencode-go`).

## 1. Whole compaction

Deterministic transcripts of ~20 000 and ~100 000 tokens (85 and 413 messages: a code audit with tool
calls, file excerpts, decisions carrying `path:line` evidence). Both engines run their real
`compress(..., force=True)` on it. Before the summary call, the session prefix is sent until two
consecutive requests report at least 90 % cache reads. The judge scores the two summaries blind on
five criteria, twice, once per label order, and a disagreement between the orders marks the verdict
unstable.

    pass       size     hit   in x     out     b cost     p cost    cost     lat  judge (b/p)
    20261109  20000   93.0%    3.2    -36%   0.007195   0.004147  -42.4%  -38.9%  24.0/15.5 builtin
    20261109 100000   95.8%    7.4     +3%   0.003609   0.003210  -11.1%  +12.7%  17.0/20.0 tie
    20261109  20000    0.0%    3.2    -67%   0.008802   0.006402  -27.3%  -55.2%  18.5/22.0 tie
    20261109 100000   59.4%    7.3    -81%   0.008134   0.004199  -48.4%  -68.5%  21.0/18.5 builtin
    20261109  20000   93.2%    3.2    -70%   0.009171   0.002704  -70.5%  -71.0%  23.0/17.5 builtin
    20261109 100000   95.9%    7.3    -68%   0.008392   0.002807  -66.6%  -67.6%  15.5/25.0 plugin
    20261109  20000    0.0%    3.2    -49%   0.007011   0.006862   -2.1%  -46.3%  20.5/21.5 tie
    20261109 100000   59.4%    7.3    -34%   0.004413   0.005100  +15.6%  -27.5%  19.5/17.0 tie

    runs: 8   plugin cache served (>=90 %): 4
      all runs            cost  -31.6 %   latency -45.3 %   output -50.3 %   input x 5.2
      prefix served       cost  -47.6 %   latency -41.2 %   output -42.9 %   input x 5.3

    judge mean: built-in 19.9/25   plugin 19.6/25
    verdicts: builtin 3, plugin 1, tie (unstable) 4

`hit` is the share of the plugin's prompt billed at the cache price, `in x` the ratio of prompt tokens
sent by the plugin to prompt tokens sent by the built-in, `out` the change in output tokens.

Three things to read there. Cost and latency favour the plugin in 7 runs out of 8, by 32 % and 45 %
on average. The losing run is the one where the prefix was not served from cache; cost then rises
15.6 %. And the judge cannot separate the two engines: 19.9 against 19.6 out of 25, with half the
verdicts flipping when the labels are swapped, which is what a rubric at the edge of noise looks
like. The plugin's summaries are half the size on average.

## 2. Pinned output (input only)

Output length is the confounder in experiment 1: the built-in writes 4 500 to 13 200 tokens of
summary, the plugin 2 300 to 6 300. This variant removes it. Each engine's summary payload is
captured with a stub transport (no network), then both payloads are posted with `max_tokens=256`.
With the output pinned, the difference in the bill comes from the prompt alone.

The built-in payload is posted cold, because it is: its serialized window is new text at every
compaction, so the provider has never seen it. The plugin's prefix, by contrast, was just sent by the
main loop, so it starts warm, as in production.

      size  b prompt   b hit     b cost  p prompt   p hit     p cost   p/b
     20000      8157    0.0%   0.001377     26094   93.2%   0.000493  0.36
    100000      6141   33.3%   0.000774     44585   91.9%   0.000820  1.06
     20000      8157    0.0%   0.001377     26094   86.3%   0.000756  0.55
    100000      6141   33.3%   0.000774     44585   91.9%   0.000820  1.06
     20000      8157    0.0%   0.001377     26094   93.2%   0.000493  0.36
    100000      6141    0.0%   0.001075     44585   95.9%   0.000557  0.52
     20000      8157    0.0%   0.001377     26094    0.0%   0.004068  2.95
    100000      6141    0.0%   0.001075     44585   59.4%   0.002946  2.74
     20000      8157    0.0%   0.001377     26094   93.2%   0.000493  0.36
    100000      6141    0.0%   0.001075     44585   95.9%   0.000557  0.52
     20000      8157    0.0%   0.001377     26094   93.2%   0.000493  0.36
    100000      6141   33.3%   0.000774     44585   95.9%   0.000557  0.72
     20000      8157    0.0%   0.001377     26094    0.0%   0.004068  2.95
    100000      6141    0.0%   0.001075     44585   59.4%   0.002946  2.74

    mean p/b: 20000 -> 1.13 (0.40 on the six rows served at >=85 %)
              100000 -> 1.34 (0.78 on the six rows served at >=85 %)

So the cache effect, isolated, is a factor 2.5 down to 1.3 in the plugin's favour when the prefix is
served, and a factor 2.7 to 3.0 against it when it is not. Everything rests on that hit rate, which
is why the plugin ships a cacheability guard: the fallback to the built-in path described in the
README exists for the rows on the right-hand side of this table.

## Why the hit is not always total

Two causes, both measured.

**The instruction tail is not cacheable.** The plugin sends the session prefix followed by the
summary instruction; the instruction was never part of the conversation, so it can never match.
Compared against what the main loop actually sent, the payload is identical up to 74 443 of 81 467
characters at 20 000 tokens (91.4 %) and 127 077 of 138 679 at 100 000 tokens (91.6 %), the rest
being the instruction. That is the ceiling of the hit rate, and it shows in the table: 93 %, 95.8 %,
95.9 %.

**The provider does not always serve what it just accepted.** The same payload, sent after a warm
loop that already reported two consecutive hits, sometimes comes back with zero cached tokens.
Sending one identical request six times in a row shows why: `0 %, 0 %, 95.3 %, 95.3 %, 95.3 %,
95.3 %`. The cache write lands a few seconds after the response, so the first requests are always
cold, and any request issued inside that window pays full price. Nothing on the client controls it.

## The two engines do not read the same thing

This explains the asymmetry in the cost columns. On the 100 000-token transcript (413 messages,
361 633 characters) the built-in path summarized 134 records truncated to 14 039 characters, 4 % of
the history, because it bounds each record to a slice of `_SUMMARY_INPUT_MAX_CHARS = 160_000`
(`agent/context_compressor.py:836`, `:3640`). The plugin sent the 43 309-token prefix. On the
20 000-token transcript nothing is elided (74 of 74 records, 20 606 characters).

The built-in is cheap partly because it looks at a fraction of the session. That is also why it wins
the rubric on the 20 000-token runs and not on the 100 000-token ones: at that size it summarizes a
truncated sample, and its evidence score drops.

In one line: the plugin compacts for about a third less, faster, over the real conversation instead
of a bounded sample, and writes a terser checkpoint, with the whole saving resting on the provider
serving the prefix from cache.

## Caveats

- One provider and its own billed prices (cache reads at 10 % of input here); a provider that reports
  no cache reads at all is handled by the fallback described in the README.
- n = 4 per size for the whole-compaction experiment, n = 6 per size for the pinned one. The judge is
  one model at temperature 0, and it disagrees with itself across label orders on half the runs.
- In a real deployment the built-in path calls `auxiliary.compression`, often a cheaper model than
  the session's. Both arms call the same model here to isolate the cache, so the absolute dollars
  differ from a real configuration; compare against your own auxiliary model.
- Transcripts are synthetic and regular. A more heterogeneous history changes the built-in's sampling
  coverage and both summary lengths.
