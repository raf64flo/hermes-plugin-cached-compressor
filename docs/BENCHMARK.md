# Head-to-head benchmark

Does compaction cost less when the summary call reads the session prefix from the provider cache,
instead of sending a re-serialized copy of the window at full price? And is the summary still good?

This page answers both with one experiment, and reports what did not go the plugin's way.

## Protocol

`bench/bench.py` builds a deterministic transcript (a code audit: user requests, tool calls, file
excerpts, decisions with `path:line` evidence, one closing summary request) sized to ~20 000 and
~100 000 tokens, that is 85 and 413 messages. Both engines then run their real
`compress(..., force=True)` over that transcript:

- built-in: `agent.context_compressor.ContextCompressor`;
- plugin: `CachedCompressor`, the context engine this repository ships.

Both call the same model through the same transport. The transport posts exactly the messages the
engine builds and reads `usage` back, including the router's own billed cost
(`usage.opencode.list_cost`), so no price table is invented here.

Before each run the session prefix is sent until the provider reports at least 90 % cache reads,
because a cache write is asynchronous: one or two attempts in practice. The warm state is recorded
in the results file, and a run where the prefix never entered the cache is flagged.

A rubric judge, same model at temperature 0, scores the two summaries blind on five criteria
(fidelity, decisions, open items, preserved evidence, usefulness). It votes twice, once per label
order, and the votes are averaged; a disagreement between the two orders marks the verdict as
unstable.

Two passes per size, the second with `BENCH_SALT=pass2` so its transcript is text the provider cache
has never seen. Raw records: `bench/results/*.json`.

    BENCH_API_KEY=... python bench/bench.py --tokens 20000 --tokens 100000
    BENCH_SALT=pass2 BENCH_API_KEY=... python bench/bench.py --tokens 20000 --tokens 100000

Model: `deepseek-v4.1-flash` through a router that bills and reports cache reads (`opencode-go`).

## Results

    pass      size    arm      calls  prompt  cached   hit     out   cost USD  latency  msgs
    151211    20k     built-in     1    8 065       0   0.0 %  9 975   0.007195    75.1 s  85->11
    151211    20k     plugin       1   26 017  24 192  93.0 %  6 335   0.004147    45.9 s  85->11
    151211   100k     built-in     1    5 863       0   0.0 %  4 550   0.003609    31.4 s 413->279
    151211   100k     plugin       1   43 309  41 472  95.8 %  4 684   0.003210    35.4 s 413->279
    152604    20k     built-in     1    8 157       0   0.0 % 12 630   0.008802   134.7 s  85->11
    152604    20k     plugin       1   26 094       0   0.0 %  4 147   0.006402    60.3 s  85->11
    152604   100k     built-in     1    6 141       0   0.0 % 12 021   0.008134   141.5 s 413->275
    152604   100k     plugin       1   44 585  26 496  59.4 %  2 344   0.004199    44.6 s 413->275

    plugin against built-in, mean over the four runs:
      billed cost    -32.3 %
      latency        -37.5 %   (3 of 4 runs faster, one 12.7 % slower)
      input tokens   x 5.3     (x 3.2 at 20k, x 7.4 at 100k)
      summary size   -19.4 %   (from -7.6 % to -39.7 %)

Judge, mean over the four runs: built-in 20.1/25, plugin 19.0/25. Two runs are stable verdicts for
the built-in, two are unstable ties. The plugin never wins the rubric outright.

## Reading

**Cost and latency move the plugin's way, consistently.** Four runs out of four are cheaper, three
out of four are faster. The mechanism is visible in the payload: at 93-96 % cache reads the plugin
pays the cache price for a prefix 3 to 7 times larger than the built-in's input.

**The plugin does not win on tokens sent, and it does not claim to.** Its prefix is 5.3 times larger
on average; the saving comes from the discount, not from sending less. When the provider does not
serve the prefix from cache, the plugin sends that larger prefix at full price (pass 2 at 20k, 0 %
hit). It still ended 27 % cheaper there, but only because its summary was much shorter, which is not
a property you can rely on.

**Summaries come out shorter, and the rubric notices.** The plugin's summary is 7.6 % to 39.7 %
smaller. The judge rewards detail, so the built-in wins the two stable verdicts. On the two unstable
runs the two label orders disagree, which is what a rubric at the edge of significance looks like.

**The two engines do not read the same thing.** This is the part that explains the numbers above. On
the 100 000-token transcript (413 messages, 361 633 characters) the built-in path summarized 134
records truncated to 14 039 characters, 4 % of the history, because it bounds each record to a slice
of `_SUMMARY_INPUT_MAX_CHARS = 160_000` (`agent/context_compressor.py:836`, `:3640`). The plugin sent
the 43 309-token prefix. The built-in is cheap partly because it looks at a fraction of the session,
and its evidence score (3/5 on the 100k runs) reflects that. On the 20 000-token transcript nothing
is elided (74 of 74 records, 20 606 characters) and the built-in wins the rubric on detail.

So the honest one-line summary: the plugin compacts for about a third less, faster, over the real
conversation instead of a bounded sample, and writes a terser checkpoint.

## Caveats

- One provider and its own billed prices. Another provider prices cache reads differently (10 % of
  input here); a provider that reports no cache reads at all is handled by the fallback described in
  the README, not by this benchmark.
- n = 2 per size. The judge is one model at temperature 0, and it disagrees with itself across label
  orders on half the runs.
- In a real deployment the built-in path calls `auxiliary.compression`, often a cheaper model than
  the session's. Both arms call the same model here to isolate the cache, so absolute savings in a
  real configuration differ: compare against your own auxiliary model, not against these dollars.
- The transcripts are synthetic and highly regular. A more heterogeneous history changes the size of
  both summaries and the built-in's sampling coverage.
