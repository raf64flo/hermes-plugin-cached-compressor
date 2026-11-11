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

    pass               size     hit   in x     out     b cost     p cost    cost     lat  judge (b/p)
    20261109-151211   20000   93.0%    3.2    -36%   0.007195   0.004147  -42.4%  -38.9%  24.0/15.5 builtin
    20261109-151211  100000   95.8%    7.4     +3%   0.003609   0.003210  -11.1%  +12.7%  17.0/20.0 tie (unstable)
    20261109-152604   20000    0.0%    3.2    -67%   0.008802   0.006402  -27.3%  -55.2%  18.5/22.0 tie (unstable)
    20261109-152604  100000   59.4%    7.3    -81%   0.008134   0.004199  -48.4%  -68.5%  21.0/18.5 builtin
    20261109-161006   20000   93.2%    3.2    -70%   0.009171   0.002704  -70.5%  -71.0%  23.0/17.5 builtin
    20261109-161006  100000   95.9%    7.3    -68%   0.008392   0.002807  -66.6%  -67.6%  15.5/25.0 plugin
    20261109-164354   20000    0.0%    3.2    -49%   0.007011   0.006862   -2.1%  -46.3%  20.5/21.5 tie (unstable)
    20261109-164354  100000   59.4%    7.3    -34%   0.004413   0.005100  +15.6%  -27.5%  19.5/17.0 tie (unstable)

    runs: 8   plugin cache served (>=90 %): 4
      all runs            cost  -31.6 %   latency -45.3 %   output -50.3 %   input x 5.2
      prefix served       cost  -47.6 %   latency -41.2 %   output -42.9 %   input x 5.3
      cost spread         -70.5 % to +15.6 %   median -34.9 %   sd 30.9

    judge mean: built-in 19.9/25   plugin 19.6/25
    verdicts: builtin 3, plugin 1, tie (unstable) 4

`hit` is the share of the plugin's prompt billed at the cache price, `in x` the ratio of prompt tokens
sent by the plugin to prompt tokens sent by the built-in, `out` the change in output tokens.

Three things to read there. Cost and latency favour the plugin in 7 runs out of 8, by 32 % and 45 % on
average, with a wide spread (-2 % to -70 % on cost) because the router's decision to serve the prefix
dominates the difference. The losing run is the one served at only 59 %: cost then rises 15.6 %. And
the judge cannot separate the two engines: 19.9 against 19.6 out of 25, with half the verdicts
flipping when the labels are swapped, which is what a rubric at the edge of noise looks like. The
plugin's summaries are half the size on average.

## 2. The built-in on a cheap auxiliary model

In a real deployment the built-in path does not call the session model: it calls
`auxiliary.compression`, usually a cheaper one. Experiment 1 gives both arms the session model, which
isolates the cache; this one asks the question a user actually faces.

    python bench/bench.py --tokens 20000 --tokens 100000 --repeats 3 --aux-model mimo-v2.6-flash

Three repeats per size, each on its own transcript: a repeat that reused the bytes would find both
arms already cached, and the built-in's window is stable across repeats. Prices are read from the
provider's own sheet: session `deepseek-v4.1-flash` at 0.15/0.6 USD per million input/output,
auxiliary `mimo-v2.6-flash` at 0.14/0.28.

    pass               size     hit   in x     out     b cost     p cost    cost     lat  judge (b/p)
    20261109-222637   20000   93.0%    3.4   +184%   0.001738   0.004284 +146.5%  +21.2%  14.0/25.0 plugin
    20261109-222637   20000   93.0%    3.4    +57%   0.001785   0.002686  +50.5%  -52.3%  17.0/23.5 plugin
    20261109-222637   20000   93.0%    3.4   +121%   0.001691   0.003191  +88.7% +273.6%  14.5/24.0 plugin
    20261109-222637  100000   96.0%    7.3   +173%   0.001396   0.003572 +155.9% +373.5%  17.0/24.0 plugin
    20261109-222637  100000   96.0%    7.3    +92%   0.001296   0.003381 +160.9%  -49.2%  13.0/24.5 plugin
    20261109-222637  100000   96.0%    7.3   +100%   0.001531   0.003299 +115.5%  -36.0%  16.0/24.5 plugin

    runs: 6   plugin cache served: 6
      subset                   n  cost median               CI95     mean  latency  output  input
      all runs                 6      +131.0% [  +86.1, +149.2]  +119.7%   +88.5% +121.0% x 5.3
      20000 tokens             3       +88.7% [  +50.5, +146.5]   +95.2%   +80.8% +120.5% x 3.4
      100000 tokens             3      +155.9% [ +115.5, +160.9]  +144.1%   +96.1% +121.4% x 7.3

    judge mean: built-in 15.2/25   plugin 24.2/25   (the plugin wins all six runs)

The cache is served in every run and the plugin still costs 1.9 to 2.6 times more. Both reasons are
in the columns: it writes about twice as many summary tokens, because the session model is more
verbose than the cheap one, and it pays the session price for them.

The break-even follows from the same numbers. Writing R for session price / auxiliary price, the
plugin wins while

    R < R_max = (built-in tokens at session price) / (plugin billed cost)

which measures 0.8 at 20 000 tokens and 0.7 at 100 000. A session model is almost always more
expensive than its auxiliary, so R > 1 > R_max: against a cheap auxiliary the plugin is a quality
purchase, not a saving. `bench/report.py` prints R_max from the recorded price sheet.

What the judge measures in this section is the model, not the engine: the two summaries come from
different models. 24.2 against 15.2 says the session model summarizes better than the cheap one, which
is expected and is the whole point of paying for it.

## 3. Pinned output (input only)

Output length is the confounder in experiment 1: the built-in writes 4 500 to 13 200 tokens of
summary, the plugin 2 300 to 6 300. This variant removes it. Each engine's summary payload is
captured with a stub transport (no network), then both payloads are posted with `max_tokens=256`.
With the output pinned, the difference in the bill comes from the prompt alone.

The built-in payload is posted as its engine builds it, with no warm-up: cold at 20 000 tokens, and
33.3 % cached in three of the seven passes at 100 000 tokens, which the harness does not control (its
6 141-token window sometimes matches a prefix the router still holds). The plugin's prefix, by
contrast, was just sent by the main loop, so it starts warm, as in production.

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

    mean p/b: 20000 -> 1.13 (0.40 on the five rows served at >=85 %)
              100000 -> 1.34 (0.78 on the five rows served at >=85 %)

So the cache effect, isolated, is a factor 2.5 down to 1.3 in the plugin's favour when the prefix is
served, and a factor 2.7 to 3.0 against it when it is not. Everything rests on that hit rate, which
is why the plugin ships a cacheability guard: the fallback to the built-in path described in the
README exists for the rows on the right-hand side of this table.

## Why the hit is not always total

Two causes, both measured.

**The instruction tail is not cacheable.** The plugin sends the session prefix followed by the
summary instruction; the instruction was never part of the conversation, so it can never match.
Compared against what the main loop actually sent, the payload is identical up to 74 443 of 82 397
characters at 20 000 tokens (90.3 %) and 127 077 of 138 679 at 100 000 tokens (91.6 %), the rest
being the instruction. Those are character shares, and they understate the hit the provider reports:
the transcript's code and path excerpts pack more tokens per character than the instruction's prose,
so the instruction takes only 4 to 8 % of the payload's tokens. It still caps the hit rate, and the
cap shows in the table: 93 %, 95.8 %, 95.9 %.

**The provider does not always serve what it just accepted.** The same payload, sent after a warm
loop that already reported two consecutive hits, sometimes comes back with zero cached tokens.
Sending one identical request six times in a row shows why: `0 %, 0 %, 95.3 %, 95.3 %, 95.3 %,
95.3 %`. The cache write lands a few seconds after the response, so the first requests are always
cold, and any request issued inside that window pays full price. Nothing on the client controls it.

## The two engines do not read the same thing

This explains the asymmetry in the cost columns. On the 100 000-token transcript (413 messages,
361 633 characters) the built-in summarized 134 records for 14 039 characters in total, 3.9 % of the
history. The bound is not a sampling decision: before summarizing, its own pruning had already
replaced 97 of those 134 records, all tool results, with the placeholder `[Old tool output cleared to
save context space]` (`agent/context_compressor.py:838`, `_prune_old_tool_results` at `:3274`). The
block is also capped at `_SUMMARY_INPUT_MAX_CHARS = 160_000` (`:836`, `:3640`), but that cap did not
bind here: nothing was omitted. On the 20 000-token transcript the built-in read 74 records for
20 606 characters, 28 % of the history, with 15 pruned tool results. The plugin sent the 43 309-token
prefix instead.

The built-in is cheap partly because it does not read most of the session: it summarizes a bounded
window, and the old tool output inside that window reaches the summarizer as a one-line placeholder.

In one line: the plugin compacts for about a third less, faster, over the real conversation instead
of a bounded sample, and writes a terser checkpoint, with the whole saving resting on the provider
serving the prefix from cache.

## Caveats

- One provider and its own billed prices (cache reads at 2 % of input here, derived from its bills); a
  provider that reports no cache reads at all is handled by the fallback described in the README.
- n = 4 per size for the whole-compaction experiment, n = 7 per size for the pinned one. The judge is
  one model at temperature 0, and it disagrees with itself across label orders on half the runs.
- In a real deployment the built-in path calls `auxiliary.compression`, often a cheaper model than
  the session's. Section 2 measures exactly that: against a cheap auxiliary the plugin costs more and
  summarizes better, so the dollars in section 1 describe the cache, not a deployment.
- Transcripts are synthetic and regular. A more heterogeneous history changes the built-in's sampling
  coverage and both summary lengths.

## 4. Fact recall

The rubric judge scores a summary as a whole; it cannot say which facts survived. This experiment
plants nine machine-checkable facts in three places and asks a reader (the session model, shown only
the compacted transcript) one question per fact, scored on exact token containment:

- *tail*: facts in the protected tail, which no engine is allowed to summarize away;
- *middle*: facts in ordinary conversation inside the summarized window, the family that measures the
  summarizer;
- *pruned*: facts inside a large tool output, which the parent replaces on compaction. A control: both
  engines lose these by design, and a hit there means the summary happened to quote the log.

`python bench/fact_recall.py --tokens 20000`, one 202-message transcript (~18k tokens), nine facts.

| arm | tail | middle | pruned | prompt tokens | cache hit | billed | latency | messages |
|---|---|---|---|---|---|---|---|---|
| built-in | 3/3 | 3/3 | 0/3 | 8 295 | 0 % | 0.0172 USD | 128.9 s | 202 to 10 |
| plugin | 3/3 | 3/3 | 0/3 | 23 470 | 92.7 % | 0.0031 USD | 27.1 s | 202 to 10 |

Read it as a control on quality, not as a tie-breaker: on this transcript both engines keep exactly
what they are supposed to keep, and the pruned family is lost by both, which confirms the family
measures pruning and not the summarizer. The difference is the bill: 5.5 times cheaper and 4.7 times
faster for the plugin, consistent with the whole-compaction numbers above. The probe only
discriminates when a summary actually drops a fact, and one transcript at one size is not enough to
find that boundary; it is a floor, not a curve.

## 5. One compression event, on real sessions

The unit that matters is the event, not the context that comes out of it. When a compaction fires, one call
has to read the material and write a summary, and that call is where the money goes. Four ways to pay for it,
on the same material, in the same message format:

| arm | model | instruction | session | what it represents |
|---|---|---|---|---|
| A | `gemma-4-31b-it` | first, then the material | separate route | the auxiliary summarizer |
| B | `deepseek-v4.1-flash` | appended after the material | the session that just ran | the plugin as deployed |
| C | `deepseek-v4.1-flash` | first, then the material | the session that just ran | the same model, without the prefix |
| B' | `deepseek-v4.1-flash` | appended after the material | a session that never saw the material | the plugin without a warm session |

B against C isolates the position of the instruction at constant model, constant content and constant session.
B against B' isolates what reusing the session buys. C against A isolates the model at constant request shape.
Nothing else differs: the four arms receive the material truncated at the same point, the protected tail is
excluded for all of them, and the instruction text is byte-identical.

Privacy is the constraint, not an afterthought. `bench/anonymize.py` masks credentials and tokens (OpenAI,
GitHub, GitLab, AWS, JWTs, `key=value` secrets), emails, IPv4 addresses, phone numbers, URLs (userinfo
dropped, hosts mapped to `host-N.invalid`, query values dropped), home directories and an explicit list of
names. A canary corpus plants one sentinel per category and the probe refuses to run if any survives; a leak
guard then scans every 30-character window of every transcript against the serialized report before writing
it. What is published is the report: counters only.

Known limit, stated rather than hidden: free-text person and place names that no pattern catches are not
masked by construction. The canary covers the categories above, and the probe is meant for one's own corpus,
not for someone else's.

Quality is measured by containment, not by a judge. From the material, `bench/compaction_arms.py` draws up to
three anchors of each kind with a fixed seed: a path, with its line number when it has one; a number, with its
unit; a sentence carrying a decision marker. A summary keeps a path only if the file and the line both appear,
keeps a number only if the unit survives, and keeps a decision only if the wording survives. Each kind is
scored on its own terms, so a summary that drops every path is not hidden by one that keeps the numbers.

`python bench/compaction_arms.py --sessions 6 --repeats 1`, six anonymised sessions, tokens and billed cost
as reported by the router, public list prices applied separately:

| session | arm | cache hit | billed | anchors | questions |
|---|---|---|---|---|---|
| `5f3764e77a64` | B | 100 % | 0.00041 USD | 1/2 | 0/0 |
| | C | 100 % | 0.00095 USD | 1/2 | 0/0 |
| | B' | 100 % | 0.00087 USD | 1/2 | 0/0 |
| `dc25588ebb36` | B | 99 % | 0.00065 USD | 2/7 | 3/6 |
| | C | 0 % | 0.00210 USD | 0/7 | 0/6 |
| | B' | 0 % | 0.00278 USD | 3/7 | 2/6 |
| `ef69c96cbd26` | B | 96 % | 0.00047 USD | 2/3 | 0/0 |
| | C | 0 % | 0.00132 USD | 1/3 | 0/0 |
| | B' | 0 % | 0.00161 USD | 2/3 | 0/0 |
| `a714b5bda975` | B | 99 % | 0.00041 USD | 1/6 | 1/5 |
| | C | 0 % | 0.00247 USD | 0/6 | 0/5 |
| | B' | 0 % | 0.00271 USD | 1/6 | 1/5 |
| `d9e1f5960767` | B | 95 % | 0.00057 USD | 2/3 | 1/6 |
| | C | 0 % | 0.00076 USD | 0/3 | 0/6 |
| | B' | 14 % | 0.00114 USD | 2/3 | 2/6 |
| `62d8996f4103` | B | 98 % | 0.00081 USD | 0/2 | 0/0 |
| | C | 19 % | 0.00049 USD | 0/2 | 0/0 |
| | B' | 0 % | 0.00141 USD | 0/2 | 0/0 |

The auxiliary arm A is free on its route and is not priced here; it kept 3 of 23 anchors and answered none of
its 17 questions.

| arm | events | billed per event | anchors | questions | cache hit, median |
|---|---|---|---|---|---|
| A | 6 | free | 3/23 | 0/17 | 0 % |
| B | 6 | 0.00055 USD | 8/23 | 5/17 | 98 % |
| C | 6 | 0.00135 USD | 1/23 | 0/17 | 0 % |
| B' | 6 | 0.00175 USD | 9/23 | 5/17 | 0 % |

**The invoice follows the form of the request.** B costs 2.4 times less than C and 3.2 times less than B', on
six sessions and twelve comparisons, with the same model, the same material and the same instruction.

**The reordering costs quality, independently of the cache.** C keeps one anchor of 23 and answers none of 17
questions; B keeps eight and answers five. The two arms share every setting except where the instruction sits,
so the difference is the form, not the model. The auxiliary model behaves like C: three anchors, no question
answered. What it writes is shorter, and shorter is what loses the paths and the line numbers.

**B' shows the same quality as B and a different bill.** Same bytes, same model, three times the cost, which
confirms the saving comes from the request, not from the summary.

**The counter cannot be trusted as a control.** In `5f3764e77a64`, B' reports a 100 % hit in a session that
never sent the material, and one hour later C reports 99 % where it reported 0 % six times earlier. The
provider's cache state is not controllable from the client: it depends on which instance serves the request and
which blocks it still holds. The billed cost remains a fact, because it is what the invoice says; the hit
percentage is not a reliable intermediate.

## Speed: not answerable on the router, answered on a local server

Two protocols were run against the router, and both are recorded as failures rather than results.

A streaming call returns one hundred and twenty chunks in one tenth of a second after a first packet at 1.64 s:
the gateway buffers the answer and replays it, so time to first token is not a prefill measurement there.

A two-point calibration does work: two output budgets on the same prompt give a decoding slope of 0.0041 s per
token, about 245 tokens per second, and subtracting it from a service time leaves the reading time. Applied to
B and C over three sessions and three repetitions each, it gave 6.5 s of reading in the warm session against
6.2 s in the cold one, with both arms reporting a hit above 98 % during that pass. The contrast the protocol was
built to show had disappeared, because the cache state had changed between the two runs.

That route cannot answer the question: the same bytes are warm in one pass and cold in the next, and the client
cannot force either. A local server can, because it holds one KV cache, routes nothing, and publishes what it
reused per request. The measurement below was taken on `llama-server` b12262, `Qwen2.5-7B-Instruct-Q4_K_M`,
one slot, `temperature 0`, 16 384 tokens of context, on a 12 GB laptop GPU.

Protocol, per session: send the conversation once (that is the premise of the plugin, the material has just
been sent), then B, then C, then C a second time. Four sessions whose material had never been through the
server, checked by the priming call reporting `cache_n = 0`.

| arm | form | cache_n | tokens re-read | reading time | full call |
|---|---|---|---|---|---|
| B | instruction appended | 7 890 | 56 | 35.0 ms | 1.17 s |
| C | instruction first | 0 | 7 946 | 2 592.9 ms | 4.16 s |
| C again | instruction first | 7 946 | 55 | 19.9 ms | 1.14 s |

**Moving the instruction to the front throws the prefix away.** The conversation is in the cache, the plugin
form reuses all of it and reads only the instruction, fifty-six tokens. The native form re-reads the whole
material, because its first token differs. On the same material, same model, same machine: seventy-four times
less reading, and three and a half times less wall clock per compression.

**The third line proves what the cache follows.** The same instruction-first request, sent a second time, is
immediately warm, which rules out the arm itself as the cause: what costs is not the order, it is that the
order changed since the material was sent.

Two things this does not say. It does not transfer to a router whose cache or routing you do not control, which
is why the section above exists. And it measures a seven-billion-parameter model on one machine, so the
absolute times are not comparable to the router's, only the ratio between the two forms on identical hardware.

## Limits

Six sessions, one repetition, one router, six events per arm. Anchors are thin, two to seven per session, and
questions could be built for only three sessions of six, so the quality figures are a direction rather than a
measurement with error bars. The cache hit is reported for information and is not used as evidence anywhere.
No speed claim is made about the router, for the reason above; the local measurement is a separate result, on
one seven-billion-parameter model and one machine.
