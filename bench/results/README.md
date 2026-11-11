# Reports

Counters only. No transcript content, no paths from the machine, no identifiers.

## Current

- `arms-20261111-partial.json`: the four-arm compression-event experiment behind section 5 of
  [`../BENCHMARK.md`](../BENCHMARK.md). Four anonymised sessions, three cells lost to router throttling and
  therefore absent. This is the reference for any claim about the cost or the quality of a compression event.
- `fact-recall-20261110-102955.json`: the planted-fact recall behind section 4.

## Superseded

- `hetero-20261110-105929.json`: the engine-level comparison on four real sessions, source of the withdrawn
  "34 % cheaper" and "67 to 79 % cache hit" figures. The cost totals in it are a total-cost comparison, not a
  compression-event comparison, and its `hit` column was measured by a probe that ran before the call it was
  measuring, which primed the bytes. Kept for the record, not to be cited.
- `bench-*`, `input-only-*`: the synthetic transcripts behind sections 1 to 3, from the same harness generation
  as the report above. The comparisons within each report hold because both arms were measured the same way,
  but their absolute hit figures should not be quoted without repeating the protocol of section 5.
