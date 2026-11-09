"""Aggregate the benchmark records into the tables published in docs/BENCHMARK.md.

    python bench/report.py            # reads bench/results/*.json

Two kinds of record:

- ``bench-*.json``      whole compaction, both engines, output length free
- ``input-only-*.json`` same payloads with the output pinned, so the bill isolates the prompt
"""

from __future__ import annotations

import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def load(pattern: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        data["_file"] = os.path.basename(path)
        records.append(data)
    return records


def hit_of(arm: dict) -> float:
    return arm["cached_tokens"] / arm["prompt_tokens"] if arm["prompt_tokens"] else 0.0


def whole_compaction() -> None:
    rows = []
    for record in load("bench-*.json"):
        for run in record["runs"]:
            builtin, plugin = run["arms"]["builtin"], run["arms"]["plugin"]
            rows.append({
                "pass": "-".join(record["_file"].split("-")[1:]).split(".")[0],
                "size": run["target_tokens"],
                "builtin": builtin,
                "plugin": plugin,
                "d_cost": 100 * (plugin["cost_usd"] / builtin["cost_usd"] - 1),
                "d_lat": 100 * (plugin["latency_s"] / builtin["latency_s"] - 1),
                "out_ratio": 100 * (plugin["completion_tokens"] / builtin["completion_tokens"] - 1),
                "in_ratio": plugin["prompt_tokens"] / builtin["prompt_tokens"],
                "hit": hit_of(plugin),
                "judge": run.get("judge") or {},
            })
    if not rows:
        print("no bench-*.json records")
        return

    print("## Whole compaction\n")
    print(f"{'pass':16s} {'size':>6s} {'hit':>7s} {'in x':>6s} {'out':>7s} {'b cost':>10s} "
          f"{'p cost':>10s} {'cost':>7s} {'lat':>7s}  judge (b/p)")
    for row in rows:
        judge = row["judge"]
        totals = judge.get("totals") or {}
        builtin_total = (totals.get("builtin") or {}).get("total")
        plugin_total = (totals.get("plugin") or {}).get("total")
        verdict = judge.get("better_arm", "-")
        stable = "" if judge.get("label_order_agrees", True) else " (unstable)"
        print(f"{row['pass']:16s} {row['size']:6d} {row['hit']:7.1%} {row['in_ratio']:6.1f} "
              f"{row['out_ratio']:+6.0f}% {row['builtin']['cost_usd']:10.6f} "
              f"{row['plugin']['cost_usd']:10.6f} {row['d_cost']:+6.1f}% {row['d_lat']:+6.1f}%  "
              f"{builtin_total}/{plugin_total} {verdict}{stable}")

    def mean(subset: list[dict], key: str) -> float:
        return statistics.mean(row[key] for row in subset)

    served = [row for row in rows if row["hit"] >= 0.9]
    print(f"\nruns: {len(rows)}   plugin cache served (>=90%): {len(served)}")
    for label, subset in (("all runs", rows), (f"prefix served ({len(served)} runs)", served)):
        if not subset:
            continue
        print(f"  {label:22s} cost {mean(subset, 'd_cost'):+6.1f}%   "
              f"latency {mean(subset, 'd_lat'):+6.1f}%   "
              f"output {mean(subset, 'out_ratio'):+6.1f}%   input x{mean(subset, 'in_ratio'):.1f}")

    judges = [(row["judge"].get("totals") or {}, row["judge"]) for row in rows if row["judge"]]
    if judges:
        builtin_mean = statistics.mean(
            (totals.get("builtin") or {}).get("total", 0) for totals, _ in judges)
        plugin_mean = statistics.mean(
            (totals.get("plugin") or {}).get("total", 0) for totals, _ in judges)
        tally: dict[str, int] = {}
        for _, judge in judges:
            key = judge.get("better_arm", "-")
            key += "" if judge.get("label_order_agrees", True) else " (unstable)"
            tally[key] = tally.get(key, 0) + 1
        print(f"\njudge mean: built-in {builtin_mean:.1f}/25   plugin {plugin_mean:.1f}/25")
        print("verdicts:", ", ".join(f"{name} {count}" for name, count in sorted(tally.items())))


def pinned_output() -> None:
    rows = []
    for record in load("input-only-*.json"):
        for run in record["runs"]:
            builtin, plugin = run["arms"]["builtin"], run["arms"]["plugin"]
            rows.append({"size": run["target_tokens"], "builtin": builtin, "plugin": plugin,
                         "ratio": run.get("cost_ratio"), "hit": hit_of(plugin)})
    if not rows:
        print("\nno input-only-*.json records")
        return
    print("\n## Pinned output (input only)\n")
    print(f"{'size':>6s} {'b prompt':>9s} {'b hit':>7s} {'b cost':>10s} {'p prompt':>9s} "
          f"{'p hit':>7s} {'p cost':>10s} {'p/b':>6s}")
    for row in rows:
        builtin, plugin = row["builtin"], row["plugin"]
        print(f"{row['size']:6d} {builtin['prompt_tokens']:9d} {hit_of(builtin):7.1%} "
              f"{builtin['cost_usd']:10.6f} {plugin['prompt_tokens']:9d} {hit_of(plugin):7.1%} "
              f"{plugin['cost_usd']:10.6f} {row['ratio']:6.2f}")
    for size in sorted({row["size"] for row in rows}):
        subset = [row for row in rows if row["size"] == size]
        served = [row for row in subset if row["hit"] >= 0.85]
        part = (f"   served (>=85%): {statistics.mean(row['ratio'] for row in served):.2f}"
                if served else "")
        print(f"  {size:6d}: mean p/b {statistics.mean(row['ratio'] for row in subset):.2f}{part}")


if __name__ == "__main__":
    whole_compaction()
    pinned_output()
    sys.exit(0)
