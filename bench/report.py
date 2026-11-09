"""Aggregate the benchmark records into the tables published in docs/BENCHMARK.md.

    python bench/report.py            # reads bench/results/*.json

Two kinds of record:

- ``bench-*.json``      whole compaction, both engines, output length free
- ``input-only-*.json`` same payloads with the output pinned, so the bill isolates the prompt

Cost differences are reported as a median with a percentile bootstrap interval, per size and for the
runs the cache actually served, because the run count is small and the spread is wide.
"""

from __future__ import annotations

import glob
import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SERVED = 0.90
RESAMPLES = 10_000


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


def bootstrap(values: list[float], seed: int = 7) -> tuple[float, float]:
    """Percentile bootstrap interval of the mean, at 95 %."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(RESAMPLES))
    return means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]


def whole_compaction() -> None:
    rows = []
    for record in load("bench-*.json"):
        config = record.get("aux_model") or "same model"
        for run in record["runs"]:
            builtin, plugin = run["arms"]["builtin"], run["arms"]["plugin"]
            rows.append({
                "config": config,
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

    for config in sorted({row["config"] for row in rows}):
        config_rows = [row for row in rows if row["config"] == config]
        label = ("built-in on the session model" if config == "same model"
                 else f"built-in on {config}")
        print(f"\n## Whole compaction: {label}\n")
        print(f"{'pass':16s} {'size':>6s} {'hit':>7s} {'in x':>6s} {'out':>7s} {'b cost':>10s} "
              f"{'p cost':>10s} {'cost':>7s} {'lat':>7s}  judge (b/p)")
        for row in config_rows:
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

        served = [row for row in config_rows if row["hit"] >= SERVED]
        groups = [("all runs", config_rows)]
        for size in sorted({row["size"] for row in config_rows}):
            groups.append((f"{size} tokens", [row for row in config_rows if row["size"] == size]))
        groups.append((f"prefix served (>={SERVED:.0%})", served))
        groups.append(("prefix not served", [row for row in config_rows if row["hit"] < SERVED]))

        print(f"\nruns: {len(config_rows)}   plugin cache served: {len(served)}")
        print(f"  {'subset':22s} {'n':>3s} {'cost median':>12s} {'CI95':>18s} {'mean':>8s} "
              f"{'latency':>8s} {'output':>7s} {'input':>6s}")
        for name, subset in groups:
            if not subset:
                continue
            values = [row["d_cost"] for row in subset]
            low, high = bootstrap(values)
            print(f"  {name:22s} {len(values):3d} {statistics.median(values):+11.1f}% "
                  f"[{low:+7.1f}, {high:+6.1f}] {statistics.mean(values):+7.1f}% "
                  f"{statistics.mean(row['d_lat'] for row in subset):+7.1f}% "
                  f"{statistics.mean(row['out_ratio'] for row in subset):+6.1f}% "
                  f"x{statistics.mean(row['in_ratio'] for row in subset):4.1f}")

        judges = [(row["judge"].get("totals") or {}, row["judge"])
                  for row in config_rows if row["judge"]]
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


def price_effect() -> None:
    """Break-even price ratio: how much cheaper the auxiliary model may be before the plugin loses.

    The plugin pays the session model for the uncached part of the prefix and the output; the
    built-in pays the auxiliary model for its whole (freshly serialized) window and output. Writing
    ``R`` for session price / auxiliary price, the plugin wins while
    ``R < (builtin tokens at session price) / (plugin billed cost)``.
    """
    rows = []
    for record in load("bench-*.json"):
        prices = record.get("prices") or {}
        if record.get("model") not in prices:
            continue
        session = prices[record["model"]]
        for run in record["runs"]:
            builtin, plugin = run["arms"]["builtin"], run["arms"]["plugin"]
            aux = prices.get(builtin.get("model") or record["model"], session)
            full = (builtin["prompt_tokens"] * session["input"]
                    + builtin["completion_tokens"] * session["output"]) / 1e6
            if not plugin["cost_usd"] or not full:
                continue
            rows.append({
                "size": run["target_tokens"],
                "aux_model": builtin.get("model") or record["model"],
                "session": session,
                "aux": aux,
                "r_max": full / plugin["cost_usd"],
                "billed_delta": 100 * (plugin["cost_usd"] / builtin["cost_usd"] - 1)
                if builtin["cost_usd"] else None,
            })
    if not rows:
        print("\nno price sheet in the records: rerun bench.py to capture it")
        return

    print("\n## Price of the auxiliary model\n")
    seen = {}
    for row in rows:
        seen.setdefault((row["session"]["input"], row["session"]["output"],
                         row["aux"]["input"], row["aux"]["output"]), 0)
    for (s_in, s_out, a_in, a_out) in seen:
        if (s_in, s_out) == (a_in, a_out):
            continue
        print(f"session {s_in}/{s_out} USD/M in/out, auxiliary {a_in}/{a_out} USD/M")
    print(f"  {'size':>6s} {'aux model':22s} {'R break-even':>12s} {'billed':>8s}")
    for size in sorted({row["size"] for row in rows}):
        subset = [row for row in rows if row["size"] == size]
        values = [row["r_max"] for row in subset]
        billed = [row["billed_delta"] for row in subset if row["billed_delta"] is not None]
        name = subset[0]["aux_model"]
        print(f"  {size:6d} {name:22s} {statistics.median(values):11.1f}x "
              f"{statistics.median(billed):+7.1f}%" if billed else
              f"  {size:6d} {name:22s} {statistics.median(values):11.1f}x")
    print("  the plugin wins while the session model costs less than the break-even multiple of the")
    print("  auxiliary model; above it, summarizing with the cheap model is cheaper.")


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
    price_effect()
    pinned_output()
    sys.exit(0)
