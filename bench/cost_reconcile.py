"""Reconcile the cost of a benchmark run across its three sources.

1. the router, which reports what it bills per call (``usage.opencode.list_cost``);
2. Langfuse, which recomputes from the usage it ingested for the same run;
3. the local estimate, tokens times list prices.

The point is not to pick a winner but to know which source can be trusted for which decision: the router
knows its discounts, Langfuse knows the shape of the traffic, the local estimate knows the public list price.

Usage:

    LANGFUSE_BASE=http://localhost:3000 LANGFUSE_PUBLIC=... LANGFUSE_SECRET=... \
    python bench/cost_reconcile.py bench/results/arms-20261111-120000.json
"""

from __future__ import annotations

import base64
import json
import os
import sys
import urllib.parse
import urllib.request

BASE = os.environ.get("LANGFUSE_BASE", "http://localhost:3000")
PUBLIC = os.environ.get("LANGFUSE_PUBLIC", "")
SECRET = os.environ.get("LANGFUSE_SECRET", "")


def observations(start: str, end: str) -> list[dict]:
    """Every generation Langfuse holds in the window, paginated by 100."""
    auth = base64.b64encode(f"{PUBLIC}:{SECRET}".encode()).decode()
    kept, page = [], 1
    while True:
        query = urllib.parse.urlencode({"type": "GENERATION", "fromStartTime": start,
                                        "toStartTime": end, "limit": 100, "page": page})
        request = urllib.request.Request(f"{BASE}/api/public/observations?{query}",
                                         headers={"Authorization": f"Basic {auth}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
        rows = payload.get("data") or []
        kept.extend(rows)
        if page >= int((payload.get("meta") or {}).get("totalPages") or 1):
            return kept
        page += 1


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    with open(sys.argv[1], encoding="utf-8") as handle:
        report = json.load(handle)
    start, end = report["window"]
    rows = [r for r in observations(start, end) if "compaction-" in str(r.get("name") or "")]

    langfuse: dict[str, dict] = {}
    for row in rows:
        arm = str(row.get("name")).split("-", 1)[1]
        usage = row.get("usage") or {}
        bucket = langfuse.setdefault(arm, {"calls": 0, "input": 0, "output": 0,
                                           "cache_read": 0, "latency_s": 0.0})
        bucket["calls"] += 1
        bucket["input"] += int(usage.get("input") or 0)
        bucket["output"] += int(usage.get("output") or 0)
        bucket["cache_read"] += int((row.get("usageDetails") or {}).get("cache_read") or 0)
        bucket["latency_s"] = round(bucket["latency_s"] + float(row.get("latency") or 0.0), 2)

    print(f"{'bras':4s} {'appels':>6s} {'entree':>8s} {'sortie':>7s} "
          f"{'routeur':>10s} {'langfuse':>10s} {'estimation':>11s}")
    for arm, total in sorted(report["totals"].items()):
        seen = langfuse.get(arm, {})
        print(f"{arm:4s} {seen.get('calls', 0):6d} {seen.get('input', 0):8d} "
              f"{seen.get('output', 0):7d} {total['cost_router_usd']:10.5f} "
              f"{'-':>10s} {total['cost_estimate_usd']:11.5f}")
    print("\nlangfuse porte l'usage et la latence, pas le prix : il ne connait le cout que si un modele "
          "tarife est declare. Le rapprochement utile est donc usage et latence contre le rapport, "
          "et le cout se lit sur le routeur ou sur l'estimation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
