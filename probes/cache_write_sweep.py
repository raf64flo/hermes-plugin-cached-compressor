"""Fenêtre d'écriture du cache : sonde 0/2/5/10 s de la note de durcissement.

    BENCH_API_KEY=... python probes/cache_write_sweep.py
    BENCH_API_KEY=... python probes/cache_write_sweep.py --sweep 0,2,5,10 --repeats 6 --tokens 4000 \
        --out results.json

Pour chaque délai, envoie le même contenu N fois dans la même session, avec ce délai entre deux
envois, et note le hit de chaque appel. La question de la note : l'écriture du cache arrive-t-elle
assez tard pour qu'un délai dans le chemin de compaction change quelque chose ? Sans effet
monotone du délai sur le hit, pas de knob : dormir pénaliserait tous les tours sans garantie.

Le JSON de sortie est publiable : compteurs et timings seulement, aucun contenu.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.parse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cache_report_check import _as_int, _post, _usage_fields  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.environ.get(
        "BENCH_BASE", "https://opencode.ai/inference/go/openai/v1"))
    parser.add_argument("--model", default=os.environ.get("BENCH_MODEL", "deepseek-v4.1-flash"))
    parser.add_argument("--tokens", type=int, default=4_000, help="taille approximative du prompt")
    parser.add_argument("--repeats", type=int, default=6, help="envois par délai")
    parser.add_argument("--sweep", default="0,2,5,10", help="délais en secondes, séparés par des virgules")
    parser.add_argument("--out", default=None, help="écrire le résultat JSON (compteurs seulement)")
    args = parser.parse_args()

    key = os.environ.get("BENCH_API_KEY") or os.environ.get("HERMES_API_KEY")
    if not key:
        print("BENCH_API_KEY absent", file=sys.stderr)
        return 2

    delays = []
    for item in args.sweep.split(","):
        value = float(item.strip())
        if value < 0:
            print(f"délai invalide : {item}", file=sys.stderr)
            return 2
        delays.append(value)

    paragraph = ("Audit du module src/state.py : le slot conserve le dernier payload fil observé. "
                 "Décision : borner en octets et libérer à la fin de session. ")
    messages = [{"role": "user",
                 "content": paragraph * max(1, int(args.tokens * 3.6 / len(paragraph)))}]

    runs: list[dict] = []
    for delay in delays:
        session = f"cache-sweep-{delay:g}-{os.getpid()}"
        attempts: list[dict] = []
        for attempt in range(1, args.repeats + 1):
            payload = _post(args.base, key, args.model, messages, session, max_tokens=8)
            usage = payload.get("usage") or {}
            found = _usage_fields(usage)
            prompt = usage.get("prompt_tokens") or 0
            cached = next((_as_int(value) for value in found.values() if _as_int(value) is not None), 0)
            hit = cached / prompt if prompt else 0.0
            record = {"attempt": attempt, "prompt_tokens": prompt, "cache_read_tokens": cached,
                      "hit_ratio": round(hit, 4), "prefill_s": payload["_elapsed_s"]}
            attempts.append(record)
            print(f"délai {delay:g}s appel {attempt}/{args.repeats}: prompt={prompt} "
                  f"hit={hit:.1%} prefill={payload['_elapsed_s']:.2f}s")
            if attempt < args.repeats and delay > 0:
                time.sleep(delay)
        first_hit = next((a["attempt"] for a in attempts if a["hit_ratio"] >= 0.5), None)
        last_hits = [a["hit_ratio"] for a in attempts[-3:]]
        runs.append({"delay_s": delay, "attempts": attempts, "first_hit_attempt": first_hit,
                     "mean_last_3_hit": round(sum(last_hits) / len(last_hits), 4)
                     if last_hits else None})

    # Effet monotone : le hit moyen de fin de série augmente-t-il avec le délai ?
    ordered = sorted(runs, key=lambda run: run["delay_s"])
    means = [run["mean_last_3_hit"] for run in ordered if run["mean_last_3_hit"] is not None]
    monotone = all(b >= a - 0.05 for a, b in zip(means, means[1:])) and len(means) == len(ordered)
    spread = (max(means) - min(means)) if means else 0.0
    if monotone and spread >= 0.10:
        verdict = ("effet monotone : le hit de fin de série croît avec le délai ; "
                   "un réglage de fenêtre est justifié, à confirmer sur d'autres backends")
    else:
        verdict = ("pas d'effet monotone : l'écriture du cache n'attend pas ce délai ; "
                   "pas de knob, pas de sommeil dans le chemin de compaction")
    print()
    print("résumé :")
    for run in ordered:
        pattern = " ".join(f"{a['hit_ratio']:.0%}" for a in run["attempts"])
        print(f"  délai {run['delay_s']:>4g}s  hits [{pattern}]  "
              f"premier hit >= 50% : {run['first_hit_attempt'] or 'jamais'}")
    print(f"VERDICT : {verdict}")

    if args.out:
        host = urllib.parse.urlparse(args.base).netloc
        report = {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": host,
                  "model": args.model, "target_tokens": args.tokens, "repeats": args.repeats,
                  "runs": runs, "monotone": monotone, "spread": round(spread, 4), "verdict": verdict}
        path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"JSON écrit : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
