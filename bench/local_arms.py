"""Le cache de prefixe, mesure sur un serveur local.

Le banc a quatre bras mesure un cout facture par un routeur dont l'etat de cache n'est pas controlable. Ici,
la meme question est posee a `llama-server`, qui tient un seul cache KV, publie `cache_n` et `prompt_n` par
requete, et ne route rien. Ce que ce banc etablit, c'est le mecanisme : le materiau etant deja passe sur le
serveur, quelle forme de requete en reutilise le prefixe et quelle forme le jette.

Toutes les requetes sont sequentielles, un seul slot, temperature 0.

    export LOCAL_BASE=http://127.0.0.1:8090
    python bench/local_arms.py --sessions 2 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize  # noqa: E402
import compaction_arms as arms  # noqa: E402

BASE = os.environ.get("LOCAL_BASE", "http://127.0.0.1:8090")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUCTION = arms.INSTRUCTION


def flatten(messages: list[dict]) -> str:
    """Le materiau tel qu'il a deja ete envoye, mis a plat pour l'endpoint /completion."""
    return "".join(f"{m.get('role', 'user')}: {m.get('content') or ''}\n" for m in messages)


def call(prompt: str, n_predict: int, timeout: int = 900) -> dict:
    body = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0,
                       "cache_prompt": True}).encode()
    request = urllib.request.Request(f"{BASE}/completion", data=body,
                                     headers={"Content-Type": "application/json"})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    timings = payload.get("timings", {})
    return {"cache_n": timings.get("cache_n"), "prompt_n": timings.get("prompt_n"),
            "prompt_ms": round(timings.get("prompt_ms", 0), 1),
            "decode_tps": round(timings.get("predicted_per_second", 0), 1),
            "out": timings.get("predicted_n"),
            "wall_s": round(time.monotonic() - started, 2)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default=os.path.join(REPO, "bench", "results"))
    args = parser.parse_args()

    report: dict = {"server": BASE, "instruction": INSTRUCTION, "passes": []}
    for session in anonymize.load_sessions(args.sessions, 12_000):
        material = flatten(arms.sanitize(session["messages"][:-arms.PROTECTED_TAIL]))
        print(f"\n{session['session']}  materiau {len(material)} caracteres")
        for repeat in range(args.repeats):
            row = {"session": session["session"], "repeat": repeat + 1}
            # La conversation vient de passer : c'est l'hypothese du plugin.
            row["prime"] = call(material, 1)
            # La forme du plugin : meme prefixe, instruction ajoutee a la fin.
            row["B"] = call(material + "\n" + INSTRUCTION, 64)
            # La forme native : instruction en tete, donc prefixe different des le premier token.
            row["C"] = call(INSTRUCTION + "\n" + material, 64)
            # Meme forme une seconde fois : le cache se reamorce, ce qui prouve qu'il suit le prefixe.
            row["C2"] = call(INSTRUCTION + "\n" + material, 64)
            print(f"  passe {repeat + 1}  primer cache_n={row['prime']['cache_n']:>5} | "
                  f"B cache_n={row['B']['cache_n']:>5} lecture={row['B']['prompt_ms']:>7.1f}ms | "
                  f"C cache_n={row['C']['cache_n']:>5} lecture={row['C']['prompt_ms']:>7.1f}ms | "
                  f"C2 cache_n={row['C2']['cache_n']:>5} lecture={row['C2']['prompt_ms']:>7.1f}ms")
            report["passes"].append(row)

    rows = report["passes"]
    # Seule la premiere passe d'un materiau est froide par construction : les suivantes trouvent le prefixe
    # deja en cache, y compris pour la forme native, et ne distinguent donc plus rien.
    first = [r for r in rows if r["repeat"] == 1]
    # Un materiau deja passe sur le serveur garde son prefixe en cache : seule une premiere passe dont
    # l'amorcage relit tout est reellement froide.
    cold = [r for r in first if r["prime"]["cache_n"] == 0]
    excluded = len(first) - len(cold)
    print(f"\n{len(cold)} materiau(x) froid(s) par construction"
          + (f", {excluded} exclu(s) : prefixe deja en cache" if excluded else ""))
    first = cold
    for arm in ("B", "C"):
        cache = statistics.median([r[arm]["cache_n"] for r in first])
        read = statistics.median([r[arm]["prompt_n"] for r in first])
        prompt_ms = statistics.median([r[arm]["prompt_ms"] for r in first])
        wall = statistics.median([r[arm]["wall_s"] for r in first])
        print(f"  {arm:3s} cache_n={cache:6.0f}  tokens relus={read:6.0f}  "
              f"lecture={prompt_ms:8.1f}ms  appel={wall:5.2f}s")
    if len(rows) > len(first):
        repeats = [r for r in rows if r["repeat"] > 1]
        print(f"regime etabli sur {len(repeats)} passe(s) supplementaire(s) : "
              f"B lecture={statistics.median([r['B']['prompt_ms'] for r in repeats]):.1f}ms, "
              f"C lecture={statistics.median([r['C']['prompt_ms'] for r in repeats]):.1f}ms "
              "(les deux prefixes sont alors en cache, l'ecart disparait)")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, time.strftime("local-arms-%Y%m%d-%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
