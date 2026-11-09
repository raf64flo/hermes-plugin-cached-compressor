"""Ce backend rapporte-t-il ses lectures de cache, et sous quel champ ?

    BENCH_API_KEY=... python probes/cache_report_check.py
    BENCH_API_KEY=... python probes/cache_report_check.py --base URL --model ID --tokens 4000

Envoie deux fois le même prompt long, dans la même session, et lit l'usage de chaque réponse. Le
premier envoi remplit le cache, le second le relit. Trois choses en sortent : le champ que ce backend
utilise, le hit réellement servi, et le temps de prefill de chaque appel.

Un backend muet n'est pas un backend sans cache : il peut en avoir un et ne pas le facturer. Dans ce
cas le plugin repasse au compresseur intégré (deux mesures muettes), et le gain éventuel est un gain
de temps, pas d'argent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

#: Champs connus, dans l'ordre où le plugin les essaie.
FIELDS = ("cached_tokens", "prompt_cache_hit_tokens", "cache_read_input_tokens",
          "cached_content_token_count", "cachedContentTokenCount")
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/131.0 Safari/537.36")


def _post(base: str, key: str, model: str, messages: list[dict], session: str,
          max_tokens: int, retries: int = 3) -> dict:
    """POST one chat completion. Honors ``retry-after`` on a 429 (usage-limit window),
    up to ``retries`` times, capped at two minutes per wait; anything else fails fast."""
    last_error: urllib.error.HTTPError | None = None
    for attempt in range(retries + 1):
        body = {"model": model, "messages": messages, "max_tokens": max_tokens}
        request = urllib.request.Request(
            base.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                     "User-Agent": USER_AGENT, "x-opencode-session": session})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as err:
            if err.code == 429 and attempt < retries:
                retry_after = err.headers.get("retry-after")
                try:
                    delay = min(float(retry_after), 120.0) if retry_after else 15.0
                except ValueError:
                    delay = 15.0
                print(f"  429 (appel {attempt + 1}); retry-after={retry_after!r}, attente {delay:0.0f}s")
                last_error = err
                time.sleep(delay)
                continue
            err_body = err.read().decode("utf-8", errors="replace")[:400]
            raise SystemExit(f"HTTP {err.code} {err_body}") from err
        else:
            payload["_elapsed_s"] = round(time.monotonic() - started, 2)
            return payload
    raise SystemExit(f"HTTP 429 après {retries} retries") from last_error


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _usage_fields(usage: dict) -> dict[str, object]:
    """Tous les champs de cache présents, à la racine comme dans ``prompt_tokens_details``."""
    details = usage.get("prompt_tokens_details") or {}
    found: dict[str, object] = {}
    for field in FIELDS:
        if field in details:
            found[f"prompt_tokens_details.{field}"] = details[field]
        if field in usage:
            found[field] = usage[field]
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=os.environ.get(
        "BENCH_BASE", "https://opencode.ai/inference/go/openai/v1"))
    parser.add_argument("--model", default=os.environ.get("BENCH_MODEL", "deepseek-v4.1-flash"))
    parser.add_argument("--tokens", type=int, default=4_000, help="taille approximative du prompt")
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="délai entre deux envois : l'écriture de cache arrive après la réponse")
    args = parser.parse_args()

    key = os.environ.get("BENCH_API_KEY") or os.environ.get("HERMES_API_KEY")
    if not key:
        print("BENCH_API_KEY absent", file=sys.stderr)
        return 2

    paragraph = ("Audit du module src/state.py : le slot conserve le dernier payload fil observé. "
                 "Décision : borner en octets et libérer à la fin de session. ")
    messages = [{"role": "user",
                 "content": paragraph * max(1, int(args.tokens * 3.6 / len(paragraph)))}]
    session = f"cache-check-{os.getpid()}"

    seen: set[str] = set()
    hits: list[float] = []
    elapsed: list[float] = []
    for attempt in range(1, args.repeat + 1):
        payload = _post(args.base, key, args.model, messages, session, max_tokens=8)
        usage = payload.get("usage") or {}
        found = _usage_fields(usage)
        prompt = usage.get("prompt_tokens") or 0
        cached = next((_as_int(value) for value in found.values() if _as_int(value) is not None), 0)
        hit = cached / prompt if prompt else 0.0
        hits.append(hit)
        elapsed.append(payload["_elapsed_s"])
        seen.update(found)
        print(f"appel {attempt}: prompt={prompt} champs={found or 'aucun'} "
              f"hit={hit:.1%} prefill={payload['_elapsed_s']:.2f}s")
        if attempt < args.repeat and args.sleep > 0:
            time.sleep(args.sleep)

    print()
    if not seen:
        print("VERDICT : ce backend ne rapporte aucun champ de cache.")
        print("          Le plugin le traitera comme une route sans cache et repassera au natif.")
        return 1
    print(f"VERDICT : champ(s) rapporté(s) -> {', '.join(sorted(seen))}")
    print(f"          hit du dernier appel {hits[-1]:.1%}, prefill {elapsed[0]:.2f}s -> "
          f"{elapsed[-1]:.2f}s")
    if hits[-1] >= 0.5:
        print("          Cache servi : le mode préfixe a un sens sur cette route.")
    else:
        print("          Aucun hit au second envoi : soit l'écriture est asynchrone (réessayer),")
        print("          soit le cache est désactivé sur ce backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
