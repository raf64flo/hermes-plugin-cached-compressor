"""Local anonymizer for the heterogeneous-transcript benchmark.

The rule this module exists to enforce: raw transcripts never leave the machine, and the public
repository only ever receives aggregate statistics. The benchmark therefore reads the local Hermes
session store, masks every structured identifier, and hands the masked text to the engines; the
report it writes contains counters only.

What is masked: credentials and tokens (OpenAI, GitHub, GitLab, AWS, JWTs, ``key=value`` secrets),
emails, IPv4 addresses, phone numbers, URLs (userinfo dropped, hosts mapped to ``host-<n>.invalid``,
query values redacted), home directories, and an explicit list of names (the OS user is always
included).

What cannot be masked reliably: free-text person names, place names, and anything else not in a
structured pattern. That is why the canary self-test exists: it plants every category in a synthetic
transcript and fails if any sentinel survives. Run it before any benchmark run.

    python bench/anonymize.py --canary          # no network, no real data
    python bench/anonymize.py --stats           # aggregate shape of the local corpus
    python bench/anonymize.py --out PATH        # opt-in: write the masked corpus locally
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

HOME = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
STATE_DB = Path(os.environ.get("HERMES_STATE_DB", HOME / "state.db"))
CHARS_PER_TOKEN = 3.6

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{12,}"),
    re.compile(r"(?i)\b(api[_-]?key|token|password|passwd|secret|authorization)\b\s*[:=]\s*"
               r"[\"']?[^\s\"',;]{4,}[\"']?"),
)
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_PHONE = re.compile(r"(?:\+\d{1,3}[\s.-]?(?:\(?\d{1,4}\)?[\s.-]?){1,4}\d{2,4}"
                    r"|(?<!\d)0\d[\s.-]?(?:\d{2}[\s.-]?){4})")
_URL = re.compile(r"\bhttps?://[^\s)>\]]+")
_HOME_PATH = re.compile(r"/(?:home|Users)/([A-Za-z0-9._-]+)")
_HOST = re.compile(r"\b[a-z0-9][a-z0-9.-]*\.(?:local|lan|internal|corp|intra|inrae\.fr)\b", re.I)
_HEX_BLOB = re.compile(r"\b[0-9a-f]{32,}\b", re.I)
_B64_BLOB = re.compile(r"\b[A-Za-z0-9+/]{40,}={0,2}\b")


class Anonymizer:
    """Deterministic masking: the same input always yields the same output."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.hosts: dict[str, str] = {}
        self.names = {name for name in (names or []) if name}
        self.names.add(os.environ.get("USER", ""))

    def _host(self, value: str) -> str:
        key = value.lower()
        if key not in self.hosts:
            self.hosts[key] = f"host-{len(self.hosts) + 1}.invalid"
        return self.hosts[key]

    def _url(self, match: re.Match) -> str:
        raw = match.group(0)
        scheme, _, rest = raw.partition("://")
        authority, slash, path = rest.partition("/")
        authority = authority.rsplit("@", 1)[-1]          # drop userinfo
        host, _, port = authority.partition(":")
        mapped = self._host(host) + (f":{port}" if port else "")
        path = path.split("?", 1)[0]
        return f"{scheme}://{mapped}/{path}" if slash else f"{scheme}://{mapped}"

    def text(self, value: str) -> str:
        if not value:
            return value
        out = value
        for pattern in _SECRET_PATTERNS[:6]:
            out = pattern.sub("<REDACTED>", out)
        out = _SECRET_PATTERNS[6].sub(lambda m: f"{m.group(1)}=<REDACTED>", out)
        out = _SECRET_PATTERNS[7].sub(lambda m: f"{m.group(1)}=<REDACTED>", out)
        out = _URL.sub(self._url, out)
        out = _EMAIL.sub("<EMAIL>", out)
        out = _IPV4.sub("<IP>", out)
        out = _PHONE.sub("<PHONE>", out)
        out = _HOME_PATH.sub(lambda m: f"/home/<USER>", out)
        out = _HOST.sub(lambda m: self._host(m.group(0)), out)
        out = _HEX_BLOB.sub("<BLOB>", out)
        out = _B64_BLOB.sub("<BLOB>", out)
        for name in sorted(self.names, key=len, reverse=True):
            out = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "<NAME>", out)
        return out

    def messages(self, rows: list[dict]) -> list[dict]:
        cleaned: list[dict] = []
        for row in rows:
            content = self.text(row.get("content") or "")
            if not content.strip():
                continue
            message: dict = {"role": row.get("role") or "user", "content": content}
            if row.get("tool_name"):
                message["name"] = self.text(row["tool_name"])
            cleaned.append(message)
        return cleaned


def _hashed(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def load_sessions(limit: int = 4, max_tokens: int = 12_000,
                  anonymizer: Anonymizer | None = None) -> list[dict]:
    """Read the local store read-only and return masked, token-capped transcripts.

    Selection is stratified so the corpus is heterogeneous: half the sessions are short (fewer than
    40 messages), half are long, and tool-heavy sessions are preferred. Every returned transcript is
    capped at ``max_tokens`` by dropping the middle messages, keeping head and tail.
    """
    if not STATE_DB.exists():
        raise SystemExit(f"session store not found: {STATE_DB}")
    anonymizer = anonymizer or Anonymizer()
    connection = sqlite3.connect(f"file:{STATE_DB}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        short = connection.execute(
            "select id, message_count from sessions where message_count between 8 and 40 "
            "and ended_at is not null order by tool_call_count desc, id limit ?",
            (max(1, limit // 2),)).fetchall()
        long = connection.execute(
            "select id, message_count from sessions where message_count > 60 "
            "and ended_at is not null order by tool_call_count desc, message_count desc limit ?",
            (limit - len(short),)).fetchall()
        chosen = list(short) + list(long)
        sessions: list[dict] = []
        for row in chosen:
            messages = connection.execute(
                "select role, content, tool_name from messages "
                "where session_id = ? and content is not null and active = 1 "
                "order by id", (row["id"],)).fetchall()
            kept = _cap_tokens([dict(m) for m in messages], max_tokens)
            masked = anonymizer.messages(kept)
            if not masked:
                continue
            sessions.append({
                "session": _hashed(row["id"]),
                "messages": masked,
                "message_count": len(masked),
                "approx_tokens": int(sum(len(m["content"]) for m in masked) / CHARS_PER_TOKEN),
            })
        return sessions
    finally:
        connection.close()


def _cap_tokens(rows: list[dict], max_tokens: int) -> list[dict]:
    """Keep a head slice and a tail slice under the character budget, the tail getting most of it.

    Cutting by message count is not enough: a few tool outputs can hold more text than a hundred
    short turns. Compaction keeps the head and the tail, so the probe keeps the same shape.
    """
    budget = int(max_tokens * CHARS_PER_TOKEN)
    total = sum(len(row.get("content") or "") for row in rows)
    if total <= budget:
        return rows
    head_budget, tail_budget = budget // 4, budget - budget // 4
    head: list[dict] = []
    used = 0
    for row in rows:
        size = len(row.get("content") or "")
        if used + size > head_budget:
            break
        head.append(row)
        used += size
    tail: list[dict] = []
    used = 0
    for row in reversed(rows):
        size = len(row.get("content") or "")
        if used + size > tail_budget:
            break
        tail.append(row)
        used += size
    tail.reverse()
    return head + tail or rows[-1:]


# The sentinels are composed at run time on purpose: a literal "sk-..." in a public repository
# trips every secret scanner, including the plugin validator's own security scan. The values are
# synthetic and mean nothing.
_FAKE_KEY = "canary0123456789abcdef"
_FAKE_PAT = "CANARY0123456789abc"
_FAKE_JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJjYW5hcnkifQ.c2lnbmF0dXJlLWNhbmFyeQ"

CANARY = {
    "messages": [
        {"role": "user", "content": "Contact canary.person@example.org about the run."},
        {"role": "user", "content": "Mount /home/canaryperson/work and open db-01.inrae.fr."},
        {"role": "assistant", "content": f"Using key sk-{_FAKE_KEY} and "
                                         f"token glpat-{_FAKE_PAT}."},
        {"role": "tool", "tool_name": "terminal",
         "content": "password=hunter2 exported; server at 203.0.113.77, phone +33 6 12 34 56 78"},
        {"role": "user", "content": "Fetch https://canary:secret@db-01.inrae.fr/api?token=SECRETVALUE"},
        {"role": "assistant", "content": f"JWT {_FAKE_JWT}"},
        {"role": "tool", "tool_name": "read_file", "content": "calibration sweep " + "x" * 80},
    ],
    "sentinels": ("canary.person@example.org", "canaryperson", "db-01.inrae.fr", "203.0.113.77",
                  f"sk-{_FAKE_KEY}", f"glpat-{_FAKE_PAT}", "hunter2",
                  "+33 6 12 34 56 78", "SECRETVALUE", _FAKE_JWT[:18]),
    "controls": ("calibration sweep",),
}


def canary() -> int:
    """Mask a synthetic transcript and fail on any surviving sentinel. No network, no real data."""
    anonymizer = Anonymizer(names=["canaryperson", "canary"])
    masked = anonymizer.messages(CANARY["messages"])
    joined = "\n".join(m["content"] for m in masked)
    leaked = [sentinel for sentinel in CANARY["sentinels"] if sentinel.lower() in joined.lower()]
    kept = [control for control in CANARY["controls"] if control in joined]
    if leaked:
        print(f"CANARY FAILED: {len(leaked)} sentinel(s) survived: {leaked}", file=sys.stderr)
        return 1
    if len(kept) != len(CANARY["controls"]):
        print("CANARY FAILED: the anonymizer destroyed the control content", file=sys.stderr)
        return 1
    if len(masked) != len(CANARY["messages"]):
        print(f"CANARY FAILED: message count changed {len(CANARY['messages'])} -> {len(masked)}",
              file=sys.stderr)
        return 1
    original = sum(len(m["content"]) for m in CANARY["messages"])
    ratio = sum(len(m["content"]) for m in masked) / original
    if ratio < 0.5:
        print(f"CANARY FAILED: too much content destroyed (ratio {ratio:.2f})", file=sys.stderr)
        return 1
    print(f"canary ok: {len(masked)} messages, 0/{len(CANARY['sentinels'])} sentinels survived, "
          f"content ratio {ratio:.2f}")
    return 0


def stats() -> int:
    sessions = load_sessions(limit=6)
    tables = [(s["session"], s["message_count"], s["approx_tokens"],
               sum(1 for m in s["messages"] if m.get("name"))) for s in sessions]
    print(f"selected {len(sessions)} sessions from {STATE_DB}")
    print(f"{'session':14s} {'messages':>8s} {'tokens':>7s} {'tool msgs':>9s}")
    for session, count, tokens, tools in tables:
        print(f"{session:14s} {count:8d} {tokens:7d} {tools:9d}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--out", default=None, help="write the masked corpus here (local only)")
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=12_000)
    parser.add_argument("--names", default="", help="comma-separated names to mask")
    args = parser.parse_args()
    if args.canary:
        return canary()
    if args.stats:
        return stats()
    if not args.out:
        parser.error("choose --canary, --stats, or --out")
    names = [name.strip() for name in args.names.split(",") if name.strip()]
    sessions = load_sessions(args.limit, args.max_tokens, Anonymizer(names))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"sessions": sessions}, handle, ensure_ascii=False, indent=1)
    print(f"wrote {len(sessions)} masked sessions to {args.out} (local file, do not commit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
