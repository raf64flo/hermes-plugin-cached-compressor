"""Does reusing the session make the compression call faster?

Same material, same instruction, same output budget, same model. The only difference is the session: warm in
B, never used in B'. Each arm runs three times, alternating, and the median is reported.

Timing a single call cannot answer this: a backoff, a queue or a slow instance lands in the same number. Two
things make it measurable here. Every call is timed on its successful attempt only, retries excluded. And the
decoding part is subtracted with a slope measured beforehand, so what remains is the time spent reading the
prompt, which is exactly what a warm prefix removes.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anonymize  # noqa: E402
import bench as harness  # noqa: E402
import compaction_arms as arms  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--pause", type=float, default=3.0)
    parser.add_argument("--per-token", type=float, default=0.0)
    parser.add_argument("--out", default=os.path.join(harness.REPO, "bench", "results"))
    args = parser.parse_args()

    sessions = anonymize.load_sessions(args.sessions, 12_000)
    sample = arms.sanitize(sessions[0]["messages"][:-arms.PROTECTED_TAIL])
    per_token = args.per_token
    for attempt in range(3):
        if per_token:
            break
        per_token = arms.calibrate(sample, args.pause)
        if not per_token:
            print(f"calibration attempt {attempt + 1} failed, retrying", file=sys.stderr)
            time.sleep(5)
    if not per_token:
        print("calibration failed three times, refusing to report timings", file=sys.stderr)
        return 2
    print(f"calibration: {per_token} s par token de sortie ({1 / per_token:.0f} tokens/s)")

    report: dict = {"per_token_s": per_token, "max_tokens": args.max_tokens, "sessions": []}
    for session in sessions:
        material = arms.sanitize(session["messages"][:-arms.PROTECTED_TAIL])
        instruction = {"role": "user", "content": arms.INSTRUCTION}
        warm_session = f"warm-{session['session']}"
        cold_session = f"cold-{session['session']}-{uuid.uuid4().hex[:6]}"

        warm = arms.Transport(harness.BASE, harness.MODEL, harness.KEY, {"max_tokens": 1},
                              session=warm_session)
        warm(messages=material, tools=harness.TOOLS)
        time.sleep(args.pause)

        record = {"session": session["session"], "arms": {}}
        # B' is not a cold control: a session never used still hits the prefix cache when the instance
        # already holds the blocks. C is cold for a different reason, a different first message, and that is
        # what isolates the prefill.
        for arm_name, chosen in (("B", warm_session), ("C", warm_session)):
            messages, _ = arms.build(arm_name, material)
            readings = []
            for _ in range(args.repeats):
                transport = arms.Transport(harness.BASE, harness.MODEL, harness.KEY,
                                           {"max_tokens": args.max_tokens}, session=chosen)
                transport(messages=messages, tools=harness.TOOLS)
                call = transport.calls[-1]
                readings.append(call)
                time.sleep(args.pause)
            service = [c["service_s"] for c in readings]
            outputs = [c["completion_tokens"] for c in readings]
            prefill = [round(s - o * per_token, 2) for s, o in zip(service, outputs)]
            record["arms"][arm_name] = {
                "repeats": args.repeats,
                "hit": round(statistics.median([c["cached_tokens"] / max(c["prompt_tokens"], 1)
                                                for c in readings]), 3),
                "service_s": service,
                "output": outputs,
                "prefill_est_s": prefill,
                "prefill_median_s": round(statistics.median(prefill), 2),
            }
            print(f"{session['session']} {arm_name:2s} hit={record['arms'][arm_name]['hit']:.2f} "
                  f"service={service} sortie={outputs} prefill~{prefill} "
                  f"median={record['arms'][arm_name]['prefill_median_s']}s")
        report["sessions"].append(record)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, time.strftime("speed-%Y%m%d-%H%M%S.json"))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)
    warm_all = [s["arms"]["B"]["prefill_median_s"] for s in report["sessions"]]
    cold_all = [s["arms"]["C"]["prefill_median_s"] for s in report["sessions"]]
    print(f"\nprefill median, session chaude: {round(statistics.median(warm_all), 2)}s "
          f"| session neuve: {round(statistics.median(cold_all), 2)}s")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
