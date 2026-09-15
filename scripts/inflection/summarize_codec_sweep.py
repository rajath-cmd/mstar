"""Summarise a codec-cadence sweep: TTFA vs throughput vs CORRECTNESS."""

from __future__ import annotations

import json
import pathlib
import sys


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    rows = []
    for d in sorted(root.glob("arm_*")):
        if d.is_file():
            continue
        chunk, ctx = d.name.removeprefix("arm_").split("_")
        res = d / "results.json"
        chk = d / "audio_check.json"
        if not res.is_file():
            rows.append((int(chunk), int(ctx), None, None, None, None, "server failed"))
            continue
        r = json.load(open(res))
        lv = r["levels"]

        def val(conc, *path, _lv=lv):
            node = _lv.get(str(conc), {}).get("mstar", {})
            for k in path:
                node = node.get(k, {}) if isinstance(node, dict) else None
                if node is None:
                    return None
            return node if isinstance(node, (int, float)) else None

        complete = None
        dur = None
        if chk.is_file():
            c = json.load(open(chk))
            complete = c.get("opening_words_present")
            dur = c.get("duration_s")
        rows.append((int(chunk), int(ctx), val(1, "ttfa_ms", "p50"), val(32, "ttfa_ms", "p50"),
                     val(32, "xrt_aggregate"), dur,
                     "OK" if complete else ("TRUNCATED" if complete is False else "unchecked")))

    print(f"\n{'chunk':>6} {'ctx':>4} {'TTFA c=1':>9} {'TTFA c=32':>10} {'xRT c=32':>9} {'audio_s':>8}  verdict")
    for chunk, ctx, t1, t32, x32, dur, verdict in sorted(rows):
        f = lambda v, s="{:.0f}": s.format(v) if isinstance(v, (int, float)) else "-"  # noqa: E731
        print(f"{chunk:>6} {ctx:>4} {f(t1):>9} {f(t32):>10} {f(x32,'{:.1f}'):>9} {f(dur,'{:.2f}'):>8}  {verdict}")
    print("\nA TRUNCATED arm is disqualified regardless of its TTFA: it drops the")
    print("start of the stream and still sounds fluent, which is the worst failure mode.")


if __name__ == "__main__":
    main()
