# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Upstream pure trace conversion; collection/build entry points excluded."""
from collections import defaultdict

def _to_perfetto(d):  # noqa: PLR0912
    """In-place transform of an Insight trace into a Perfetto-friendly one.

    Insight encodes each pipe as one track and packs concurrent, pipelined
    instructions onto it; Perfetto drops overlapping `ph:X` complete events and
    can mis-pair `B`/`E` flag events. This fixes both, losslessly (same events,
    same ts/dur; only `tid` changes and `B`/`E` is re-encoded as `X`):

      * sub-lanes: greedily pack each duration event into the lowest lane whose
        previous event ended, so no two events on a lane overlap. A split pipe
        `MTE1` becomes `MTE1#0..#k`.
      * atomic flags: merge each `B`/`E` pair into one `ph:X` slice.
      * thread_sort_index rebuilt as base*100+lane so lanes render adjacent.
    """
    EPS = 1e-9
    evs = d["traceEvents"]
    orig_sort = {(e["pid"], e["tid"]): e["args"]["sort_index"] for e in evs if e.get("name") == "thread_sort_index"}

    def is_core(e):
        return str(e.get("pid", "")).startswith("core")

    # 1. Duration intervals per (pid,tid): X events + B/E pairs (matched by id).
    intervals = defaultdict(list)
    be = defaultdict(dict)
    for e in evs:
        if not is_core(e):
            continue
        ph = e.get("ph")
        key = (e["pid"], e["tid"])
        if ph == "X":
            intervals[key].append(
                {
                    "ts": e["ts"],
                    "end": e["ts"] + e.get("dur", 0.0),
                    "name": e["name"],
                    "args": e.get("args", {}),
                }
            )
        elif ph in ("B", "E"):
            slot = be[(e["pid"], e["tid"], e.get("id"))]
            slot[ph] = e["ts"]
            slot.setdefault("src", e)
    for (pid, tid, eid), slot in be.items():
        s = slot.get("B", slot.get("E"))
        en = slot.get("E", slot.get("B"))
        if s is not None and en is not None and s > en:
            s, en = en, s
        src = slot["src"]
        intervals[(pid, tid)].append(
            {
                "ts": s,
                "end": en,
                "name": src["name"],
                "args": src.get("args", {}),
                "id": eid,
            }
        )

    # 2. Greedy lane assignment (interval partitioning) per track.
    max_lane = defaultdict(int)
    be_lane = {}
    for key, iv in intervals.items():
        iv.sort(key=lambda t: (t["ts"], -(t["end"] - t["ts"])))
        lane_end = []
        for it in iv:
            placed = None
            for ln in range(len(lane_end)):
                if lane_end[ln] <= it["ts"] + EPS:
                    placed = ln
                    break
            if placed is None:
                placed = len(lane_end)
                lane_end.append(it["end"])
            else:
                lane_end[placed] = it["end"]
            it["lane"] = placed
            max_lane[key] = max(max_lane[key], placed)
            if "id" in it:
                be_lane[(key[0], key[1], it["id"])] = placed

    def split(pid, tid):
        return max_lane.get((pid, tid), 0) > 0

    def laned(pid, tid, lane):
        return f"{tid}#{lane}" if split(pid, tid) else tid

    out = []
    # 3. Emit every duration interval as an atomic X with its sub-lane tid.
    for (pid, tid), iv in intervals.items():
        for it in iv:
            out.append(
                {
                    "ph": "X",
                    "name": it["name"],
                    "ts": it["ts"],
                    "dur": it["end"] - it["ts"],
                    "pid": pid,
                    "tid": laned(pid, tid, it["lane"]),
                    "args": it["args"],
                }
            )

    # 4. Pass through the rest; anchor flow arrows to their id-pair's lane.
    for e in evs:
        ph = e.get("ph")
        if e.get("name") == "thread_sort_index" or ph in ("X", "B", "E"):
            continue
        ne = dict(e)
        if is_core(e) and ph in ("s", "t", "f") and split(e["pid"], e["tid"]):
            ln = be_lane.get((e["pid"], e["tid"], e.get("id")), 0)
            ne["tid"] = f"{e['tid']}#{ln}"
        out.append(ne)

    # 5. Rebuild thread_sort_index: integer, contiguous, lanes adjacent.
    for (pid, tid), base in orig_sort.items():
        mx = max_lane.get((pid, tid), 0)
        if mx > 0:
            for k in range(mx + 1):
                out.append(
                    {
                        "args": {"sort_index": base * 100 + k},
                        "name": "thread_sort_index",
                        "ph": "M",
                        "pid": pid,
                        "tid": f"{tid}#{k}",
                    }
                )
        else:
            out.append(
                {"args": {"sort_index": base * 100}, "name": "thread_sort_index", "ph": "M", "pid": pid, "tid": tid}
            )

    d["traceEvents"] = out
    return d

