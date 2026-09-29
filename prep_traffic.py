#!/usr/bin/env python3
"""Preprocess the AgentX corpus (semianalysisai/cc-traces-weka-062126, full 1M variant) into the compact
binary the simulator reads (sim_core.js: loadTraffic).

The replay DAG follows AIPerf's WekaTraceLoader + agentic-replay scheduler (ai-dynamo/aiperf @ c78644090ff,
2026-09-28; files dataset/loader/weka_trace.py, dataset/loader/weka_agent_chains.py,
timing/trajectory_source.py, timing/branch_orchestrator.py, timing/replay_dependencies.py).

Per trace we build:
  * streams. Stream 0 is the root agent chain. Every other stream is a child conversation:
      - flattened agent chains that hash-id LCP detection splits off the top-level requests (st_role 3),
      - each subagent's own chain (st_role 1) and the sibling chains detected inside it (st_role 2).
    All children have st_kind 1, so older readers still treat them as spawned sub-streams.
  * the AIPerf replay DAG:
      - end-to-start delay per request within its stream;
      - spawn turn / join turn per branch, where a branch groups every subagent (or flat chain) that shares the same
        (spawn, join) pair;
      - an overlap flag per branch (dispatch at parent issue instead of parent return) and a dispatch offset per
        child.
  * cross-stream replay barriers (AIPerf infer_cross_stream_predecessors, per replay scope = the root with its flat
    chains, or one subagent with its sibling chains): for every request, on every other stream of its scope, the
    latest request that had completed at its recorded start (pruned to the frontier). All AIPerf edges are kept.
    CSR arrays: pred_head (nR+1) and pred_list, as global request indices (optional: sim_core.js runs without them).
  * tr_dur = last recorded request start - first (AIPerf TrajectorySource._trace_time_bounds), the t* range.
Validated against AIPerf's own loader code on all 393 traces (streams, per-request stream assignment, delays,
spawn/join/overlap/offset, replay scopes and predecessor sets are identical).
  * the prefix tree of 64-token blocks, compressed into "pieces" (see README.md, "KV residency"):
    hash_ids are prefix-chained and topologically increasing (child id > parent id), so a block id IS a trie
    node.  A piece is a maximal chain of nodes cut at branch points and at request ends, so every request
    touches whole pieces only -> exact LRU over pieces == exact LRU over 64-token pages.
  * lcp_prev (blocks shared with the previous request of the same stream) -> static-slot hit
  * lcp_best (blocks already seen anywhere in the trace, recorded order) -> infinite-cache hit

Output: <out>/traffic.bin (little-endian arrays, layout in <out>/traffic.json) + traffic.json (header + stats).
Usage (run on a cpu_only Slurm node, not the login node):
  python3 prep_traffic.py --traces /data/philei/agentx_data/062126/traces.jsonl --out /data/philei/m3_traffic_sim/data
"""
import argparse
import bisect
import heapq
import json
import math
import os
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

EPS = 1e-6  # AIPerf _JOIN_EPSILON_SECONDS / weka_agent_chains._EPSILON_SECONDS
SEAM_MAX_GAP_S = 3600.0  # Environment.DATASET.WEKA_SEAM_MAX_GAP_SECONDS default
SEAM_MIN_OVERLAP = 0.5  # Environment.DATASET.WEKA_SEAM_MIN_OVERLAP_RATIO default
TITLE_GEN_MAX_OUT = 64  # weka_trace._TITLE_GEN_MAX_OUTPUT_TOKENS
ROLE_ROOT, ROLE_SUB, ROLE_SUB_SIBLING, ROLE_FLAT = 0, 1, 2, 3


def lcp(a, b):
    m = min(len(a), len(b))
    if m == 0:
        return 0
    ne = np.nonzero(a[:m] != b[:m])[0]
    return int(ne[0]) if len(ne) else m


def _api(r):
    a = r.get("api_time")
    return a if a is not None and math.isfinite(a) else 0.0


def _end(r):
    return r["t"] + max(_api(r), 0.0)


# ------------------------------------------------------------------------------------------------------
# Agent-chain detection: a self-contained port of AIPerf weka_agent_chains.detect_agent_chains (phase 1 greedy
# extension/fork + phase 2 join-seam election with re-keying) and weka_trace._split_off_preamble, at
# ai-dynamo/aiperf c78644090ff with the default seam guard.  Only the partition into chains is needed here;
# the aux/worker-group classification in AIPerf only names the child sessions.
# A request is (outer_idx, dict) where dict has t, api_time, model, out, hash_ids and _h (np.int64 hash array).
# ------------------------------------------------------------------------------------------------------
class _Chain:
    __slots__ = ("requests", "fork", "spliced_into", "tail_outer", "tail_hash", "tail_end", "tail_model")

    def __init__(self, fork=None):
        self.requests = []
        self.fork = fork  # [parent_chain, fork_outer_idx, depth, fork_time] or None
        self.spliced_into = None
        self.tail_outer = -1
        self.tail_hash = np.empty(0, np.int64)
        self.tail_end = 0.0
        self.tail_model = ""


def _np_lcp(a, b):
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0
    neq = a[:n] != b[:n]
    i = int(neq.argmax())
    return i if neq[i] else n


def detect_agent_chains(normals):
    """-> (chains, main_index, worker_indices); chains[i].requests are (outer_idx, req) in (t, outer) order."""
    if not normals:
        return [], 0, []
    ordered = sorted(normals, key=lambda it: (it[1]["t"], it[0]))
    chains, chain_of, forks_by_tail, req_by = [], {}, {}, {}

    def append(ci, oi, r):
        c = chains[ci]
        c.requests.append((oi, r))
        chain_of[oi] = ci
        if r["hash_ids"]:
            c.tail_outer, c.tail_hash, c.tail_end, c.tail_model = oi, r["_h"], _end(r), r.get("model")

    for oi, r in ordered:  # phase 1
        req_by[oi] = r
        if not r["hash_ids"]:  # no LCP evidence: ride the main chain, invisible to tails
            if not chains:
                chains.append(_Chain())
            chains[0].requests.append((oi, r))
            chain_of[oi] = 0
            continue
        h = r["_h"]
        if not chains:
            chains.append(_Chain())
            append(0, oi, r)
            continue
        # extension target: deepest same-model tail that is a full prefix of h and ended by t
        best, best_len, hn = None, -1, h.shape[0]
        for idx, c in enumerate(chains):
            tl = c.tail_hash.shape[0]
            if tl == 0 or tl > hn or tl <= best_len or c.tail_model != r.get("model") or c.tail_end > r["t"] + EPS:
                continue
            if c.tail_hash[tl - 1] != h[tl - 1]:
                continue
            if bool((h[:tl] == c.tail_hash).all()):
                best, best_len = idx, tl
        if best is not None:
            append(best, oi, r)
            continue
        # fork: deepest-LCP tail (ties: deeper tail, lower index)
        parent, key = None, (0, 0)
        for idx, c in enumerate(chains):
            if c.tail_hash.shape[0] == 0:
                continue
            d = _np_lcp(c.tail_hash, h)
            if d and (d, c.tail_hash.shape[0]) > key:
                parent, key = idx, (d, c.tail_hash.shape[0])
        depth = key[0]
        if parent is None and all(c.tail_hash.shape[0] == 0 for c in chains):
            append(0, oi, r)
            continue
        fork = [parent, chains[parent].tail_outer if parent is not None else None, depth, r["t"]]
        chains.append(_Chain(fork))
        append(len(chains) - 1, oi, r)
        if fork[1] is not None and depth > 0:
            forks_by_tail.setdefault(fork[1], []).append(len(chains) - 1)

    # phase 2: splice join-seam continuations onto dead tails
    alias = {}

    def resolve(i):
        while i in alias:
            i = alias[i]
        return i

    def last_hash_outer(c):
        for oi, q in reversed(c.requests):
            if q["hash_ids"]:
                return oi
        return None

    keys = sorted(forks_by_tail)
    heapq.heapify(keys)
    processed = set()
    while keys:
        fo = heapq.heappop(keys)
        if fo in processed:
            continue
        processed.add(fo)
        owner = resolve(chain_of[fo])
        oc = chains[owner]
        if last_hash_outer(oc) != fo:
            continue
        registered = [ci for ci in forks_by_tail[fo] if chains[ci].spliced_into is None]
        tr = req_by[fo]
        t_end, tail_blocks = _end(tr), len(tr["hash_ids"])

        def blocked(ci):
            if tail_blocks == 0:
                return False
            gap = chains[ci].requests[0][1]["t"] - t_end
            return gap > SEAM_MAX_GAP_S and chains[ci].fork[2] / tail_blocks < SEAM_MIN_OVERLAP

        cands = [
            ci
            for ci in registered
            if chains[ci].fork is not None
            and chains[ci].fork[2] > 0
            and t_end <= chains[ci].requests[0][1]["t"] + EPS
            and chains[ci].requests[0][1].get("model") == tr.get("model")
            and not blocked(ci)
        ]
        if not cands:
            continue
        elected = max(cands, key=lambda ci: (chains[ci].fork[2], -chains[ci].fork[3], -ci))
        tc = chains[elected]
        oc.requests.extend(tc.requests)
        for oi, _ in tc.requests:
            chain_of[oi] = owner
        tc.spliced_into = owner
        alias[elected] = owner
        new_tail = last_hash_outer(oc)
        if new_tail is None:
            continue
        nth = req_by[new_tail]["_h"]
        rekeyed = False
        for ci in registered:  # re-key the non-elected forks against the merged chain's new tail
            c = chains[ci]
            if ci == elected or c.spliced_into is not None or c.fork is None:
                continue
            d = _np_lcp(nth, c.requests[0][1]["_h"])
            if d <= 0:
                continue
            c.fork[1], c.fork[2], c.fork[0] = new_tail, d, owner
            forks_by_tail.setdefault(new_tail, []).append(ci)
            rekeyed = True
        if rekeyed:
            processed.discard(new_tail)
            heapq.heappush(keys, new_tail)

    main = resolve(chain_of[ordered[0][0]])
    workers = [i for i, c in enumerate(chains) if c.spliced_into is None and i != main]
    workers.sort(key=lambda i: (chains[i].requests[0][1]["t"], chains[i].requests[0][0]))
    return chains, main, workers


def split_off_preamble(normals):
    """AIPerf weka_trace._split_off_preamble: peel one leading prefix-disjoint throwaway request."""
    if len(normals) < 2:
        return [], normals
    ordered = sorted(normals, key=lambda it: (it[1]["t"], it[0]))
    oi, r = ordered[0]
    if not r["hash_ids"]:
        return [], normals
    rest = ordered[1:]
    if any(lcp(r["_h"], o["_h"]) > 0 for _, o in rest if o["hash_ids"]):
        return [], normals
    if r["out"] > TITLE_GEN_MAX_OUT:
        other = set()
        for _, o in rest:
            other.update(o["hash_ids"])
        if not other.isdisjoint(r["hash_ids"]):
            return [], normals
    return [(oi, r)], sorted(rest, key=lambda it: it[0])


def split_chains(indexed):
    """-> list of chains (each a list of (idx, req)); chain 0 = main (with the preamble re-attached)."""
    pre, rest = split_off_preamble(indexed)
    chains, main, workers = detect_agent_chains(rest)
    if not chains:
        return [sorted(pre, key=lambda it: (it[1]["t"], it[0]))] if pre else []
    m = list(chains[main].requests)
    if pre:
        m = sorted(pre + m, key=lambda it: (it[1]["t"], it[0]))
    return [m] + [list(chains[w].requests) for w in workers]


# ------------------------------------------------------------------------------------------------------
# Cross-stream replay barriers: AIPerf timing/replay_dependencies.infer_cross_stream_predecessors
# ------------------------------------------------------------------------------------------------------
def cross_stream_predecessors(stream_reqs):
    """stream_reqs: {stream: [(local_req_idx, start, end), ...]} of one replay scope.
    -> {local_req_idx: [pred local_req_idx, ...]} (latest completed request of every other stream, pruned)."""
    by_end = {}
    for s, L in stream_reqs.items():
        o = sorted(L, key=lambda x: (x[2], x[1], x[0]))  # max() key of AIPerf: (end, start, key)
        by_end[s] = (o, [x[2] for x in o])
    out = {}
    for s, L in stream_reqs.items():
        for ri, ts, _ in L:
            front = []
            for s2, (o, ends) in by_end.items():
                if s2 == s:
                    continue
                k = bisect.bisect_right(ends, ts) - 1
                while k >= 0 and not (o[k][1] < ts):  # completed = start < ts and end <= ts
                    k -= 1
                if k >= 0:
                    front.append(o[k])
            preds = [c[0] for c in front if not any(c[1] < l[1] and c[2] <= l[1] for l in front if l is not c)]
            if preds:
                out[ri] = sorted(preds)
    return out


def process(args):
    ti, line = args
    d = json.loads(line)
    top, subs = [], []
    for oi, r in enumerate(d["requests"]):
        if r["type"] == "subagent":
            subs.append((oi, r))
        elif r.get("hash_ids"):  # AIPerf keeps hash-less requests on the main chain; this corpus has none
            r["_h"] = np.asarray(r["hash_ids"], dtype=np.int64)
            r["_src"] = (oi, None)
            top.append((oi, r))
    if not top:
        return None
    # ---- top-level flat-chain split (weka_trace._detect_and_split_flat_chains; only when > 1 request)
    if len(top) > 1:
        ch = split_chains(top)
        main, flats = ch[0], ch[1:]
        if not flats:
            main = top
    else:
        main, flats = top, []
    main_pos = sorted(((oi, k) for k, (oi, _) in enumerate(main)))  # (outer idx, main-chain position)
    main_t = [r["t"] for _, r in main]

    def preceding(oi, default):
        p = [k for o, k in main_pos if o < oi]
        return max(p) if p else default

    def join_after(oi, end):
        for o, k in main_pos:
            if o > oi and main_t[k] + EPS >= end:
                return k
        return -1

    # ---- branches: (spawn, join) -> children; subagent groups first, then flat groups (AIPerf order)
    branches = []  # dict(spawn, join, start, children=[(role, reqs, scope)])
    sub_groups, sub_order = {}, []
    nsub = 0
    for oi, g in subs:
        gt = float(g["t"])
        inner = []
        for qi, q in enumerate(g["requests"]):
            if not q.get("hash_ids"):
                continue
            q["_src"] = (oi, qi)
            assert q["t"] >= gt - EPS, "relative inner timestamps are not supported"
            q["t"] = max(q["t"], gt)  # absolute basis: canonical_t = max(inner.t, marker.t)
            q["_h"] = np.asarray(q["hash_ids"], dtype=np.int64)
            inner.append(q)
        sp = preceding(oi, None)
        if sp is None:
            continue  # AIPerf drops subagents with no preceding parent turn
        if g.get("duration_ms") is not None:
            dur = g["duration_ms"] / 1000.0
            g_end = gt + (dur if math.isfinite(dur) and dur >= 0 else 0.0)
        elif inner:
            g_end = max(_end(q) for q in inner)
        else:
            g_end = gt
        jn = join_after(oi, g_end)
        chains = split_chains(list(enumerate(inner))) if inner else []
        scope = 1 + nsub
        nsub += 1
        kids = [(ROLE_SUB if c == 0 else ROLE_SUB_SIBLING, [q for _, q in ch], scope) for c, ch in enumerate(chains)]
        key = (sp, jn)
        if key not in sub_groups:
            sub_groups[key] = dict(spawn=sp, join=jn, start=gt, children=[])
            sub_order.append(key)
        sub_groups[key]["start"] = min(sub_groups[key]["start"], gt)
        sub_groups[key]["children"].extend(kids)
    branches += [sub_groups[k] for k in sub_order]
    flat_groups, flat_order = {}, []
    for ch in flats:
        fo = ch[0][0]
        sp = preceding(fo, 0)
        jn = join_after(fo, max(_end(r) for _, r in ch))
        key = (sp, jn)
        if key not in flat_groups:
            flat_groups[key] = dict(spawn=sp, join=jn, start=ch[0][1]["t"], children=[])
            flat_order.append(key)
        flat_groups[key]["start"] = min(flat_groups[key]["start"], ch[0][1]["t"])
        flat_groups[key]["children"].append((ROLE_FLAT, [r for _, r in ch], 0))
    branches += [flat_groups[k] for k in flat_order]
    # ---- streams
    streams = [dict(role=ROLE_ROOT, reqs=[r for _, r in main], spawn=-1, join=-1, off=0.0, ovl=0, t0=0.0, scope=0)]
    for b in branches:
        pr = main[b["spawn"]][1]
        papi = pr.get("api_time")
        papi = max(0.0, papi) if papi is not None and math.isfinite(papi) else None
        ovl = papi is not None and papi > 0 and b["start"] < pr["t"] + papi  # branch_orchestrator.on_credit_issued
        origin = pr["t"] if ovl else b["start"]
        for role, reqs, scope in b["children"]:
            if not reqs:
                continue
            streams.append(
                dict(
                    role=role,
                    reqs=reqs,
                    spawn=b["spawn"],
                    join=b["join"],
                    off=max(0.0, reqs[0]["t"] - origin),
                    ovl=int(ovl),
                    t0=b["start"],
                    scope=scope,
                )
            )
    reqs = [(si, k, q) for si, s in enumerate(streams) for k, q in enumerate(s["reqs"])]
    n = len(reqs)
    first = np.cumsum([0] + [len(s["reqs"]) for s in streams])
    # ---- end-to-start delay within each stream
    delay = np.zeros(n)
    for i in range(1, n):
        if reqs[i][0] == reqs[i - 1][0]:
            delay[i] = max(0.0, reqs[i][2]["t"] - reqs[i - 1][2]["t"] - _api(reqs[i - 1][2]))
    # ---- prefix tree
    H = [q["_h"] for _, _, q in reqs]
    maxid = max(int(h.max()) for h in H)
    parent = np.full(maxid + 1, -2, dtype=np.int64)
    is_end = np.zeros(maxid + 1, dtype=bool)
    for h in H:
        parent[h[0]] = -1
        parent[h[1:]] = h[:-1]
        is_end[h[-1]] = True
    present = parent != -2
    nodes = np.nonzero(present)[0]  # increasing id == topological order
    par = parent[nodes]
    child_cnt = np.bincount(par[par >= 0], minlength=maxid + 1)
    start = (par < 0) | (child_cnt[np.maximum(par, 0)] > 1) | is_end[np.maximum(par, 0)]
    start[par < 0] = True
    piece_of = np.full(maxid + 1, -1, dtype=np.int64)
    np_ = 0
    piece_parent = []
    piece_len = []
    so, pl, nl = start.tolist(), par.tolist(), nodes.tolist()
    pof = piece_of
    for x, s, p in zip(nl, so, pl):
        if s:
            pof[x] = np_
            piece_parent.append(int(pof[p]) if p >= 0 else -1)
            piece_len.append(1)
            np_ += 1
        else:
            q = pof[p]
            pof[x] = q
            piece_len[q] += 1
    # ---- per-request arrays
    leaf = np.array([piece_of[h[-1]] for h in H], dtype=np.int64)
    blocks = np.array([len(h) for h in H], dtype=np.int64)
    lcp_prev = np.zeros(n, dtype=np.int64)
    for i in range(1, n):
        if reqs[i][0] == reqs[i - 1][0]:
            lcp_prev[i] = lcp(H[i], H[i - 1])
    # infinite-cache hit in recorded (start-time) order; "seen" is prefix-closed along a path
    order = sorted(range(n), key=lambda i: (reqs[i][2]["t"], reqs[i][0], reqs[i][1]))
    seen = np.zeros(maxid + 1, dtype=bool)
    lcp_best = np.zeros(n, dtype=np.int64)
    for i in order:
        h = H[i]
        lo, hi = 0, len(h)
        while lo < hi:
            mid = (lo + hi) // 2
            if seen[h[mid]]:
                lo = mid + 1
            else:
                hi = mid
        lcp_best[i] = lo
        seen[h] = True
    pp = np.array(piece_parent, dtype=np.int64)
    pdepth = np.zeros(np_, dtype=np.int64)
    for q in range(np_):
        pdepth[q] = 0 if pp[q] < 0 else pdepth[pp[q]] + 1
    t0 = min(q["t"] for _, _, q in reqs)
    t1 = max(q["t"] for _, _, q in reqs)  # AIPerf _trace_time_bounds: min/max request start
    # ---- cross-stream barriers per replay scope (weka_trace._install_replay_dependencies). Every AIPerf edge is kept,
    # including the ones the spawn/join DAG already implies (counted for the stats only), so a snapshot start at t*
    # gates exactly like AIPerf's ReplayBarrierCoordinator seeded with the completed prefixes.
    by_scope = defaultdict(dict)
    for i, (si, k, q) in enumerate(reqs):
        by_scope[streams[si]["scope"]].setdefault(si, []).append((i, q["t"], _end(q)))
    preds = defaultdict(list)
    n_implied = 0
    for sc, sr in by_scope.items():
        if len(sr) < 2:
            continue
        for ri, P in cross_stream_predecessors(sr).items():
            c = reqs[ri][0]
            for p in P:
                d_ = reqs[p][0]
                if c == 0:  # root turn at/after the join of the predecessor's stream
                    implied = streams[d_]["join"] >= 0 and reqs[ri][1] >= streams[d_]["join"]
                elif d_ == 0:  # child after its spawn turn (ended, or issued on the overlap path)
                    ppos, sp = reqs[p][1], streams[c]["spawn"]
                    implied = ppos < sp or (ppos == sp and not streams[c]["ovl"])
                else:  # predecessor's stream joined at/before this child's spawn turn
                    implied = streams[d_]["join"] >= 0 and streams[d_]["join"] <= streams[c]["spawn"]
                n_implied += implied
                preds[ri].append(p)
    n_all = sum(len(P) for P in preds.values())
    # the DAG + barriers must be acyclic (else a replay would deadlock)
    succ = defaultdict(list)
    indeg = np.zeros(n, dtype=np.int64)

    def edge(a, b):
        succ[a].append(b)
        indeg[b] += 1

    for si, s in enumerate(streams):
        for k in range(1, len(s["reqs"])):
            edge(first[si] + k - 1, first[si] + k)
        if si:
            edge(s["spawn"], first[si])
            if s["join"] >= 0:
                edge(first[si] + len(s["reqs"]) - 1, s["join"])
    for ri, P in preds.items():
        for p in P:
            edge(p, ri)
    stack = [i for i in range(n) if indeg[i] == 0]
    seen_n = 0
    while stack:
        a = stack.pop()
        seen_n += 1
        for b in succ[a]:
            indeg[b] -= 1
            if indeg[b] == 0:
                stack.append(b)
    assert seen_n == n, f"trace {ti}: replay DAG with barriers has a cycle"
    pred_cnt = np.array([len(preds.get(i, ())) for i in range(n)], dtype=np.int64)
    pred_list = np.array([p for i in range(n) for p in preds.get(i, ())], dtype=np.int64)
    ns = len(streams)
    assert ns < 4096, "sim_core slot keys assume < 4096 streams per trace"
    return dict(
        ti=ti,
        req=dict(
            stream=np.array([si for si, _, _ in reqs]),
            idx=np.array([k for _, k, _ in reqs]),
            t=np.array([q["t"] - t0 for _, _, q in reqs]),
            api=np.array([_api(q) for _, _, q in reqs]),
            delay=delay,
            blocks=blocks,
            out=np.array([int(q["out"]) for _, _, q in reqs]),
            leaf=leaf,
            lcp_prev=lcp_prev,
            lcp_best=lcp_best,
            pathlen=pdepth[leaf] + 1,
            pred_cnt=pred_cnt,
        ),
        pred_list=pred_list,
        stream=dict(
            kind=np.array([0] + [1] * (ns - 1)),
            role=np.array([s["role"] for s in streams]),
            spawn=np.array([s["spawn"] for s in streams]),
            join=np.array([s["join"] for s in streams]),
            off=np.array([s["off"] for s in streams]),
            ovl=np.array([s["ovl"] for s in streams]),
            t0=np.array([s["t0"] - t0 if si else 0.0 for si, s in enumerate(streams)]),
            t1=np.array([max(_end(q) for q in s["reqs"]) - t0 if si else 0.0 for si, s in enumerate(streams)]),
            first=first[:-1],
            cnt=np.array([len(s["reqs"]) for s in streams]),
            scope=np.array([s["scope"] for s in streams]),  # replay scope (validation only, not written)
        ),
        piece=dict(parent=pp, len=np.array(piece_len, dtype=np.int64)),
        dur=t1 - t0,
        bar=(n_all, n_implied),
        src=[q["_src"] for _, _, q in reqs],  # (outer idx, inner idx or None): validation only, not written
    )


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument(
        "--traces",
        default=os.environ.get("AGENTX_TRACES", "/data/philei/agentx_data/062126/traces.jsonl"),
        help="traces.jsonl of semianalysisai/cc-traces-weka-062126 (HF dataset); env AGENTX_TRACES",
    )
    ap.add_argument("--out", default=os.environ.get("M3SIM_DATA", os.path.join(here, "data")))
    ap.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    lines = []
    with open(a.traces) as f:
        for i, line in enumerate(f):
            if a.limit and i >= a.limit:
                break
            if line.strip():
                lines.append((i, line))
    print(f"read {len(lines)} traces in {time.time() - t0:.0f}s", flush=True)
    with Pool(a.procs) as p:
        res = [r for r in p.imap(process, lines, chunksize=1) if r is not None]
    del lines
    print(f"processed in {time.time() - t0:.0f}s", flush=True)
    # ---- concatenate with global offsets
    R, S, P, PL = [], [], [], []
    tr_req0, tr_st0, tr_pc0, tr_dur = [], [], [], []
    nr = ns = npc = 0
    bar_all = bar_implied = 0
    for r in res:
        tr_req0.append(nr)
        tr_st0.append(ns)
        tr_pc0.append(npc)
        tr_dur.append(r["dur"])
        bar_all += r["bar"][0]
        bar_implied += r["bar"][1]
        q = dict(r["req"])
        q["leaf"] = q["leaf"] + npc
        s = dict(r["stream"])
        s["spawn"] = np.where(s["spawn"] >= 0, s["spawn"] + nr, -1)  # spawn/join are main-stream request idx -> global
        s["join"] = np.where(s["join"] >= 0, s["join"] + nr, -1)
        s["first"] = s["first"] + nr
        q["trace"] = np.full(len(q["blocks"]), len(tr_req0) - 1)
        q["stream"] = q["stream"] + ns
        pc = dict(r["piece"])
        pc["parent"] = np.where(pc["parent"] >= 0, pc["parent"] + npc, -1)
        R.append(q)
        S.append(s)
        P.append(pc)
        PL.append(r["pred_list"] + nr)
        nr += len(q["blocks"])
        ns += len(s["kind"])
        npc += len(pc["len"])
    cat = lambda L, k: np.concatenate([x[k] for x in L])
    pred_cnt = cat(R, "pred_cnt")
    pred_head = np.concatenate([[0], np.cumsum(pred_cnt)])
    pred_list = np.concatenate(PL) if PL else np.zeros(0, dtype=np.int64)
    arrays = [
        # name, dtype, data
        ("req_trace", "u2", cat(R, "trace")),
        ("req_stream", "u4", cat(R, "stream")),
        ("req_idx", "u2", cat(R, "idx")),
        ("req_t", "f4", cat(R, "t")),
        ("req_api", "f4", cat(R, "api")),
        ("req_delay", "f4", cat(R, "delay")),
        ("req_blocks", "u2", cat(R, "blocks")),
        ("req_out", "u4", cat(R, "out")),
        ("req_leaf", "u4", cat(R, "leaf")),
        ("req_lcp_prev", "u2", cat(R, "lcp_prev")),
        ("req_lcp_best", "u2", cat(R, "lcp_best")),
        ("st_kind", "u1", cat(S, "kind")),
        ("st_role", "u1", cat(S, "role")),
        ("st_spawn", "i4", cat(S, "spawn")),
        ("st_join", "i4", cat(S, "join")),
        ("st_off", "f4", cat(S, "off")),
        ("st_ovl", "u1", cat(S, "ovl")),
        ("st_t0", "f4", cat(S, "t0")),
        ("st_t1", "f4", cat(S, "t1")),
        ("st_first", "u4", cat(S, "first")),
        ("st_cnt", "u2", cat(S, "cnt")),
        ("pc_parent", "i4", cat(P, "parent")),
        ("pc_len", "u2", cat(P, "len")),
        ("tr_req0", "u4", np.array(tr_req0)),
        ("tr_st0", "u4", np.array(tr_st0)),
        ("tr_pc0", "u4", np.array(tr_pc0)),
        ("tr_dur", "f4", np.array(tr_dur)),
        ("pred_head", "u4", pred_head),  # optional (sim_core.js runs without barriers when absent)
        ("pred_list", "u4", pred_list),
    ]
    for name, dt, x in arrays:
        if dt[0] in "ui" and len(x):
            info = np.iinfo(np.dtype("<" + dt))
            assert x.min() >= info.min and x.max() <= info.max, (name, x.min(), x.max())
    os.makedirs(a.out, exist_ok=True)
    layout = []
    off = 0
    with open(os.path.join(a.out, "traffic.bin"), "wb") as f:
        for name, dt, x in arrays:
            b = np.ascontiguousarray(x.astype("<" + dt)).tobytes()
            pad = (-off) % 8
            f.write(b"\0" * pad)
            off += pad
            layout.append(dict(name=name, dtype=dt, offset=off, n=int(len(x))))
            f.write(b)
            off += len(b)
    blocks = cat(R, "blocks")
    lb = cat(R, "lcp_best")
    lp = cat(R, "lcp_prev")
    pl = cat(R, "pathlen")
    role = cat(S, "role")
    stats = dict(
        traces=len(res),
        requests=int(nr),
        streams=int(ns),
        streams_by_role=dict(
            root=int((role == 0).sum()),
            subagent=int((role == 1).sum()),
            subagent_sibling=int((role == 2).sum()),
            flat_chain=int((role == 3).sum()),
        ),
        pieces=int(npc),
        input_tokens=int(blocks.sum() * 64),
        new_tokens_inf=int((blocks - lb).sum() * 64),
        inf_hit_rate=float(lb.sum() / blocks.sum()),
        prev_stream_hit_rate=float(lp.sum() / blocks.sum()),
        pathlen_mean=float(pl.mean()),
        pathlen_p99=float(np.percentile(pl, 99)),
        pathlen_max=int(pl.max()),
        pathlen_sum=int(pl.sum()),
        barrier_edges=int(len(pred_list)),
        barrier_edges_dag_implied=int(bar_implied),
        barrier_gated_requests=int((pred_cnt > 0).sum()),
        bytes=off,
    )
    json.dump(
        dict(layout=layout, stats=stats, source=a.traces, block=64),
        open(os.path.join(a.out, "traffic.json"), "w"),
        indent=1,
    )
    print(json.dumps(stats, indent=1), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
