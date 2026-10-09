/* sim_core.js - MiniMax-M3 pipeline-prefill traffic simulator (AgentX replay x per-op roofline).
 *
 * One file, no dependencies. Used by the artifact (inlined, runs in a Web Worker) and by node (sweep.js,
 * validate.js). See README.md for the model, its calibration and the knobs.
 *
 * Units: cost model in milliseconds; simulation clock in seconds; KV in tokens (64-token blocks in the trace).
 */
(function (root) {
  'use strict';

  // ------------------------------------------------------------------------------------------------------
  // Model + hardware constants
  // ------------------------------------------------------------------------------------------------------
  const M3 = {
    E: 6144, Hq: 64, Hkv: 4, d: 128, Hi: 4, di: 128, sparseKeys: 2048, Ex: 128, topk: 4, I: 3072, Is: 3072,
    Id: 12288, V: 200064, L: 60, nDense: 3, maxCtx: 1048576,
  };
  // Blackhole chip (tt-moe-nappkin lib/system.py): 110 cores x 4.096 kFLOP/cycle x 1.35 GHz (LoFi), HiFi2 = 1/2;
  // 512 GB/s GDDR6, 32 GiB; ethernet 2 links x 200 Gb/s x 2 dirs per axis neighbour (100 GB/s bidir CCL, 50 GB/s uni).
  const HW = { F_lofi: 608e12, F_hifi: 304e12, dram: 512e9, dramCap: 32 * 2 ** 30, linkBi: 100e9, linkUni: 50e9, pcie: 64e9 };
  const BF8 = 1.0625, BF4 = 0.5625, BF16 = 2;
  // Measured M3 decode speed per user (tokens/s/u) vs context, 64 users ("Minimax M3 Updates" deck, decode
  // performance slide). Time per token is linear in context: a fixed ring trip plus attention over the KV, fitted by
  // least squares (8.48 ms + 0.0175 ms per 1k tokens, within 0.6% of every point). decodeCurve 'm3' scales it so
  // that decodeTps is the speed at 100k context (the deck's targets are quoted @100k).
  const M3_DECODE_TSU = [[8000, 116], [60000, 105], [100000, 98], [140000, 92], [310000, 72], [550000, 55]];
  const M3_DECODE_TRIP = (() => { // [seconds per token at 0 context, seconds per token per context token]
    const xs = M3_DECODE_TSU.map(([c]) => c), ys = M3_DECODE_TSU.map(([, t]) => 1 / t), n = xs.length;
    const mx = xs.reduce((a, b) => a + b) / n, my = ys.reduce((a, b) => a + b) / n;
    let sxy = 0, sxx = 0; for (let i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
    const b = sxy / sxx; return [my - b * mx, b];
  })();
  const DECODE_REF_CTX = 100000;
  // decode speed per user (tokens/s/u) for a session with `ctx` tokens of context while the ring has room
  function decodeSpeed(cfg, ctx) {
    if (cfg.decodeCurve !== 'm3') return cfg.decodeTps;
    const [a, b] = M3_DECODE_TRIP;
    return cfg.decodeTps * (a + b * DECODE_REF_CTX) / (a + b * ctx);
  }
  const MB = 1e6, GB = 1e9;
  // Decode KV per token per layer on one chip (tt-blaze M3 decode, bf8: K/V sharded by head over the 4 mesh rows and
  // replicated over the 2 columns, index_k split over the columns): 1088 / 4 + 136 / 2 = 340 B. Copying a context of
  // `tok` tokens pool -> lane over `layers` layers reads and writes it in DRAM at 50% (as prefill's pool copies).
  const DEC_KVB_CHIP = 2 * M3.Hkv * M3.d * BF8 / 4 + M3.di * BF8 / 2;
  const decCopyS = (tok, layers) => tok * layers * DEC_KVB_CHIP * 2 / (HW.dram * 0.5);

  const OPS_MOE = ['norm_ag', 'qkv', 'idx_branch', 'misc', 'o_proj', 'attn_rs', 'shared', 'router', 'dispatch', 'experts', 'combine', 'moe_reduce'];
  const OPS_MSA_SEG = ['ag_kv', 'ag_idx', 'indexer', 'sparse', 'kv_a2a'];
  const OPS_DENSE = ['norm_ag', 'qkv', 'misc', 'o_proj', 'attn_rs', 'dense_mlp'];
  const OPS_DENSE_SEG = ['ring', 'kv_a2a'];
  const CCL_OPS = new Set(['norm_ag', 'attn_rs', 'dispatch', 'combine', 'ag_kv', 'ag_idx', 'kv_a2a', 'shared', 'moe_reduce', 'dense_mlp']);
  // zone name (parse_zone_perf.py) -> op
  const ZONE_MOE = {
    qkv: 'attn/qkv_proj', idx_branch: 'attn/index_branch', ag_kv: 'attn/ag_kv', ag_idx: 'attn/ag_index_k', indexer: 'attn/indexer',
    sparse: 'attn/sparse_sdpa', o_proj: 'attn/o_proj', attn_rs: 'attn/ccl_out_reduce_scatter', shared: 'mlp/shared_expert',
    router: 'mlp/router_topk', dispatch: 'mlp/dispatch', experts: 'mlp/experts_mm', combine: 'mlp/combine', moe_reduce: 'mlp/moe_reduce',
  };
  const ZONE_DENSE = { qkv: 'attn/qkv_proj', ring: 'attn/ring_joint_sdpa', o_proj: 'attn/o_proj', attn_rs: 'attn/ccl_out_reduce_scatter', dense_mlp: 'mlp' };
  // target ("fixed kernels") efficiency per op and latency floors (ms)
  const TARGET_EFF = { matmul: 0.7, dram: 0.8, link: 0.8 };
  const OP_CLASS = {
    norm_ag: 'link', qkv: 'matmul', idx_branch: 'matmul', misc: 'dram', o_proj: 'matmul', attn_rs: 'link', shared: 'matmul',
    router: 'matmul', dispatch: 'link', experts: 'matmul', combine: 'link', moe_reduce: 'dram', ag_kv: 'link', ag_idx: 'link',
    indexer: 'matmul', sparse: 'matmul', kv_a2a: 'link', dense_mlp: 'matmul', ring: 'matmul', ring_scan: 'link',
  };
  const LAT_MEAS = { ccl: 0.04, op: 0.01 };
  const LAT_TGT = { ccl: 0.02, op: 0.005 };

  // ------------------------------------------------------------------------------------------------------
  // Per-op roofline (seconds at 100% efficiency).  ctx: {sp,tp,P,T,idxB}  seg: {n,k,cap}
  // ------------------------------------------------------------------------------------------------------
  function roofTok(op, c) {
    const { sp, tp, P, T } = c;
    const E = M3.E, tl = T / sp;
    // routed tokens: the padded chunk tail is trimmed from dispatch/experts/combine/moe_reduce (padding_config
    // actual_isl); the router itself still scores the whole padded chunk
    // c.trl: real routed tokens on the busiest SP rank, when ranks are uneven (owner placement)
    const Tr = c.Tr === undefined ? T : c.Tr, trl = c.trl === undefined ? Tr / sp : c.trl;
    const rsBytes = (tp - 1) / tp * tl * E * BF16;
    const tf = torusFactors(c);
    const a2a = (m) => (sp > 1 ? sp * m / (4 * HW.linkUni) * tf.a2a : m / HW.dram); // SP line (or ring), bisection-bound
    const tpLink = rsBytes / HW.linkBi * tf.tpAG;                                    // one TP all-gather / reduce-scatter
    switch (op) {
      case 'norm_ag': return tpLink;                                                 // x2 per layer
      case 'qkv': return Math.max(2 * tl * E * 9216 / tp / HW.F_hifi, E * 9216 / tp * BF8 / HW.dram);
      case 'idx_branch': return Math.max(2 * tl * E * 640 / tp / HW.F_hifi, E * 640 * BF8 / HW.dram);
      case 'misc': return 8 * tl * E / tp * BF16 / HW.dram;                       // rope, residual, typecasts, cache write
      case 'o_proj': return Math.max(2 * tl * 8192 / tp * E / HW.F_hifi, 8192 * E / tp * BF8 / HW.dram);
      case 'attn_rs': return tpLink;
      case 'shared': return Math.max(6 * tl * E * M3.Is / tp / HW.F_hifi, 3 * E * M3.Is / tp * BF16 / HW.dram) + tpLink;
      case 'router': return Math.max(2 * tl * E * M3.Ex / HW.F_hifi, E * M3.Ex * BF16 / HW.dram);
      case 'dispatch': return a2a(trl * (M3.topk / tp) * E * 1.0);
      case 'combine': return a2a(trl * (M3.topk / tp) * E * BF16);
      case 'experts': {
        const flops = Tr * M3.topk / P * 6 * E * M3.I * c.imb;
        const wbytes = M3.Ex / P * 3 * E * M3.I * BF4;
        // today's kernel reads weights then computes (measured: time grows linearly with routed tokens even at
        // 160 tokens/expert); a fixed kernel overlaps them -> blend sum -> max with the op-efficiency knob
        const cc = flops / HW.F_lofi, mm = wbytes / HW.dram, ov = c.ovl || 0;
        return (1 - ov) * (cc + mm) + ov * Math.max(cc, mm);
      }
      case 'moe_reduce': return trl * (M3.topk / tp) * E * BF16 * 2 / HW.dram + tpLink;
      case 'dense_mlp': return Math.max(6 * tl * E * M3.Id / tp / HW.F_hifi, 3 * E * M3.Id / tp * BF16 / HW.dram) + tpLink;
    }
    return 0;
  }
  // A galaxy is two 4x4 tori (every row and column of each 4x4 half is a 4-ring). Every measured run and profile
  // was taken with line (mesh) collectives, so the calibrated efficiencies are line efficiencies. An axis that runs
  // as a ring (c.ringSp / c.ringTp, set by makePlan from cfg.torus) gets, per the nappkin lib/ops.py convention:
  //   all-gather / reduce-scatter over N: the busiest link carries (N-1)/2 shards instead of N/2 -> x(N-1)/N;
  //   all-to-all: twice the bisection -> x0.5; ring-joint KV pass: no wrap hop back across the line -> x0.5;
  //   collective latency x0.5.
  // TP-axis collectives: norm_ag, attn_rs and the reduce tails of shared / moe_reduce / dense_mlp.
  // SP-axis collectives: dispatch/combine (all-to-all over the SP line), ag_kv, ag_idx, kv_a2a, ring_scan.
  const TP_CCL = new Set(['norm_ag', 'attn_rs', 'shared', 'moe_reduce', 'dense_mlp']);
  function torusFactors(c) {
    const s = !!c.ringSp, t = !!c.ringTp;
    return { tpAG: t ? (c.tp - 1) / c.tp : 1, spAG: s ? (c.sp - 1) / c.sp : 1, a2a: s ? 0.5 : 1, scan: s ? 0.5 : 1,
      latTp: t ? 0.5 : 1, latSp: s ? 0.5 : 1 };
  }
  const cclLat = (op, c, lat) => { const f = torusFactors(c); return lat.ccl * (TP_CCL.has(op) ? f.latTp : f.latSp); };
  function roofSeg(op, c, s) {
    const { sp, tp } = c;
    // s.nl: query rows per SP rank, when ranks are uneven (owner placement); otherwise an even split
    const nl = s.nl === undefined ? s.n / sp : s.nl, kvlen = s.k + s.n;
    const kvHeadB = 2 * (M3.Hkv / tp) * M3.d * BF8;          // K+V bytes/token on one chip (heads over TP)
    switch (op) {
      case 'ag_kv':
        if (sp <= 1) return 0;
        return (sp - 1) / sp * (c.msaLocal ? Math.min(kvlen, 16384) : kvlen) * kvHeadB / HW.linkBi * torusFactors(c).spAG;
      case 'ag_idx':
        if (sp <= 1 || c.msaLocal) return 0;
        return (sp - 1) / sp * kvlen * M3.di * c.idxB / HW.linkBi * torusFactors(c).spAG;
      case 'indexer': // block-pooled index scoring + top-k; per-token part dominates at these sizes
        return 2 * nl * M3.Hi * M3.di * (2048 + kvlen / 128) / HW.F_hifi;
      case 'sparse': {
        const keys = Math.min(M3.sparseKeys, kvlen);
        return Math.max(4 * nl * (M3.Hq / tp) * keys * M3.d / HW.F_hifi, nl * keys * kvHeadB / 32 / HW.dram);
      }
      case 'kv_a2a': // even split: route the new K/V/index rows to the SP rank that owns them
        if (!c.evenSplit || sp <= 1) return 0;
        return sp * nl * (kvHeadB + M3.di * c.idxB) / (4 * HW.linkUni) * torusFactors(c).a2a;
      case 'ring_c': return 4 * nl * (M3.Hq / tp) * (s.k + s.n / 2) * M3.d / HW.F_hifi;
      case 'ring_scan': { // ring-joint gathers the valid prefix [0, kv_len) (op-bounded since tt-metal #47539);
        // bounded === false models the pre-#47539 op that gathered the whole per-device cache shard (capacity/sp)
        const scan = c.bounded === false ? Math.max(s.cap, kvlen) : kvlen;
        return scan / sp * kvHeadB / HW.linkBi * torusFactors(c).scan;
      }
    }
    return 0;
  }
  const isCcl = (op) => CCL_OPS.has(op);
  // Wave quantization of attention kernels: work units = (32-row query tiles per chip) x (query heads per chip),
  // spread over 110 Tensix cores; a short segment leaves cores idle (sequential per-user attention pays this per
  // segment, a fused multi-user kernel pays it once for the whole batch).
  const CORES = 110;
  function waveFactor(nTok, c) {
    const units = Math.max(1, Math.ceil(nTok / c.sp / 32)) * (M3.Hq / c.tp);
    return Math.ceil(units / CORES) * CORES / units;
  }
  const WAVE_OPS = new Set(['sparse', 'indexer', 'ring_c']);
  const segRows = (s, c) => (s.nl === undefined ? s.n : s.nl * c.sp); // a segment's tokens as seen by its busiest rank
  // Owner placement: each new token is computed on the SP rank that owns its KV row under the block-cyclic layout
  // (blocks of blk = C/SP positions, block j on rank j mod SP), as tt-metal #57636 does for a chunk. Walks
  // positions [p, p+m) block by block and returns how many of them fit with every rank at or below cap rows, given
  // `load` (rows already on each rank); apply = add them to `load` (and to `rows`, the segment's own per-rank rows).
  // p, m, blk and cap are multiples of 32.
  function ownerWalk(load, p, m, cap, blk, sp, apply, rows) {
    const L = apply ? load : load.slice();
    let t = 0;
    while (t < m) {
      const q = p + t, r = Math.floor(q / blk) % sp, piece = Math.min(m - t, blk - (q % blk));
      const take = Math.min(piece, cap - L[r]);
      if (take <= 0) break;
      L[r] += take; if (rows) rows[r] += take; t += take;
      if (take < piece) break;
    }
    return t;
  }

  // ------------------------------------------------------------------------------------------------------
  // Calibration: per-mesh measured efficiencies from zone profiles + pipeline-level fit ([2,4] 16-stage runs)
  // ------------------------------------------------------------------------------------------------------
  const ZONE_T = 5120, ZONE_K = 51200, ZONE_CAP = 56320;
  const IMB0 = 1.2; // expert-load imbalance assumed in the roofline (calibration and DEFAULTS.expertImb)
  const MAX_REQ = 990016; // largest request in the AgentX corpus (the dataset caps input at 990,016 tokens)

  function zoneEff(zones, mesh, idxB) {
    const [sp, tp] = mesh;
    const c = { sp, tp, P: sp * tp, T: ZONE_T, idxB, imb: IMB0, evenSplit: false, bounded: true, msaLocal: false };
    const seg = { n: ZONE_T, k: ZONE_K, cap: ZONE_CAP };
    const eff = { moe: {}, dense: {} };
    const zm = (name) => (zones[name] ? zones[name].mean : null);
    const moeZ = (k) => zm('layer03_sparse/' + k), denZ = (k) => zm('layer00_dense/' + k);
    const fit = (roof, meas, op) => {
      const lat = isCcl(op) ? LAT_MEAS.ccl : LAT_MEAS.op;
      return Math.min(1, Math.max(0.003, roof * 1e3 / Math.max(1e-3, meas - lat)));
    };
    // MoE layer
    let listed = 0;
    for (const op of OPS_MOE.concat(OPS_MSA_SEG)) {
      if (op === 'norm_ag') {
        const m = (moeZ('input_norm_allgather') + moeZ('post_attn_norm_allgather')) / 2;
        eff.moe[op] = fit(roofTok(op, c), m, op); listed += 2 * m; continue;
      }
      if (op === 'misc' || op === 'kv_a2a') continue;
      let m = moeZ(ZONE_MOE[op]);
      if (op === 'shared') m = moeZ('mlp/shared_expert');
      const r = OPS_MSA_SEG.includes(op) ? roofSeg(op, c, seg) : roofTok(op, c);
      eff.moe[op] = fit(r, m, op); listed += m;
    }
    eff.moe.misc = fit(roofTok('misc', c), Math.max(0.05, zm('layer03_sparse') - listed), 'misc');
    eff.moe.kv_a2a = eff.moe.dispatch;
    // dense layer
    listed = 0;
    for (const op of OPS_DENSE) {
      if (op === 'norm_ag') {
        const m = (denZ('input_norm_allgather') + denZ('post_attn_norm_allgather')) / 2;
        eff.dense[op] = fit(roofTok(op, c), m, op); listed += 2 * m; continue;
      }
      if (op === 'misc') continue;
      const m = denZ(ZONE_DENSE[op]);
      eff.dense[op] = fit(roofTok(op, c), m, op); listed += m;
    }
    const ring = denZ('attn/ring_joint_sdpa'); listed += ring;
    eff.dense.misc = fit(roofTok('misc', c), Math.max(0.05, zm('layer00_dense') - listed), 'misc');
    eff.dense.ring_c = fit(roofSeg('ring_c', c, seg), ring, 'ring');
    eff.dense.ring_scan = 0.015; // overwritten by the pipeline fit
    eff.dense.kv_a2a = eff.moe.dispatch;
    return eff;
  }

  function interpEff(a, b, w) { // geometric interpolation of two eff tables
    const out = { moe: {}, dense: {} };
    for (const k of ['moe', 'dense']) for (const op in a[k]) out[k][op] = Math.exp((1 - w) * Math.log(a[k][op]) + w * Math.log(b[k][op]));
    return out;
  }

  // Build a calibration object from calib_data.json (zones + pipeline rows + tables).
  function calibrate(data) {
    const idxB = BF16; // the measured runs use M3_INDEX_CACHE_BF16=1
    const effs = {};
    for (const m in data.zones) effs[m] = zoneEff(data.zones[m], m.split('x').map(Number), idxB);
    // [4,4] has no profile: geometric midpoint of [2,4] and [8,4] (log-chip-count midpoint)
    if (effs['2x4'] && effs['8x4']) effs['4x4'] = interpEff(effs['2x4'], effs['8x4'], 0.5);
    const cal = { effs, pipe: { moeMult: [1, 0], stageOv: [2, 0], ringC: null, ringScan: 0.015, denseFix: 0, embed: 1.6, block: [12, 2.6], hop: [12, 0] }, fit: {} };
    fitPipeline(cal, data);
    return cal;
  }

  // Fit the pipeline-level corrections on the [2,4] 16-stage runs A/B/C.
  function fitPipeline(cal, data) {
    const P = data.pipeline; if (!P || !P.B) return;
    const mesh = [2, 4];
    const base = (T) => ({ sp: 2, tp: 4, P: 8, T, idxB: BF16, imb: IMB0, evenSplit: false, bounded: true, msaLocal: false });
    const eff = cal.effs['2x4'];
    const ctx = (T, na) => Object.assign(base(T), { Tr: na });
    // samples: every (cell, rank, chunk position) median of the loaded blocks; actual tokens and kv are exact
    const samples = [];
    for (const run of ['A', 'B', 'C']) {
      const r = P[run]; const counts = r.layers.split(',').map(Number); const T = r.chunk;
      for (const cell of r.cells || []) {
        const kA = Math.floor(cell.cached / T) * T;
        for (let pos = 0; pos < cell.n_ch; pos++) {
          const na = Math.min(T, cell.new - pos * T);
          let start = 0;
          counts.forEach((n, s) => {
            const nd = Math.max(0, Math.min(3, start + n) - start); start += n;
            const y = cell.pos_ms[s][pos]; if (y == null) return;
            samples.push({ run, T, na, k: kA + pos * T, cap: cell.cached + 51200, s, n, nd, y });
          });
        }
      }
    }
    cal.fit.samples = samples.length;
    const moeZone = (p) => layerMs('moe', ctx(p.T, p.na), [{ n: p.T, na: p.na, k: p.k, cap: p.cap }], eff.moe, LAT_MEAS, 'request');
    // 1) MoE-only stages: stage_ms = nMoE * (a + b*5120/T) * moeZone + o0   (o0 >= 0: per-chunk stage overhead)
    const moeS = samples.filter((p) => p.nd === 0);
    const X = moeS.map((p) => { const z = moeZone(p); return [p.n * z, p.n * z * 5120 / p.T, 1]; });
    const Y = moeS.map((p) => p.y);
    let beta = lstsq(X, Y);
    if (beta[2] < 0) { const b2 = lstsq(X.map((x) => [x[0], x[1]]), Y); beta = [b2[0], b2[1], 0]; }
    cal.pipe.moeMult = [beta[0], beta[1]]; cal.pipe.stageOv = [beta[2], 0];
    cal.fit.moe_rmse = rmse(X, Y, beta);
    // 2) dense ring: single-dense-layer stages (B/C ranks 1,2) -> grid search eff_ring_c, eff_ring_scan and a fixed
    //    per-call cost of the dense attention (d ms). In these runs the lane capacity is cached + 51200, which moves
    //    with kv_len, so a capacity term and a kv term are nearly collinear across cells; within a cell (fixed
    //    capacity, growing kv) and across chunk sizes (the slope scales with T) the data is kv-bound: a joint
    //    regression gives ~210 ms per 1M kv tokens vs ~19 ms per 1M of capacity at chunk 5120 (2.4 at 2048),
    //    and the op gathers only [0, kv_len) since #47539. The missing ~4.5 ms per-call intercept is fitted here.
    //    model = rest + max(ringC_roof / eff_c, scan_roof / eff_s) + d
    const dense = samples.filter((p) => p.nd === 1 && p.n === 1 && p.s > 0);
    const ov = (T) => cal.pipe.stageOv[0] + cal.pipe.stageOv[1] * T / 1000;
    const off = Object.assign({}, eff.dense, { ring_c: 1e9, ring_scan: 1e9 });
    const pre = dense.map((p) => {
      const c = ctx(p.T, p.na), s = { n: p.T, na: p.na, k: p.k, cap: p.cap };
      return {
        R: layerMs('dense', c, [s], off, LAT_MEAS, 'request') + ov(p.T),
        A: roofSeg('ring_c', c, s) * 1e3 * waveFactor(s.n, c) / waveFactor(ZONE_T, c),
        B: roofSeg('ring_scan', c, s) * 1e3, ly: Math.log(p.y),
      };
    });
    let best = null;
    for (let d = 0; d <= 10; d += 0.25) for (let lc = Math.log(0.05); lc <= Math.log(1.0); lc += 0.02) for (let ls = Math.log(0.001); ls <= Math.log(0.5); ls += 0.1) {
      const ic = Math.exp(-lc), is = Math.exp(-ls);
      let err = 0;
      for (const q of pre) { const m = q.R + Math.max(q.A * ic, q.B * is) + d; err += (Math.log(m) - q.ly) ** 2; }
      if (!best || err < best.err) best = { err, rc: Math.exp(lc), rs: Math.exp(ls), d };
    }
    cal.pipe.ringC = best.rc; cal.pipe.ringScan = best.rs; cal.pipe.denseFix = best.d;
    cal.fit.dense_rmse_log = Math.sqrt(best.err / Math.max(1, dense.length));
    // 3) embedding = rank0 - rank1 (both one dense layer) in B/C
    const r0 = samples.filter((p) => p.s === 0 && p.n === 1), r1 = new Map(dense.filter((p) => p.s === 1).map((p) => [`${p.run}|${p.k}|${p.na}`, p.y]));
    const emb = r0.map((p) => p.y - (r1.get(`${p.run}|${p.k}|${p.na}`) ?? p.y)).sort((a, b) => a - b);
    cal.pipe.embed = Math.max(0, emb.length ? emb[Math.floor(emb.length / 2)] : 1.6);
  }

  function lstsq(X, y) {
    const n = X[0].length; const A = Array.from({ length: n }, () => new Array(n).fill(0)); const b = new Array(n).fill(0);
    for (let i = 0; i < X.length; i++) for (let j = 0; j < n; j++) { b[j] += X[i][j] * y[i]; for (let k = 0; k < n; k++) A[j][k] += X[i][j] * X[i][k]; }
    for (let i = 0; i < n; i++) { // gaussian elimination
      let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      [A[i], A[p]] = [A[p], A[i]]; [b[i], b[p]] = [b[p], b[i]];
      for (let r = i + 1; r < n; r++) { const f = A[r][i] / A[i][i]; for (let k = i; k < n; k++) A[r][k] -= f * A[i][k]; b[r] -= f * b[i]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= A[i][k] * x[k]; x[i] = s / A[i][i]; }
    return x;
  }
  function rmse(X, y, beta) { let e = 0; for (let i = 0; i < X.length; i++) { let p = 0; for (let j = 0; j < beta.length; j++) p += X[i][j] * beta[j]; e += (p - y[i]) ** 2; } return Math.sqrt(e / X.length); }

  // ------------------------------------------------------------------------------------------------------
  // Layer cost (ms) for a batch of segments.  kind: 'moe' | 'dense'; attn: 'fused' = one call for all segments, else one per segment
  // ------------------------------------------------------------------------------------------------------
  function opMs(op, roofS, eff, lat, c) {
    return (isCcl(op) ? cclLat(op, c, lat) : lat.op) + roofS * 1e3 / eff[op];
  }
  // attention part of a layer, per-request (seq) or one kernel over the whole chunk (fused).
  // acc.g collects the KV-gather time that a prefetch could hide (MSA ag_kv/ag_idx; the dense ring's gather beyond
  // its compute).
  function attnMs(kind, c, segs, eff, lat, fused, acc) {
    const z = waveFactor(ZONE_T, c);
    let t = 0;
    if (kind === 'moe') {
      for (const op of OPS_MSA_SEG) {
        if (op === 'kv_a2a' && !c.evenSplit) continue;
        if ((op === 'ag_kv' || op === 'ag_idx') && c.sp <= 1) continue;
        if (op === 'ag_idx' && c.msaLocal) continue;
        const l = isCcl(op) ? cclLat(op, c, lat) : lat.op;
        const wave = WAVE_OPS.has(op);
        let sum = 0;
        for (const s of segs) sum += roofSeg(op, c, s) * 1e3 / eff[op] * (wave && !fused ? waveFactor(segRows(s, c), c) / z : 1);
        if (wave && fused) sum *= waveFactor(c.T, c) / z;
        if (acc && (op === 'ag_kv' || op === 'ag_idx')) acc.g += sum;
        if (acc && (op === 'indexer' || op === 'sparse')) acc.w += sum;
        t += sum + (fused ? l : l * segs.length);
      }
    } else {
      const wf = fused ? waveFactor(c.T, c) / z : 0;
      for (const s of segs) {
        const w = fused ? wf : waveFactor(segRows(s, c), c) / z;
        const rc = roofSeg('ring_c', c, s) * 1e3 / eff.ring_c * w, rs = roofSeg('ring_scan', c, s) * 1e3 / eff.ring_scan;
        t += Math.max(rc, rs);
        if (acc) acc.g += Math.max(0, rs - rc);
        if (c.evenSplit && c.sp > 1) t += cclLat('kv_a2a', c, lat) + roofSeg('kv_a2a', c, s) * 1e3 / eff.kv_a2a;
      }
      t += (fused ? 1 : segs.length) * (lat.op + (lat.denseFix || 0)); // per ring-joint call
    }
    return t;
  }
  function layerMs(kind, c, segs, eff, lat, attn) {
    let t = 0, w = 0;
    for (const op of kind === 'moe' ? OPS_MOE : OPS_DENSE) {
      const m = (op === 'norm_ag' ? 2 : 1) * opMs(op, roofTok(op, c), eff, lat, c);
      t += m; if (!isCcl(op)) w += m;
    }
    // KV prefetch: the gather of the cached prefix does not depend on this chunk, so it can run on the links while
    // the cores do the layer's non-collective work (projections, experts, misc, the other segments' attention
    // compute); only the gather time beyond that window stays exposed
    const at = (fused) => {
      const acc = c.prefetch ? { g: 0, w: 0 } : null;
      const sg = fused && segs.length && segs[0].nlf !== undefined ? segs.map((s) => Object.assign({}, s, { nl: s.nlf })) : segs;
      const x = attnMs(kind, c, sg, eff, lat, fused, acc);
      return acc ? x - Math.min(acc.g, w + acc.w) : x;
    };
    // a fused multi-user kernel can always fall back to the per-request schedule, so it is never slower
    const seq = at(false);
    return t + (attn === 'fused' ? Math.min(seq, at(true)) : seq);
  }

  // ------------------------------------------------------------------------------------------------------
  // System plan: stages, layer split, memory -> KV capacity, cost functions
  // ------------------------------------------------------------------------------------------------------
  const DEFAULTS = {
    galaxies: 4, replicas: 1, mesh: [2, 4], stages: 16, split: 'auto',
    opEff: 0,              // 0 = today's measured kernels, 1 = roofline target efficiencies
    torus: 'full',         // ring collectives: 'full' = only [4,4] stages (a whole 4x4 torus) | 'axes' | 'off' (makePlan)
    asyncHandoff: false,   // stage-to-stage D2D overlapped with compute (no blocking send), link-rate transfer
    boundedDense: true,    // dense ring-joint gathers [0, kv_len) (op-bounded since #47539); false = whole lane capacity
    msaLocal: false,       // MSA: SP-local indexer + top-k merge, fetch only selected K/V blocks (no prefix all-gather)
    // index_k cache dtype / de-replicated over TP. Today: bf8 x TP replicas (the deployed runner rejects bf16); the
    // #57827 calibration runs used bf16, so the calibration and validation replays pin idxBf16: true
    idxBf16: false, idxDerep: false,
    // reqPad: what each request's tokens in a batch are padded to. 'chunk' (today) = a multiple of the
    //   chunk C, which is also the block-cyclic KV slab (C/SP rows per SP rank); 'tile' = whole 32-row tiles on every
    //   SP rank (see placement).
    // attn: attention calls in a batch. 'chunk' = one call per chunk of each request, each re-gathering the
    //   request's cached prefix (a request that takes several chunks of a chunk-padded batch; today's kernels process
    //   one chunk at a time); 'request' = one call per request per batch (its prefix gathered once); 'fused' = one call
    //   for the whole batch. 'chunk' and 'request' differ only with batching and chunk padding.
    chunk: 5120, reqPad: 'chunk', batch: false, budget: 16384, attn: 'request', policy: 'rr',
    // placement (tile padding only): where a request's new tokens in a batch are computed. 'even' = contiguous equal
    //   slices over the SP ranks (padded to 32*SP), then an all-to-all writes each K/V row to the rank that owns it.
    //   'owner' = each token on the rank that owns its KV row (the host rotation chunk padding uses), so no
    //   all-to-all and padding only to 32, but ranks are uneven (balanced when C = 32*SP): every rank is padded to the
    //   busiest one, and each rank holds at most budget/SP rows. Chunk padding always places
    //   by owner.
    placement: 'even',
    // prefetchKV: overlap the KV-prefix gathers with the layer's non-collective compute
    prefetchKV: false,
    // batchDynShape: a batch that is not full (e.g. a single request) runs at the tokens it holds (whole chunks /
    //   32*SP granules), as ops do without tracing; false = padded to the full budget, as a traced build with one
    //   fixed shape must be
    batchDynShape: true,
    // batchChunksPerRequest L (round robin, batched, chunk padding): how a batch's budget is split among the requests
    //   waiting for it. Rounds over the queue: in each round every request, in queue order, gets up to L more chunks
    //   (fewer if it needs fewer, or if the budget runs out), until the budget is used or no request needs more.
    //   0 = no limit: the front request takes as many chunks as it can fill, then the next (greedy). 1 = one chunk
    //   per request per round: an even split. A request alone fills the batch whatever L is. Each request makes one
    //   attention call per batch (reading its cached prefix) for all its chunks, so many requests per batch cost
    //   throughput. On the pool, a request joins a batch only if it gets a lane.
    // batchTokensPerRequest: the same for tile padding, in tokens (rounded down to whole tiles, at least one)
    batchChunksPerRequest: 0, batchTokensPerRequest: 0,
    // batchMaxReqs: the most requests in one batch (0 = no limit beyond the budget and, on fixed pool lanes, the lane
    //   count). With static slots a request holds its slot for the whole pipeline trip, so this bounds the slots in
    //   flight to batchMaxReqs x stages; on fixed pool lanes it sets the derived lane count.
    batchMaxReqs: 0,
    cache: 'slots',        // slots | pool | paging | inf
    // pool lanes per stage, derived from r = the most requests a batch holds (1 without
    //   batching, the chunks per batch with chunk padding; every request in a batch needs its own lane)
    //   and b = the buffers of the copy mode (sequential 1, double 2, overlap3 3): per-stage lane table = b x r (a
    //   stage works on one batch at a time); global lane table = stages x r (a lane is held for the whole trip
    //   through the pipeline, one batch per stage in flight; sequential copies only, since a lane reserved for the
    //   whole trip gains nothing from buffering). lanesOverride (batching on
    //   the pool only) sets `lanes` instead; tile-padded batching has no chunks, so it must override. Arena
    //   lanes and the other caches ignore the count and the override.
    lanes: 3, lanesOverride: false, laneScope: 'stage', laneLen: M3.maxCtx, laneArena: false, arenaTokens: 4e6,
    // resume at any 32-token boundary (tt-metal #57636, merged), so the cached prefix is used whole; false rounds it
    // down to a chunk multiple, the pre-#57636 behaviour (CLI only: --set unaligned=false)
    slotLen: M3.maxCtx, unaligned: true,
    // pool copy-in (cached prefix pool->lane) / copy-out (new KV lane->pool):
    //   'sequential' = (default) copy-out, copy-in, then prefill: full copy time on the stage, lanes held only while
    //                  computing
    //   'double'     = double-buffered: copies overlap compute (copyContention of their time is charged for DRAM
    //                  sharing); a lane is held for its own copy-out after the last chunk plus the next occupant's
    //                  copy-in before its first chunk (peak ~2x the computing lanes, briefly)
    //   'overlap3'   = static triple buffering: next batch copying in, current computing, previous copying out, each
    //                  for a whole chunk period (peak 3x)
    copyMode: 'sequential', copyContention: 0.25,
    // KV offload tiers (hostTier): device DRAM -> host DRAM -> SSD, exclusive LRU tiers. A page or slot evicted
    //   from device goes to host DRAM (over PCIe, device -> host), host DRAM evicts to SSD, SSD drops. Hits are read
    //   back before admission: host parts over PCIe (host -> device), SSD parts from the drives then over PCIe.
    //   Per galaxy, from Tenstorrent Galaxy Blackhole Server User Guide v1.8 (docs.tenstorrent.com) unless noted:
    //   - host DRAM 576 GB (6x 96 GB DDR5-4800 RDIMM). The share left for KV is derived in makePlan from the
    //     reserves below: hostHeadroom (fraction kept free), OS and services, the model's runtime (tt-metal,
    //     inference server, program caches), staging for KV migration to decode, UMD's pinned host memory (1 GiB
    //     hugepage channel per PCIe chip: tt-metal asks for min(4, chips per MMIO device) = 1 on a Galaxy, where
    //     every chip has its own PCIe link), and, with hostStageWeights, this galaxy's share of the model weights
    //     (host copies while loading / reloading). About 370 GB of KV per galaxy at 8 galaxies.
    //   - PCIe host <-> chips: per tray one chip on Gen5 x8 and seven on x1 links that train at Gen4 (User Guide
    //     "PCIe Gen5 1x8 and 7x1"; Quanta S7TK product spec "7x1 PCIe G4"). KV is spread evenly over the chips and
    //     each chip moves its own share over its own link, so the x1 chips set the pace: 32 x 1.97 GB/s = 63 GB/s
    //     per direction per galaxy (PCIe Gen4 x1 = 16 GT/s, 128b/130b). Full duplex: fetches (host -> device) and
    //     write-backs (device -> host) do not share it. Routing through each tray's x8 chip over Ethernet would
    //     allow up to 4 x (31.5 + 7 x 1.97) = 181 GB/s, which no software does today.
    //   - SSD: 4x E1.S NVMe; 16 TB per galaxy (planned hardware; shipped systems carry 4x 7.68 TB Samsung PM9D3a
    //     MZTL67T6HBLC = 30.7 TB). PM9D3a: up to 12,000 MB/s sequential read, 6,800 MB/s write (Samsung datasheet, PCIe 5.0 x4);
    //     Rev C chassis run the E1.S links at Gen4 x4 (7.88 GB/s, BIOS setting in the Exabox runbook), so 4x 7.88 =
    //     31.5 GB/s read and 4x 6.8 = 27.2 GB/s write. Reads and writes share the drives (one queue, conservative).
    hostTier: false, hostDramGBPerGalaxy: 576, hostHeadroom: 0.1, hostOsGB: 16, hostRuntimeGB: 32, hostKvStagingGB: 32,
    hostPinnedGBPerChip: 1.073741824, hostStageWeights: true,
    ssdTBPerGalaxy: 16, ssdReadGBsPerGalaxy: 31.5, ssdWriteGBsPerGalaxy: 27.2, pcieGBsPerGalaxy: 63,
    // reserveGB: per-chip DRAM kept free besides the modelled weights and activations; 1 GB reproduces the measured
    // slot fit (35 x 1M slots on 16x[2,4], bf8 index_k, even split; CCL scratch + transient buffers)
    reserveGB: 1, expertImb: IMB0, maxInflight: 0,
    // decodeSlots: KV slots on the decode side (0 = unlimited). A request takes one before it may start prefill (it
    //   waits in FIFO order, inside its TTFT, while all are held) and frees it when decode ends: prefill only runs
    //   requests decode has room for. As in tt-d-gen, which needs the decode slot up front to start KV migration
    //   eagerly. M3 decode today holds ~62-64 sessions (one per pipeline stage, tt-blaze #4220) on 16 galaxies.
    decodeSlots: 0,
    // decode ring: decodeTps = tokens per second per user (TSU) while the ring has room, i.e. one token per trip
    //   through the pipeline (trip time 1 / TSU); decodeStages = the most sessions the ring carries at once, one
    //   token per stage (64 for a 64-stage ring, m x 64 with m-row batched decode; 0 = unlimited). Every request
    //   past prefill decodes at once: with N decoding, each runs at TSU x min(1, decodeStages / N) (processor
    //   sharing), so an oversubscribed ring slows everyone and its sessions hold their KV slots longer. Aggregate
    //   decode throughput is at most decodeStages x TSU.
    decodeStages: 0,
    // decodeCurve: 'flat' = every session decodes at decodeTps; 'm3' = speed falls with the session's context along
    //   the measured M3 curve (M3_DECODE_TSU), with decodeTps the speed at 100k context (180 -> about 213 at 8k, 169
    //   at 140k, 132 at 310k, 101 at 550k). KV migration to decode is not modelled: it streams layer by layer during
    //   prefill, so only the last layer of the last chunk is left at the end, and that overlaps the first token's trip.
    //   Default 'm3' at decodeTps 180: the decode target (180 tokens/s/u @100k) with the measured shape; decodeTps 98
    //   reproduces the measured table.
    decodeCurve: 'm3',
    // decodeCache: the decode side's KV layout. decodeSlots sizes its memory in 1M-token slots (0 = unlimited). Every
    //   layout reserves a request's decode KV when the request is admitted to prefill and releases it when its decode
    //   ends; released KV stays cached (LRU) and a later request with the same prefix finds it there.
    //   'slots'  = (today) decodeSlots fixed 1M slots, one per request; a released slot keeps its stream's KV, LRU over
    //              idle slots (as prefill's static slots).
    //   'hybrid' = decodeLanes 1M lanes the decode kernels run on, the rest of the memory a content-addressed paged
    //              pool (as prefill's pool); a request's pages are pinned in the pool from admission.
    //              decodeLaneScope 'global': a decoding session holds a lane for its whole decode (its context copied
    //                in once when decode starts, then its pages are unpinned); at most decodeLanes sessions decode and
    //                the rest wait for a lane (counted in TTFT). decodeLanes 0 = decodeStages (one per ring stage;
    //                x the batches once decode is batched).
    //              decodeLaneScope 'stage': decodeLanes lanes per stage (0 = 1). With more sessions decoding than
    //                lanes, every token copies the session's whole context pool -> lane on each stage, as prefill's
    //                per-stage lanes do for every chunk, which slows every session (decCopyTrip).
    //   'paging' = an ideal paged decode kernel: all of the memory is the pool, no lanes, no copies.
    // decodeHostTier: released slots and unpinned pages go on to the decode galaxies' host DRAM, then SSD (the
    //   hostTier specs per galaxy, decodeGalaxies of them), and are read back (before decode starts) when their prefix
    //   is needed again. KV migration from prefill is not modelled, so a decode-side hit only saves migration
    //   (reported as decHitRate and migTps).
    decodeCache: 'slots', decodeLanes: 0, decodeLaneScope: 'global', decodeHostTier: false, decodeGalaxies: 16,
    // decodeBackpressure (fixed decode slots with the decode offload tiers): what a request needs from decode before
    //   its prefill may start.
    //   'slot'  (today) = a free decode slot; it waits (in TTFT) until one frees.
    //   'queue' = a free slot if there is one (KV migrates straight into it); otherwise its KV migrates to the decode
    //             galaxies' SSD (parked: PCIe + SSD write) and it takes a slot once its prefill is done and one frees
    //             (FIFO, ahead of new admissions), then reads its KV back from SSD (what the slot does not already
    //             hold) before its first token. A request waits before prefill only while decodeQueueMax requests
    //             (0 = no limit) are parked.
    decodeBackpressure: 'slot', decodeQueueMax: 0,
    concurrency: 64, decodeTps: 180, duration: 1800, seed: 1, idleCap: 10, startMin: 0, startMax: 1, maxWarmup: 1e6,
    gapCap: Infinity,      // AgentX forbids capping recorded idle gaps (only the 10 s system-idle cap applies)
    // policy: 'rr' (default) = round robin, see below; 'rtc' = run to completion (no preemption), oldest first
    //   ('fcfs', its old name, is still accepted): every chunk continues the oldest started request, new requests
    //   start only when no started one has tokens left (batched: started requests first, new ones fill the leftover
    //   room); the study and the configs built from it (study.js withFeatures) use 'rtc';
    //   'srpt' = also run to completion, with the waiting queue sorted shortest-new-first (srptMaxWait s of aging);
    //   'rr' = round robin over the started requests (tt-d-gen PrefillQueue/PrefillWriter): a request is admitted
    //   (slot/lane acquired) at the back of the queue; each turn the front request takes one chunk, or with batching
    //   as many of the batch's remaining chunks as it can fill (one attention call), and goes to the back if it has
    //   tokens left (it is never split into two runs within one batch).
    srptMaxWait: 30,
    // Round robin on the pool (policy 'rr', cache 'pool'). A lane is a per-stage KV slot the attention kernels run on
    //   (fixed laneLen, or request-sized in the arena). Every turn copies the segment's new KV out to the pool, where
    //   the partial KV stays pinned until the request finishes, so a lane never holds the only copy and can be handed
    //   to another request at any turn. The lane count is the same as for the other policies (see `lanes`).
    //   Copy-in: a request whose lane still holds its context (not handed to another request since its last turn)
    //     reuses it without a copy-in; on a miss it takes a free lane, else the least recently used one, and copies
    //     its whole context so far in. Arena lanes always copy in.
    //   With the pool, and with paging, round robin admits a request only while the pool can hold every in-progress
    //   request in full (like tt-d-gen, where admission needs a free slot and in-flight slots are never evicted).
  };

  function kvBytesPerTokenLayer(cfg, tp) { // whole stage, physical
    return 2 * M3.Hkv * M3.d * BF8 + M3.di * (cfg.idxBf16 ? BF16 : BF8) * (cfg.idxDerep ? 1 : tp);
  }

  // Layer split over S stages (one entry per stage, summing to L). 'auto' minimises the bottleneck: the most time any
  //   stage spends per chunk, with model = { tm, td, ov, embed } (ms per MoE / dense layer averaged over the chunks'
  //   KV lengths, per-stage overhead, stage 0's embedding). In a pipeline each stage's busy time adds up over the
  //   chunk stream, so throughput follows the stage with the most mean time per chunk. Dense layers (the first
  //   nDense) may share a stage with each other and with MoE layers. A DP over contiguous partitions finds the least
  //   bottleneck B; a second DP picks, among the splits with every stage <= B, the one with the least sum of squared
  //   stage times, so the stages off the bottleneck stay balanced too. O(S L^2) each, about a millisecond.
  function splitLayers(cfg, S, model) {
    const L = M3.L, D = M3.nDense;
    if (Array.isArray(cfg.split)) return cfg.split.slice();
    const even = () => { const b = Math.floor(L / S), r = L % S; return Array.from({ length: S }, (_, i) => b + (i < r ? 1 : 0)); };
    if (cfg.split === 'even' || S === 1 || S > L) return even();
    const { tm, td, ov, embed } = model;
    const pre = new Float64Array(L + 1);
    for (let l = 0; l < L; l++) pre[l + 1] = pre[l] + (l < D ? td : tm);
    const sc = (i, j) => pre[j] - pre[i] + ov + (i === 0 ? embed : 0); // stage holding layers [i, j)
    // pass 1: f[j] = least bottleneck of s stages over layers [0, j)
    let f = new Float64Array(L + 1).fill(Infinity); f[0] = 0;
    for (let s = 1; s <= S; s++) {
      const g = new Float64Array(L + 1).fill(Infinity);
      for (let j = s; j <= L - (S - s); j++) for (let i = s - 1; i < j; i++) g[j] = Math.min(g[j], Math.max(f[i], sc(i, j)));
      f = g;
    }
    const B = f[L] * (1 + 1e-9);
    // pass 2: least sum of squared stage times with every stage <= B, with back pointers; the tiny s * c term breaks
    // ties between equal-cost orderings towards the lighter stages at the end (the last stage also holds the LM head)
    let q = new Float64Array(L + 1).fill(Infinity); q[0] = 0;
    const from = [];
    for (let s = 1; s <= S; s++) {
      const g = new Float64Array(L + 1).fill(Infinity), bp = new Int32Array(L + 1);
      for (let j = s; j <= L - (S - s); j++) for (let i = s - 1; i < j; i++) {
        const c = sc(i, j); if (c > B) continue;
        const v = q[i] + c * c + 1e-6 * s * c; if (v < g[j]) { g[j] = v; bp[j] = i; }
      }
      q = g; from.push(bp);
    }
    const counts = new Array(S);
    for (let s = S - 1, j = L; s >= 0; s--) { const i = from[s][j]; counts[s] = j - i; j = i; }
    return counts;
  }

  function makePlan(cfgIn, cal) {
    const cfg = Object.assign({}, DEFAULTS, cfgIn);
    const tile = cfg.reqPad === 'tile';
    const [sp, tp] = cfg.mesh; const P = sp * tp; const S = cfg.stages;
    // tile padding: the chunk is only the KV slab, fixed at 128*SP (MSA needs whole 128-row KV blocks on each rank);
    // the token budget sizes the batch, with or without batching
    if (tile) cfg.chunk = 128 * sp;
    const chipsRep = 32 * cfg.galaxies / cfg.replicas;
    const errors = [];
    if (S * P > chipsRep + 1e-9) errors.push(`needs ${S * P} chips per replica, have ${chipsRep}`);
    if (!Number.isInteger(cfg.galaxies / cfg.replicas)) errors.push('galaxies must divide evenly into replicas');
    if (M3.Hkv % tp !== 0) errors.push('tp must divide 4 KV heads');
    const meshKey = `${sp}x${tp}`;
    const effMeas = cal.effs[meshKey] || cal.effs['2x4'];
    if (!cal.effs[meshKey]) errors.push(`no profile for mesh ${meshKey}; using [2,4] efficiencies`);
    const a = cfg.opEff;
    const eff = { moe: {}, dense: {} };
    for (const k of ['moe', 'dense']) for (const op in effMeas[k]) {
      let em = effMeas[k][op]; // MoE: the pipeline multiplier is applied on the layer total below
      if (k === 'dense' && op === 'ring_c' && cal.pipe.ringC) em = cal.pipe.ringC * (effMeas.dense.ring_c / cal.effs['2x4'].dense.ring_c);
      if (k === 'dense' && op === 'ring_scan') em = cal.pipe.ringScan;
      // opEff 0 = exactly the measured efficiency; the target never makes an op slower than it measures today
      const et = Math.max(em, TARGET_EFF[OP_CLASS[op] || 'matmul']);
      eff[k][op] = Math.exp((1 - a) * Math.log(em) + a * Math.log(et));
    }
    // ring axes (see torusFactors). 'full': only a stage that is a whole 4x4 torus runs ring collectives (fabric
    // torus mode needs the whole torus); 'axes': any 4-long axis that spans a torus row/column keeps its wrap link
    // ([2,4] -> TP ring, [4,2] -> SP ring, [8,4] -> TP ring); 'off': line collectives everywhere (the measured setup)
    const tmode = cfg.torus === true ? 'full' : cfg.torus === false ? 'off' : cfg.torus;
    const ringSp = sp === 4 && (tmode === 'axes' || (tmode === 'full' && tp === 4));
    const ringTp = tp === 4 && (tmode === 'axes' || (tmode === 'full' && sp === 4));
    // denseFix: fitted fixed cost of one dense ring-joint call (~4.5 ms today); roofline kernels shrink it to <= 1 ms
    const fix0 = cal.pipe.denseFix || 0;
    const lat = { ccl: (1 - a) * LAT_MEAS.ccl + a * LAT_TGT.ccl, op: (1 - a) * LAT_MEAS.op + a * LAT_TGT.op, denseFix: (1 - a) * fix0 + a * Math.min(fix0, 1) };
    const idxB = cfg.idxBf16 ? BF16 : BF8;
    const owner = tile && cfg.placement === 'owner';
    const gran = owner ? 32 : 32 * sp; // tile padding: request padding unit
    // evenSplit: contiguous equal slices over SP, so the new K/V rows need an all-to-all to reach the ranks that own them
    const evenSplit = tile && !owner;
    const ctxT = (T) => ({ sp, tp, P, T, idxB, imb: cfg.expertImb, evenSplit, bounded: cfg.boundedDense, msaLocal: cfg.msaLocal, ringSp, ringTp, prefetch: !!cfg.prefetchKV });
    const pm = cal.pipe;
    // the 1/T term was fitted on chunks of 2048 and 5120 tokens; clamp so small chunks do not extrapolate it
    const moeMult = (T) => (1 - a) * (pm.moeMult[0] + pm.moeMult[1] * 5120 / Math.max(T, 2048)) + a * 1;
    const ov0 = (T) => Math.max(0, pm.stageOv[0] + pm.stageOv[1] * T / 1000);
    const stageOv = (T) => (1 - a) * ov0(T) + a * Math.min(ov0(T), 0.3);
    const embedMs = (1 - a) * pm.embed + a * Math.min(pm.embed, 0.3);
    const actBytes = (T) => 12 * (T / sp) * M3.E * BF16;
    // largest batch: the budget with tile padding (batched or not), whole chunks otherwise
    const Tchunk = tile ? cfg.budget : cfg.batch ? Math.max(cfg.budget, cfg.chunk) : cfg.chunk;
    const Tmax = Tchunk;
    // pool lanes per stage (see DEFAULTS.lanes): buffers x the most requests per batch, or lanesOverride
    const laneBuffers = cfg.copyMode === 'overlap3' ? 3 : cfg.copyMode === 'double' ? 2 : 1;
    const maxR = cfg.batch && cfg.batchMaxReqs > 0 ? cfg.batchMaxReqs : Infinity;
    // requests per batch for the derived lane count; 0 = tile padding without a cap (no chunks to count)
    const reqsPerBatch = !cfg.batch ? 1 : tile ? (maxR < Infinity ? maxR : 0) : Math.min(maxR, Math.max(1, Math.floor(Tchunk / cfg.chunk)));
    const fixedLanes = cfg.cache === 'pool' && !cfg.laneArena;
    // handoff: measured = blocking send + hop latency (fitted, per chunk); async = link-rate transfer, overlapped
    const actXfer = (T) => T * M3.E * BF16 / (P * HW.linkUni * 0.5) * 1e3; // ms, each chip ships its shard
    const blockMs = (T) => (cfg.asyncHandoff ? 0 : Math.max(0, pm.block[0] + pm.block[1] * T / 1000));
    const hopMs = (T) => (cfg.asyncHandoff ? 0.5 + actXfer(T) : Math.max(0.5, pm.hop[0] + pm.hop[1] * T / 1000));
    const layer = (kind, T, segs, trl) => {
      const c = ctxT(T);
      let tr = 0; for (const s of segs) tr += s.na === undefined ? s.n : s.na;
      c.Tr = tr; c.trl = trl;
      const t = layerMs(kind, c, segs, eff[kind], lat, cfg.attn);
      return kind === 'moe' ? t * moeMult(T) : t;
    };
    // auto split objective: layer costs averaged over the KV lengths chunks see in the AgentX replay, one point per
    // decile of prefilled tokens (the decile midpoints, mean ~140k; measured with cache 'inf' at 16 and 32 stages
    // and with static slots, which all agree within ~10%)
    const repKv = [5e3, 24e3, 40e3, 60e3, 80e3, 105e3, 140e3, 190e3, 270e3, 460e3];
    const Trep = Tchunk;
    // capacity the dense ring-joint scans: fixed-size slots/lanes; 0 = request-sized (arena, paging, inf) -> kv_len
    const laneCap = cfg.cache === 'slots' ? cfg.slotLen : cfg.cache === 'pool' && !cfg.laneArena ? cfg.laneLen : 0;
    let tmRep = 0, tdRep = 0;
    for (const kv of repKv) {
      const segs = [{ n: Trep, k: kv, cap: laneCap }];
      tmRep += layer('moe', Trep, segs) / repKv.length; tdRep += layer('dense', Trep, segs) / repKv.length;
    }
    const counts = splitLayers(cfg, S, { tm: tmRep, td: tdRep, ov: stageOv(Trep), embed: embedMs });
    if (counts.length !== S || counts.reduce((x, y) => x + y, 0) !== M3.L) errors.push('split must have one entry per stage summing to 60');
    // per-stage layer kinds + memory
    const stages = []; let st = 0;
    const kvbL = kvBytesPerTokenLayer(cfg, tp);
    const attnW = 110.9e6 * BF8, sharedW = 3 * M3.E * M3.Is * BF16, denseW = 3 * M3.E * M3.Id * BF16, expW = 3 * M3.E * M3.I * BF4;
    let capTok = Infinity, wTot = 0;
    for (let s = 0; s < counts.length; s++) {
      const n = counts[s]; const nd = Math.max(0, Math.min(M3.nDense, st + n) - st); const nm = n - nd; st += n;
      let w = nm * (M3.Ex * expW + attnW * sp + sharedW * sp + M3.E * M3.Ex * BF16 * P) + nd * (attnW * sp + denseW * sp);
      if (s === 0) w += M3.V * M3.E * BF16;
      if (s === counts.length - 1) w += M3.V * M3.E * BF8;
      wTot += w;
      const free = P * (HW.dramCap - cfg.reserveGB * GB - actBytes(Tmax)) - w;
      const cap = free / (n * kvbL);
      capTok = Math.min(capTok, cap);
      stages.push({ n, nd, nm, weightsGBperChip: w / P / GB, capTok: cap });
    }
    capTok = Math.max(0, capTok);
    // cache layout
    let nSlots = 0, poolTok = 0, lanes = Infinity, arena = 0;
    if (cfg.cache === 'slots') { nSlots = Math.floor(capTok / cfg.slotLen); lanes = nSlots; }
    else if (cfg.cache === 'pool') {
      if (cfg.laneArena) { arena = Math.min(capTok, cfg.arenaTokens); poolTok = capTok - arena; }
      else {
        const r = Math.max(1, reqsPerBatch);
        lanes = cfg.lanesOverride ? cfg.lanes : cfg.laneScope === 'stage' ? laneBuffers * r : S * r;
        poolTok = capTok - lanes * cfg.laneLen;
      }
      if (poolTok < 0) errors.push('lanes do not fit in memory');
    } else if (cfg.cache === 'paging') poolTok = capTok;
    else poolTok = Infinity;
    if (cfg.cache === 'slots' && nSlots < 1) errors.push('no 1M slot fits in memory');
    // static slots: a slot is held for the request's whole trip through the pipeline (today's slot_id), so a full
    // pipeline has (requests per batch) x (stages) requests in flight, each in its own 1M slot. A tile-padded
    // batch has no chunks: up to budget / (32*SP) requests (budget / 32 with owner placement)
    if (cfg.cache === 'slots') {
      const r = !cfg.batch ? 1 : tile ? Math.min(maxR, Math.floor(Tchunk / gran)) : reqsPerBatch;
      if (r * S > nSlots) errors.push(`out of memory: ${r} requests per batch x ${S} stages = ${r * S} slots in flight, but only ${nSlots} 1M slots fit`);
    }
    // the override only applies to fixed pool lanes (other caches have none and ignore it, as they ignore `lanes`,
    // so a pool config can be re-run with cache 'inf' as is); without batching the count is derived
    if (cfg.cache === 'pool' && cfg.laneScope === 'global' && cfg.copyMode !== 'sequential') errors.push('a global lane table holds the lane for the whole pipeline trip, so it allows only sequential copies (one buffer)');
    if (cfg.lanesOverride && cfg.cache === 'pool' && !cfg.batch) errors.push('overriding the lane count needs batching (without it the count is derived)');
    if (fixedLanes && cfg.batch && reqsPerBatch === 0 && !cfg.lanesOverride) errors.push('tile-padded batching on the pool needs the lane count set (lanesOverride) or a request cap (batchMaxReqs)');
    if (cfg.lanesOverride && !(cfg.lanes >= 1)) errors.push('the lane count must be at least 1');
    for (const k of ['hostDramGBPerGalaxy', 'ssdTBPerGalaxy']) if (!(cfg[k] >= 0)) errors.push(`${k} must be >= 0`);
    for (const k of ['pcieGBsPerGalaxy', 'ssdReadGBsPerGalaxy', 'ssdWriteGBsPerGalaxy']) if (!(cfg[k] > 0)) errors.push(`${k} must be > 0`);
    if (!(cfg.decodeStages >= 0)) errors.push('decode pipeline stages must be >= 0 (0 = unlimited)');
    if (!['flat', 'm3'].includes(cfg.decodeCurve)) errors.push(`decodeCurve must be 'flat' or 'm3', got ${cfg.decodeCurve}`);
    if (!(cfg.batchMaxReqs >= 0) || cfg.batchMaxReqs !== Math.floor(cfg.batchMaxReqs)) errors.push('max requests per batch must be a whole number >= 0 (0 = no limit)');
    if (!(cfg.batchChunksPerRequest >= 0) || cfg.batchChunksPerRequest !== Math.floor(cfg.batchChunksPerRequest)) errors.push('chunks per request per round must be a whole number >= 0 (0 = no limit)');
    if (!(cfg.batchTokensPerRequest >= 0)) errors.push('tokens per request per round must be >= 0 (0 = no limit)');
    // every buffer that must hold a whole request has to fit the largest AgentX request (990,016 tokens)
    if (cfg.cache === 'slots' && cfg.slotLen < MAX_REQ) errors.push(`slots must hold the largest request (${MAX_REQ} tokens)`);
    if (cfg.cache === 'pool' && !cfg.laneArena && cfg.laneLen < MAX_REQ) errors.push(`lanes must hold the largest request (${MAX_REQ} tokens)`);
    if (cfg.cache === 'pool' && cfg.laneArena && arena < MAX_REQ) errors.push(`the lane arena must hold the largest request (${MAX_REQ} tokens)`);
    // host copies do not need the TP replicas of index_k (re-broadcast on fetch)
    const kvbHost = kvBytesPerTokenLayer(Object.assign({}, cfg, { idxDerep: true }), tp);
    // offload tiers, per replica (its galaxies' hosts and drives); host DRAM left for KV after the reserves
    const gpr = cfg.galaxies / cfg.replicas, offload = cfg.hostTier && cfg.cache !== 'inf';
    const hostBudget = {
      dram: cfg.hostDramGBPerGalaxy, headroom: cfg.hostDramGBPerGalaxy * cfg.hostHeadroom, os: cfg.hostOsGB, runtime: cfg.hostRuntimeGB,
      staging: cfg.hostKvStagingGB, pinned: 32 * cfg.hostPinnedGBPerChip, weights: cfg.hostStageWeights ? wTot / GB / gpr : 0,
    };
    hostBudget.kv = Math.max(0, hostBudget.dram - hostBudget.headroom - hostBudget.os - hostBudget.runtime - hostBudget.staging - hostBudget.pinned - hostBudget.weights);
    const tokOf = (bytesPerGalaxy) => bytesPerGalaxy * gpr / (M3.L * kvbHost);
    const hostTok = offload ? tokOf(hostBudget.kv * GB) : 0;
    const ssdTok = offload ? tokOf(cfg.ssdTBPerGalaxy * 1e12) : 0;
    // decode side (see DEFAULTS.decodeCache): memory, lanes, pool and the decode galaxies' offload tiers
    const dc = cfg.decodeCache, decMem = cfg.decodeSlots > 0 ? cfg.decodeSlots * M3.maxCtx : Infinity;
    const decLanes = dc !== 'hybrid' ? 0 : cfg.decodeLanes > 0 ? cfg.decodeLanes : cfg.decodeLaneScope === 'global' ? cfg.decodeStages : 1;
    const decPoolTok = dc === 'slots' ? 0 : decMem - decLanes * M3.maxCtx;
    if (!['slots', 'hybrid', 'paging'].includes(dc)) errors.push(`decodeCache must be 'slots', 'hybrid' or 'paging', got ${dc}`);
    if (!['global', 'stage'].includes(cfg.decodeLaneScope)) errors.push(`decodeLaneScope must be 'global' or 'stage', got ${cfg.decodeLaneScope}`);
    if (!(cfg.decodeLanes >= 0) || cfg.decodeLanes !== Math.floor(cfg.decodeLanes)) errors.push('decode lanes must be a whole number >= 0 (0 = derived)');
    if (dc === 'hybrid' && !(cfg.decodeStages > 0)) errors.push('hybrid decode needs the decode pipeline stages (decodeStages > 0)');
    if (dc !== 'slots' && decPoolTok < MAX_REQ) errors.push(`the decode pool (${decPoolTok / M3.maxCtx} 1M slots after ${decLanes} lanes) must hold the largest request (${MAX_REQ} tokens)`);
    if (cfg.decodeHostTier && !(cfg.decodeGalaxies >= 1)) errors.push('decode offload tiers need decodeGalaxies >= 1');
    if (!['slot', 'queue'].includes(cfg.decodeBackpressure)) errors.push(`decodeBackpressure must be 'slot' or 'queue', got ${cfg.decodeBackpressure}`);
    if (cfg.decodeBackpressure === 'queue' && !(dc === 'slots' && cfg.decodeSlots > 0 && cfg.decodeHostTier && cfg.ssdTBPerGalaxy > 0)) errors.push('queue backpressure parks KV on the decode SSDs: it needs fixed decode slots (decodeSlots > 0) and the decode offload tiers with an SSD');
    if (!(cfg.decodeQueueMax >= 0)) errors.push('decodeQueueMax must be >= 0 (0 = no limit)');
    const decOffload = cfg.decodeHostTier && decMem !== Infinity;
    // decode galaxies' host DRAM left for KV: the prefill reserves, with this galaxy's share of the whole model's weights
    const hb = hostBudget;
    const decHostKvGB = Math.max(0, hb.dram - hb.headroom - hb.os - hb.runtime - hb.staging - hb.pinned - (cfg.hostStageWeights ? wTot / GB / cfg.decodeGalaxies : 0));
    const decHostTok = decOffload ? decHostKvGB * GB * cfg.decodeGalaxies / (M3.L * kvbHost) : 0;
    const decSsdTok = decOffload ? cfg.ssdTBPerGalaxy * 1e12 * cfg.decodeGalaxies / (M3.L * kvbHost) : 0;
    const decLayers = cfg.decodeStages > 0 ? Math.ceil(M3.L / cfg.decodeStages) : M3.L; // most layers on one decode stage
    if (!['chunk', 'request', 'fused'].includes(cfg.attn)) errors.push(`attn must be 'chunk', 'request' or 'fused', got ${cfg.attn}`);
    if (!['chunk', 'tile'].includes(cfg.reqPad)) errors.push(`reqPad must be 'chunk' or 'tile', got ${cfg.reqPad}`);
    if (!['even', 'owner'].includes(cfg.placement)) errors.push(`placement must be 'even' or 'owner', got ${cfg.placement}`);
    return {
      cfg, sp, tp, P, S, counts, stages, eff, lat, layer, stageOv, embedMs, blockMs, hopMs, errors,
      capTok, nSlots, poolTok, lanes, arena, hostTok, ssdTok, hostBudget, kvbL, kvbHost, gran, laneCap, owner, maxR,
      // owner placement: KV block per rank, and the most rows a rank holds per batch (whole 32-row tiles)
      blk: cfg.chunk / sp, rankCap: Math.floor(cfg.budget / sp / 32) * 32,
      // time (ms) to copy `tok` tokens of KV between pool and lane on the stage holding the most layers
      copyMs: (tok) => tok * Math.max(...stages.map((x) => x.n)) * kvbL / P * 2 / (HW.dram * 0.5) * 1e3,
      // per replica: PCIe per direction, SSD read / write
      pcieBps: cfg.pcieGBsPerGalaxy * GB * gpr, ssdRdBps: cfg.ssdReadGBsPerGalaxy * GB * gpr, ssdWrBps: cfg.ssdWriteGBsPerGalaxy * GB * gpr,
      // in-flight chunks: round robin on static slots mirrors tt-d-gen's ChunkFifo, max(8, 4 x max_slots)
      maxInflight: cfg.maxInflight > 0 ? cfg.maxInflight : cfg.policy === 'rr' && cfg.cache === 'slots' ? Math.max(8, 4 * nSlots) : 2 * S + 4,
      // decode side: lanes, pool tokens, offload tiers (tokens) and links of the decode galaxies; copy times (s) of a
      // context of `tok` tokens pool -> lane: on one stage (a global lane's copy-in, all stages in parallel) and summed
      // over every layer (what each token of a session adds to its trip with per-stage lanes)
      decLanes, decPoolTok, decHostTok, decSsdTok, decHostKvGB,
      decPcieBps: cfg.pcieGBsPerGalaxy * GB * cfg.decodeGalaxies, decSsdRdBps: cfg.ssdReadGBsPerGalaxy * GB * cfg.decodeGalaxies, decSsdWrBps: cfg.ssdWriteGBsPerGalaxy * GB * cfg.decodeGalaxies,
      decLaneCopyS: (tok) => decCopyS(tok, decLayers), decCopyTrip: (tok) => decCopyS(tok, M3.L),
      tokensPerSec: null,
    };
  }

  // stage times (ms) for one chunk; extra[s] adds per-stage copy costs
  function chunkStageMs(plan, T, segs, out, trl) {
    const tm = plan.layer('moe', T, segs, trl), td = plan.layer('dense', T, segs, trl);
    const ov = plan.stageOv(T);
    for (let s = 0; s < plan.S; s++) {
      const st = plan.stages[s];
      out[s] = st.nm * tm + st.nd * td + ov + (s === 0 ? plan.embedMs : 0);
    }
    return out;
  }

  // ------------------------------------------------------------------------------------------------------
  // Traffic (binary written by prep_traffic.py)
  // ------------------------------------------------------------------------------------------------------
  function loadTraffic(buf, header) {
    const A = {};
    const ctor = { u1: Uint8Array, u2: Uint16Array, u4: Uint32Array, i4: Int32Array, f4: Float32Array };
    for (const L of header.layout) A[L.name] = new ctor[L.dtype](buf, L.offset, L.n);
    const nT = A.tr_req0.length, nR = A.req_blocks.length, nS = A.st_kind.length, nP = A.pc_len.length;
    // piece depth (for LRU tie-break) and per-trace piece counts
    const pdepth = new Int32Array(nP);
    for (let q = 0; q < nP; q++) pdepth[q] = A.pc_parent[q] < 0 ? 0 : pdepth[A.pc_parent[q]] + 1;
    const trPieces = new Int32Array(nT), trReqs = new Int32Array(nT), trStreams = new Int32Array(nT);
    for (let t = 0; t < nT; t++) {
      trPieces[t] = (t + 1 < nT ? A.tr_pc0[t + 1] : nP) - A.tr_pc0[t];
      trReqs[t] = (t + 1 < nT ? A.tr_req0[t + 1] : nR) - A.tr_req0[t];
      trStreams[t] = (t + 1 < nT ? A.tr_st0[t + 1] : nS) - A.tr_st0[t];
    }
    // spawn lists: main request -> child streams (subagents, their sibling chains, flat chains) it spawns
    const spawnHead = new Int32Array(nR).fill(-1), spawnNext = new Int32Array(nS).fill(-1);
    for (let s = nS - 1; s >= 0; s--) if (A.st_kind[s] !== 0 && A.st_spawn[s] >= 0) { spawnNext[s] = spawnHead[A.st_spawn[s]]; spawnHead[A.st_spawn[s]] = s; }
    // cross-stream replay barriers (optional CSR arrays; older traffic.bin files have none)
    const hasPred = !!(A.pred_head && A.pred_list && A.pred_head.length === nR + 1);
    return Object.assign(A, { nT, nR, nS, nP, pdepth, trPieces, trReqs, trStreams, spawnHead, spawnNext, hasPred, stats: header.stats });
  }

  // ------------------------------------------------------------------------------------------------------
  // Utilities: RNG, heaps
  // ------------------------------------------------------------------------------------------------------
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  class EvHeap { // (time, seq) ordered events with payload objects: equal times pop in insertion (FIFO) order
    constructor() { this.t = []; this.s = []; this.p = []; this.seq = 0; }
    get size() { return this.t.length; }
    push(t, p) {
      const T = this.t, Q = this.s, Pp = this.p, q = this.seq++; let i = T.length; T.push(t); Q.push(q); Pp.push(p);
      while (i > 0) { const j = (i - 1) >> 1; if (T[j] < t || (T[j] === t && Q[j] < q)) break; T[i] = T[j]; Q[i] = Q[j]; Pp[i] = Pp[j]; i = j; }
      T[i] = t; Q[i] = q; Pp[i] = p;
    }
    peekT() { return this.t[0]; }
    pop() {
      const T = this.t, Q = this.s, Pp = this.p; const top = Pp[0]; const lt = T.pop(), lq = Q.pop(), lp = Pp.pop(); const n = T.length;
      const lessThan = (a, b) => T[a] < T[b] || (T[a] === T[b] && Q[a] < Q[b]);
      if (n > 0) {
        let i = 0;
        while (true) {
          let l = 2 * i + 1; if (l >= n) break; const r = l + 1; if (r < n && lessThan(r, l)) l = r;
          if (T[l] > lt || (T[l] === lt && Q[l] > lq)) break;
          T[i] = T[l]; Q[i] = Q[l]; Pp[i] = Pp[l]; i = l;
        }
        T[i] = lt; Q[i] = lq; Pp[i] = lp;
      }
      return top;
    }
    shiftAll(dt) { for (let i = 0; i < this.t.length; i++) this.t[i] -= dt; }
  }

  class LruHeap { // min-heap on (t, -depth, key) over piece keys, lazy invalidation; fully deterministic order
    constructor() { this.t = new Float64Array(1024); this.d = new Int32Array(1024); this.k = new Int32Array(1024); this.n = 0; }
    less(i, j) {
      const t = this.t, d = this.d;
      return t[i] < t[j] || (t[i] === t[j] && (d[i] > d[j] || (d[i] === d[j] && this.k[i] < this.k[j])));
    }
    push(t, d, k) {
      if (this.n === this.t.length) { const g = (A, C) => { const B = new C(A.length * 2); B.set(A); return B; }; this.t = g(this.t, Float64Array); this.d = g(this.d, Int32Array); this.k = g(this.k, Int32Array); }
      let i = this.n++; this.t[i] = t; this.d[i] = d; this.k[i] = k;
      while (i > 0) { const j = (i - 1) >> 1; if (!this.less(i, j)) break; this.swap(i, j); i = j; }
    }
    swap(i, j) { let x = this.t[i]; this.t[i] = this.t[j]; this.t[j] = x; x = this.d[i]; this.d[i] = this.d[j]; this.d[j] = x; x = this.k[i]; this.k[i] = this.k[j]; this.k[j] = x; }
    pop() { // returns index 0 content after removal via out fields
      const t = this.t[0], d = this.d[0], k = this.k[0]; this.n--;
      if (this.n > 0) { this.t[0] = this.t[this.n]; this.d[0] = this.d[this.n]; this.k[0] = this.k[this.n]; let i = 0;
        while (true) { let l = 2 * i + 1; if (l >= this.n) break; const r = l + 1; if (r < this.n && this.less(r, l)) l = r; if (!this.less(l, i)) break; this.swap(i, l); i = l; } }
      this.ot = t; this.ok = k; return k;
    }
  }

  // ------------------------------------------------------------------------------------------------------
  // KV residency
  // ------------------------------------------------------------------------------------------------------
  // Content-addressed paged pool over prefix-tree pieces with exact LRU (piece granularity == 64-token pages).
  // Optional second tier (host DRAM): device evictions demote to host; host evictions drop.
  class PoolCache {
    // tiers: 0 none, 1 device, 2 host DRAM, 3 SSD (exclusive); onDemote(blocks, from, to) on every move down
    constructor(TR, capBlocks, hostBlocks, ssdBlocks) {
      this.TR = TR; this.cap = capBlocks; this.hcap = hostBlocks; this.scap = ssdBlocks || 0;
      this.sused = 0; this.sheap = new LruHeap();
      this.n = 0; this.tier = new Uint8Array(1 << 16); this.t = new Float64Array(1 << 16); this.pk = new Int32Array(1 << 16);
      this.len = new Int32Array(1 << 16); this.dep = new Int32Array(1 << 16); this.pin = new Int32Array(1 << 16);
      this.used = 0; this.hused = 0; this.heap = new LruHeap(); this.hheap = new LruHeap(); this.path = new Int32Array(1024);
      this.evictedBlocks = 0; this.pinnedB = 0; // blocks of the pinned pieces (each counted once)
    }
    allocNs(trace) {
      const TR = this.TR, np = TR.trPieces[trace], p0 = TR.tr_pc0[trace];
      const base = this.n; this.n += np;
      if (this.n > this.tier.length) {
        let L = this.tier.length; while (L < this.n) L *= 2;
        const g = (A, C) => { const B = new C(L); B.set(A); return B; };
        this.tier = g(this.tier, Uint8Array); this.t = g(this.t, Float64Array); this.pk = g(this.pk, Int32Array); this.len = g(this.len, Int32Array); this.dep = g(this.dep, Int32Array); this.pin = g(this.pin, Int32Array);
      }
      for (let i = 0; i < np; i++) {
        const q = p0 + i; const par = TR.pc_parent[q];
        this.pk[base + i] = par < 0 ? -1 : base + (par - p0); this.len[base + i] = TR.pc_len[q]; this.dep[base + i] = TR.pdepth[q];
      }
      return base;
    }
    walk(ns, leafLocal) { // fill this.path leaf -> root, return length
      let k = ns + leafLocal, n = 0;
      while (k >= 0) { if (n === this.path.length) { const B = new Int32Array(n * 2); B.set(this.path); this.path = B; } this.path[n++] = k; k = this.pk[k]; }
      return n;
    }
    hit(n) { // resident prefix from root: device blocks, then host DRAM and SSD blocks (any order) up to the first miss
      let dev = 0, host = 0, ssd = 0, i = n - 1;
      for (; i >= 0; i--) { const k = this.path[i]; if (this.tier[k] !== 1) break; dev += this.len[k]; }
      for (; i >= 0; i--) { const k = this.path[i]; if (this.tier[k] === 0) break; if (this.tier[k] === 3) ssd += this.len[k]; else host += this.len[k]; }
      this.hDev = dev; this.hHost = host; this.hSsd = ssd; return dev + host + ssd;
    }
    touch(n, now) {
      for (let i = 0; i < n; i++) {
        const k = this.path[i];
        if (this.tier[k] === 0) this.used += this.len[k];
        else if (this.tier[k] === 2) { this.hused -= this.len[k]; this.used += this.len[k]; }
        else if (this.tier[k] === 3) { this.sused -= this.len[k]; this.used += this.len[k]; }
        this.tier[k] = 1; this.t[k] = now; this.heap.push(now, this.dep[k], k);
      }
      this.evict();
    }
    refresh(n, now) { // LRU-refresh the resident part of the path (both tiers); it does not pin
      for (let i = 0; i < n; i++) {
        const k = this.path[i];
        if (this.tier[k] === 1) { this.t[k] = now; this.heap.push(now, this.dep[k], k); }
        else if (this.tier[k] === 2) { this.t[k] = now; this.hheap.push(now, this.dep[k], k); }
        else if (this.tier[k] === 3) { this.t[k] = now; this.sheap.push(now, this.dep[k], k); }
      }
    }
    // pin the device-resident pieces from the root that overlap the first `blocks` blocks of the path (the cached
    // prefix an in-progress request keeps reading); pinned pieces are skipped by evict() until unpinned.
    // Returns the pinned keys; this.pinB = blocks of the prefix they cover.
    pinPrefix(n, blocks) {
      const keys = []; let acc = 0;
      for (let i = n - 1; i >= 0 && acc < blocks; i--) { const k = this.path[i]; if (this.tier[k] !== 1) break; if (this.pin[k]++ === 0) this.pinnedB += this.len[k]; keys.push(k); acc += this.len[k]; }
      this.pinB = Math.min(acc, blocks); return keys;
    }
    // decode: put every piece of the path on the device and pin it (host DRAM / SSD pieces are read back, missing ones
    // arrive by migration from prefill); this.got = [device, host, SSD] blocks of it that were already cached.
    pinAll(n, now) {
      const keys = []; let dev = 0, host = 0, ssd = 0;
      for (let i = 0; i < n; i++) {
        const k = this.path[i], L = this.len[k], tr = this.tier[k];
        if (tr === 1) dev += L;
        else { this.used += L; if (tr === 2) { host += L; this.hused -= L; } else if (tr === 3) { ssd += L; this.sused -= L; } }
        this.tier[k] = 1; this.t[k] = now; this.heap.push(now, this.dep[k], k);
        if (this.pin[k]++ === 0) this.pinnedB += L;
        keys.push(k);
      }
      this.got = [dev, host, ssd]; this.evict(); return keys;
    }
    // blocks of the path not pinned yet: what pinning it would add to the pinned total
    unpinnedB(n) { let b = 0; for (let i = 0; i < n; i++) { const k = this.path[i]; if (this.pin[k] === 0) b += this.len[k]; } return b; }
    unpin(keys) { for (const k of keys) if (--this.pin[k] === 0) { this.pinnedB -= this.len[k]; if (this.tier[k] === 1) this.heap.push(this.t[k], this.dep[k], k); } }
    // space held by KV that is being written but not yet inserted (paging: pages written in place)
    reserve(blocks) { this.used += blocks; this.evict(); }
    unreserve(blocks) { this.used -= blocks; }
    evict() {
      while (this.used > this.cap && this.heap.n > 0) {
        const k = this.heap.pop(); const t = this.heap.ot;
        if (this.tier[k] !== 1 || this.t[k] !== t || this.pin[k] > 0) continue; // pinned: re-pushed by unpin()
        this.used -= this.len[k]; this.evictedBlocks += this.len[k];
        if (this.hcap > 0) this.down(k, t, 1, 2);
        else if (this.scap > 0) this.down(k, t, 1, 3);
        else this.tier[k] = 0;
      }
      while (this.hused > this.hcap && this.hheap.n > 0) {
        const k = this.hheap.pop(); const t = this.hheap.ot;
        if (this.tier[k] !== 2 || this.t[k] !== t) continue;
        this.hused -= this.len[k];
        if (this.scap > 0) this.down(k, t, 2, 3); else this.tier[k] = 0;
      }
      while (this.sused > this.scap && this.sheap.n > 0) {
        const k = this.sheap.pop(); const t = this.sheap.ot;
        if (this.tier[k] !== 3 || this.t[k] !== t) continue;
        this.sused -= this.len[k]; this.tier[k] = 0;
      }
      if (this.heap.n > 4 * (this.n + 1024)) this.rebuild(this.heap, 1);
      if (this.hheap.n > 4 * (this.n + 1024)) this.rebuild(this.hheap, 2);
      if (this.sheap.n > 4 * (this.n + 1024)) this.rebuild(this.sheap, 3);
    }
    down(k, t, from, to) { // move piece k one or two tiers down, keeping its LRU time
      this.tier[k] = to;
      if (to === 2) { this.hused += this.len[k]; this.hheap.push(t, this.dep[k], k); } else { this.sused += this.len[k]; this.sheap.push(t, this.dep[k], k); }
      if (this.onDemote) this.onDemote(this.len[k], from, to);
    }
    rebuild(h, tier) {
      const t = h.t.slice(0, h.n), k = h.k.slice(0, h.n); h.n = 0;
      for (let i = 0; i < t.length; i++) if (this.tier[k[i]] === tier && this.t[k[i]] === t[i]) h.push(t[i], this.dep[k[i]], k[i]);
    }
  }

  // Static slots: a slot holds one stream's latest request; LRU over idle slots; busy slots cannot be evicted.
  // onEvict(streamKey, blocks) is called with the evicted stream's KV (the SSD tier writes it back) and returns when
  // that write-back ends; evictEnd holds it for the new occupant, which cannot overwrite the slot before then.
  class SlotCache {
    constructor(N) { this.N = N; this.owner = new Int32Array(N).fill(-1); this.last = new Float64Array(N); this.busy = new Uint8Array(N); this.blocks = new Float64Array(N); this.of = new Map(); this.evictions = 0; }
    acquire(streamKey, now) { // -> {slot, warm} or null
      this.evictEnd = 0;
      let s = this.of.get(streamKey);
      if (s !== undefined) { if (this.busy[s]) return null; this.busy[s] = 1; return { slot: s, warm: true }; }
      let pick = -1, lru = Infinity;
      for (let i = 0; i < this.N; i++) {
        if (this.busy[i]) continue;
        if (this.owner[i] < 0) { pick = i; break; }
        if (this.last[i] < lru) { lru = this.last[i]; pick = i; }
      }
      if (pick < 0) return null;
      if (this.owner[pick] >= 0) { this.of.delete(this.owner[pick]); this.evictions++; if (this.onEvict) this.evictEnd = this.onEvict(this.owner[pick], this.blocks[pick]); }
      this.owner[pick] = streamKey; this.of.set(streamKey, pick); this.busy[pick] = 1;
      return { slot: pick, warm: false };
    }
    release(s, now, blocks) { this.busy[s] = 0; this.last[s] = now; this.blocks[s] = blocks; }
  }

  // ------------------------------------------------------------------------------------------------------
  // The replay simulation
  // ------------------------------------------------------------------------------------------------------
  const EV_READY = 1, EV_DONE = 2, EV_END = 3, EV_PUMP = 4, EV_LANE = 5, EV_FETCHED = 6, EV_SEG = 7, EV_DEC = 8, EV_DSTART = 9;

  function simulate(TR, cal, cfgIn, opts) {
    opts = opts || {};
    const plan = makePlan(cfgIn, cal);
    const cfg = plan.cfg;
    if (plan.errors.some((e) => !e.startsWith('no profile'))) return { plan, error: plan.errors.join('; ') };
    const rnd = mulberry32(cfg.seed * 2654435761 >>> 0);
    const S = plan.S, B = 64;
    const ev = new EvHeap();
    const rr = cfg.policy === 'rr';
    const rrPool = rr && cfg.cache === 'pool'; // a lane per turn, the partial KV copied out to the pool every turn
    // in-progress requests whose KV lives in the pool (round robin on the pool, or paging): their cached prefix is
    // pinned and their new KV reserved, and admission is bounded by the pool
    const pinHits = rrPool || (rr && cfg.cache === 'paging');
    let now = 0, t0 = null, tEnd = Infinity, primersLeft = 0, warmDone = false;
    let inflightReqs = 0; // queued + prefill + decode (system idle check)
    // replicas
    const reps = [];
    for (let r = 0; r < cfg.replicas; r++) {
      reps.push({
        id: r, free: new Float64Array(S), busy: new Float64Array(S), queue: [], active: [], exits: [], pumpAt: -1,
        trees: 0, lanesUsed: 0, arenaUsed: 0, progTok: 0, nProg: 0,
        // round robin on fixed lanes: which request's context each lane holds, segments using it, last use
        laneTab: rrPool && !cfg.laneArena ? Array.from({ length: plan.lanes }, () => ({ owner: null, busy: 0, t: -1 })) : null,
        pool: cfg.cache === 'slots' ? null : new PoolCache(TR, plan.poolTok / B, plan.hostTok / B, plan.ssdTok / B),
        slots: cfg.cache === 'slots' ? new SlotCache(plan.nSlots) : null, h2dFree: 0, d2hFree: 0, ssdFree: 0,
      });
    }
    // offload links per replica, each a FIFO resource: PCIe host -> device (fetches), PCIe device -> host
    // (write-backs; full duplex, so the two do not wait on each other) and the SSDs (reads and writes share them).
    // linkBusy[] counts their busy seconds that start inside the window.
    const offBytes = (blocks) => blocks * B * M3.L * plan.kvbHost;
    const offload = plan.hostTok + plan.ssdTok > 0;
    function useLink(rp, key, bytes, bps, notBefore) {
      if (!(bytes > 0)) return Math.max(now, notBefore || 0);
      const start = Math.max(now, rp[key], notBefore || 0), sec = bytes / bps;
      rp[key] = start + sec;
      if (warmDone && start >= t0 && start <= tEnd) st[key + 'Busy'] += sec;
      return rp[key];
    }
    // read `host` blocks from host DRAM and `ssd` blocks from SSD onto the device; returns when both have landed
    // (an SSD read streams into host DRAM and on over PCIe, so the two overlap)
    function offFetch(rp, host, ssd) {
      const tS = useLink(rp, 'ssdFree', offBytes(ssd), plan.ssdRdBps);
      const tP = useLink(rp, 'h2dFree', offBytes(host + ssd), plan.pcieBps);
      if (warmDone) { st.hostTok += host * B; st.ssdTok += ssd * B; }
      return Math.max(tS, tP);
    }
    // a demotion: device -> host is a PCIe write-back, host -> SSD a drive write, device -> SSD both in turn
    function offDemote(rp, blocks, from, to) {
      const b = offBytes(blocks);
      const t = from === 1 ? useLink(rp, 'd2hFree', b, plan.pcieBps) : now;
      if (to === 3) useLink(rp, 'ssdFree', b, plan.ssdWrBps, t);
      return t;
    }
    for (const rp of reps) if (rp.pool && offload) rp.pool.onDemote = (blocks, from, to) => { offDemote(rp, blocks, from, to); };
    // Offload tiers behind static slots: an evicted slot's KV (its stream's latest request), one copy per stream, in
    // host DRAM (LRU over hostTok) overflowing to SSD (LRU over ssdTok), no prefix sharing. demote(blocks, from, to)
    // moves the bytes and returns when the device write-back ends; evict() returns that time (the slot's new occupant
    // waits for it).
    function slotTiers(hostTok, ssdTok, demote) {
      const o = { host: new Map(), hostUsed: 0, ssd: new Map(), ssdUsed: 0 }; // streamKey -> blocks, in LRU order
      o.has = (key) => o.host.has(key) || o.ssd.has(key);
      o.take = (key) => { // remove the stream's copy -> { inHost, stored }
        const inHost = o.host.has(key), m = inHost ? o.host : o.ssd, stored = m.get(key);
        m.delete(key); if (inHost) o.hostUsed -= stored; else o.ssdUsed -= stored;
        return { inHost, stored };
      };
      o.evict = (key, blocks) => {
        if (!(blocks > 0)) return 0;
        const toHost = hostTok > 0;
        const t = demote(blocks, 1, toHost ? 2 : 3);
        if (toHost) { o.host.set(key, blocks); o.hostUsed += blocks; } else { o.ssd.set(key, blocks); o.ssdUsed += blocks; }
        for (const [k, b] of o.host) {
          if (o.hostUsed * B <= hostTok) break;
          o.host.delete(k); o.hostUsed -= b;
          if (ssdTok > 0) { demote(b, 2, 3); o.ssd.set(k, b); o.ssdUsed += b; }
        }
        for (const [k, b] of o.ssd) { if (o.ssdUsed * B <= ssdTok) break; o.ssd.delete(k); o.ssdUsed -= b; }
        return t;
      };
      return o;
    }
    // prefill: the stream's next request reads back the prefix it shares with its slot's evicted KV
    for (const rp of reps) if (rp.slots && offload) {
      rp.off = slotTiers(plan.hostTok, plan.ssdTok, (blocks, from, to) => offDemote(rp, blocks, from, to));
      rp.slots.onEvict = rp.off.evict;
    }
    const slotKey = (q) => q.tree.key * 4096 + (TR.req_stream[q.r] - q.tree.s0);
    const offHas = (rep, key) => !!rep.off && rep.off.has(key);
    // read a stream's KV back from host DRAM or SSD (the prefix the request shares with it); returns the fetch's end
    function slotFetch(q, rep, key) {
      const { inHost, stored } = rep.off.take(key);
      const blocks = Math.min(stored, TR.req_lcp_prev[q.r]);
      q.ssdBlocks = blocks;
      return inHost ? offFetch(rep, blocks, 0) : offFetch(rep, 0, blocks);
    }
    // stats
    const st = { done: 0, useful: 0, newTok: 0, processed: 0, hitTok: 0, inTok: 0, infHitTok: 0, hostTok: 0, ssdTok: 0, reprefill: 0, alignLoss: 0,
      h2dFreeBusy: 0, d2hFreeBusy: 0, ssdFreeBusy: 0,
      ttft: [], chunks: 0, segs: 0, laneWait: 0, reqs: 0, primers: 0, warmupS: 0, idleWarps: 0, primerTok: 0,
      gated: 0, gateWait: 0, legacyStarts: 0, skippedTraces: 0, rrSegs: 0, rrReuse: 0, rrCopyIn: 0,
      outTok: 0, decArea: 0, decingArea: 0, decMax: 0, decWaits: 0, decWait: 0, decWaited: 0, decTokArea: 0, ringFull: 0, pfSlot: 0, pfNone: 0, sendBlock: 0,
      decCtxTok: 0, decHitTok: 0, decHostTok: 0, decSsdTok: 0, dH2dBusy: 0, dD2hBusy: 0, dSsdBusy: 0, decPoolArea: 0,
      decStarts: 0, decStartDelay: 0, decLaneWaited: 0, parked: 0, parkArea: 0, tsu: [] };
    const decWin = () => warmDone && now >= t0 && now <= tEnd;
    // ---------- decode KV (DEFAULTS.decodeCache): reserved when a request is admitted to prefill (TTFT clock already
    // running), released when its decode ends, then cached: released slots keep their stream's KV, released pages
    // stay in the pool (LRU), and with decode tiers both go on to the decode galaxies' host DRAM and SSD
    const decGlobal = cfg.decodeCache === 'hybrid' && cfg.decodeLaneScope === 'global';
    const decStage = cfg.decodeCache === 'hybrid' && !decGlobal;
    const decLinks = { dH2d: 0, dD2h: 0, dSsd: 0 }; // the decode galaxies' PCIe (one per direction) and SSDs
    function decFetch(host, ssd) { // as offFetch, on the decode side
      const tS = useLink(decLinks, 'dSsd', offBytes(ssd), plan.decSsdRdBps);
      const tP = useLink(decLinks, 'dH2d', offBytes(host + ssd), plan.decPcieBps);
      if (decWin()) { st.decHostTok += host * B; st.decSsdTok += ssd * B; }
      return Math.max(tS, tP);
    }
    function decDemote(blocks, from, to) { // as offDemote, on the decode side
      const b = offBytes(blocks);
      const t = from === 1 ? useLink(decLinks, 'dD2h', b, plan.decPcieBps) : now;
      if (to === 3) useLink(decLinks, 'dSsd', b, plan.decSsdWrBps, t);
      return t;
    }
    const decOffload = plan.decHostTok + plan.decSsdTok > 0;
    const decSlots = cfg.decodeCache === 'slots' && cfg.decodeSlots > 0 ? new SlotCache(cfg.decodeSlots) : null;
    const decOff = decSlots && decOffload ? slotTiers(plan.decHostTok, plan.decSsdTok, decDemote) : null;
    if (decOff) decSlots.onEvict = decOff.evict;
    const decPool = cfg.decodeCache !== 'slots' ? new PoolCache(TR, plan.decPoolTok / B, plan.decHostTok / B, plan.decSsdTok / B) : null;
    if (decPool && decOffload) decPool.onDemote = (blocks, from, to) => { decDemote(blocks, from, to); };
    // decResvB: pool blocks reserved for the output of sessions not yet in a lane; global lanes in use, and the
    // sessions past prefill waiting for one (FIFO); pendTtft: requests counted in the window whose decode has not begun
    let decResvB = 0, decLanesUsed = 0; const decLaneQ = [], pendTtft = new Set();
    // queue backpressure: requests admitted without a slot (KV parked on the decode SSDs), and those of them past
    // prefill waiting for a slot (FIFO)
    const decQueue = cfg.decodeBackpressure === 'queue' && !!decSlots;
    let parkedN = 0; const slotQ = [];
    // decode slots: held from admission to prefill (TTFT clock already running) until decode ends; time-weighted
    // occupancy over the window, for slots held and for requests actually decoding (prefill done, not yet ended)
    let decHeld = 0, decoding = 0, decT = 0, decSpeedSum = 0; const decQ = [];
    // share of its full speed each decoding session gets: min(1, decodeStages / sessions decoding)
    const decShare = () => (cfg.decodeStages > 0 && decoding > cfg.decodeStages ? cfg.decodeStages / decoding : 1);
    function decTick() {
      if (warmDone) {
        const a = Math.max(decT, t0), b = Math.min(now, tEnd);
        if (b > a) {
          st.decArea += decHeld * (b - a); st.decingArea += decoding * (b - a); st.decTokArea += decSpeedSum * decShare() * (b - a);
          if (cfg.decodeStages > 0 && decoding >= cfg.decodeStages) st.ringFull += b - a;
          if (decPool) st.decPoolArea += (decPool.pinnedB + decResvB) * B * (b - a);
          st.parkArea += parkedN * (b - a);
        }
      }
      decT = now;
    }
    // processor-sharing ring (decodeStages > 0): every session advances by the same seconds of full-speed decode,
    // decV, at decShare() per second; a session started at decV = v needing w seconds at its full speed (output
    // tokens / its tokens/s/u) finishes when decV reaches v + w. ring is a min-heap on that threshold; one EV_DEC
    // event (tagged with decGen) is pending for the earliest finish.
    let decV = 0, decVT = 0, decGen = 0; const ring = [];
    const ringPush = (x) => { ring.push(x); for (let i = ring.length - 1; i > 0;) { const p = (i - 1) >> 1; if (ring[p].thr <= ring[i].thr) break; [ring[p], ring[i]] = [ring[i], ring[p]]; i = p; } };
    const ringPop = () => {
      const top = ring[0], last = ring.pop();
      if (ring.length) { ring[0] = last; for (let i = 0; ;) { const l = 2 * i + 1, r = l + 1; let m = i; if (l < ring.length && ring[l].thr < ring[m].thr) m = l; if (r < ring.length && ring[r].thr < ring[m].thr) m = r; if (m === i) break; [ring[m], ring[i]] = [ring[i], ring[m]]; i = m; } }
      return top;
    };
    function decAdvance() { if (decoding > 0) decV += decShare() * (now - decVT); decVT = now; }
    function decSchedule() { decGen++; if (ring.length) ev.push(now + Math.max(0, ring[0].thr - decV) / decShare(), { e: EV_DEC, gen: decGen }); }
    function onDecodeEvent(gen) {
      if (gen !== decGen) return; // superseded by a later change in the number decoding
      decAdvance();
      while (ring.length && ring[0].thr <= decV + 1e-9) { const { q } = ringPop(); onEnd(q); decAdvance(); }
      decSchedule();
    }
    // per-stage decode lanes: once more sessions decode than there are lanes per stage, every token copies its
    // session's context in on every stage (decCopySpeed), until they are back within the lanes. Re-rates the ring:
    // each session keeps its tokens left, at its new speed (call after decAdvance, with `decoding` updated).
    let decCopying = false;
    function decRerate() {
      const c = decStage && decoding > plan.decLanes;
      if (c === decCopying) return;
      decCopying = c; decSpeedSum = 0;
      for (const e of ring) {
        const s = c ? e.q.decCopySpeed : e.q.decBase;
        e.thr = decV + (e.thr - decV) * e.q.decSpeed / s; e.q.decSpeed = s; decSpeedSum += s;
      }
      ring.sort((a, b) => a.thr - b.thr); // a sorted array is a heap
    }
    // reserve the request's decode KV at admission; false if decode has no room for it now. q.decReadyAt: when its
    // cached KV is back from host DRAM / SSD (and a reclaimed slot's write-back is done); decode cannot start before.
    // a free decode slot for q -> { hitB, ready } (blocks of its context the slot already holds or that come back
    // from the decode tiers, and when they have landed), or null
    function slotAcquire(q) {
      const key = slotKey(q), a = decSlots.acquire(key, now); if (!a) return null;
      let hitB = 0, ready = decSlots.evictEnd;
      q.decSlotIdx = a.slot;
      if (a.warm) hitB = TR.req_lcp_prev[q.r];
      else if (decOff && decOff.has(key)) {
        const { inHost, stored } = decOff.take(key);
        hitB = Math.min(stored, TR.req_lcp_prev[q.r]);
        ready = Math.max(ready, inHost ? decFetch(hitB, 0) : decFetch(0, hitB));
      }
      return { hitB, ready };
    }
    function decTryTake(q) {
      decTick();
      let hitB = 0, ready = 0;
      if (decSlots) {
        if (slotQ.length) return false; // parked sessions past prefill get the next free slots
        const a = slotAcquire(q); if (!a) return false;
        hitB = a.hitB; ready = a.ready;
      } else if (decPool) { // the whole context pinned in the pool (shared pieces once), plus the output's pages
        const pool = decPool, n = pool.walk(q.tree.dns, TR.req_leaf[q.r] - TR.tr_pc0[q.tree.trace]), outB = Math.ceil(TR.req_out[q.r] / B);
        const held = pool.pinnedB + decResvB;
        if (held > 0 && held + pool.unpinnedB(n) + outB > pool.cap) return false;
        q.decPins = pool.pinAll(n, now); q.decOutB = outB; decResvB += outB; pool.reserve(outB);
        const [dev, host, ssd] = pool.got; hitB = dev + host + ssd;
        if (host + ssd > 0) ready = decFetch(host, ssd);
      }
      q.decReadyAt = ready;
      if (decWin() && (decSlots || decPool)) { st.decCtxTok += TR.req_blocks[q.r] * B; st.decHitTok += hitB * B; }
      return true;
    }
    function decTake(q) {
      decTick(); decHeld++; q.decSlot = true;
      if (warmDone && now >= t0 && now <= tEnd && decHeld > st.decMax) st.decMax = decHeld;
    }
    // decode's side of admission to prefill: decode KV reserved (decTake), or with queue backpressure the KV parked
    // on the decode SSDs; false = the request waits
    function decAdmit(q) {
      if (decTryTake(q)) { decTake(q); return true; }
      if (!decQueue || (cfg.decodeQueueMax > 0 && parkedN >= cfg.decodeQueueMax)) return false;
      decTick(); parkedN++; q.parked = true; q.decSlot = true; q.decReadyAt = 0;
      decDemote(TR.req_blocks[q.r], 1, 3); // migrated in, written out to SSD
      if (decWin()) { st.parked++; st.decCtxTok += TR.req_blocks[q.r] * B; }
      return true;
    }
    // a parked session past prefill takes a free slot and reads back from SSD what the slot does not already hold;
    // false if no slot is free
    function decUnpark(q) {
      const a = slotAcquire(q); if (!a) return false;
      if (slotQ[0] === q) slotQ.shift();
      decTick(); parkedN--; q.parked = false; decHeld++;
      if (decWin() && decHeld > st.decMax) st.decMax = decHeld;
      q.decReadyAt = Math.max(a.ready, decFetch(0, TR.req_blocks[q.r] - a.hitB));
      decStep(q); decDrain();
      return true;
    }
    // admit waiting requests (FIFO) while decode has room
    function decDrain() {
      while (decQ.length && decAdmit(decQ[0])) {
        const w = decQ.shift();
        if (warmDone && now >= t0 && now <= tEnd) { st.decWaited++; st.decWait += now - w.decWaitAt; }
        admit(w);
      }
    }
    // a primer (max_tokens = 1) leaves its KV on the decode side like any finished request
    function decWarm(q) {
      if (decSlots) {
        const key = slotKey(q), a = decSlots.acquire(key, now);
        if (a) { if (!a.warm && decOff && decOff.has(key)) decOff.take(key); decSlots.release(a.slot, now, TR.req_blocks[q.r]); }
      } else if (decPool) decPool.touch(decPool.walk(q.tree.dns, TR.req_leaf[q.r] - TR.tr_pc0[q.tree.trace]), now);
    }
    // global lane: the session's context is copied pool -> lane (every stage in parallel), then its pages and output
    // reservation leave the pinned set (they stay cached), which may let waiting requests in
    function decLaneTake(q) {
      decTick(); decLanesUsed++; q.decLane = true;
      if (q.decPins) { decPool.unpin(q.decPins); q.decPins = null; }
      if (q.decOutB) { decPool.unreserve(q.decOutB); decResvB -= q.decOutB; q.decOutB = 0; }
      q.decReadyAt = now + plan.decLaneCopyS(TR.req_blocks[q.r] * B);
      decStep(q);
      decDrain();
    }
    // decode ends: free the slot / pages / lane (the KV stays cached) and hand the lane to the next waiting session
    function decRelease(q) {
      if (decSlots) decSlots.release(q.decSlotIdx, now, TR.req_blocks[q.r]);
      while (slotQ.length && decUnpark(slotQ[0])) { /* the freed slot goes to the oldest parked session */ }
      if (q.decPins) { decPool.unpin(q.decPins); q.decPins = null; }
      if (q.decOutB) { decPool.unreserve(q.decOutB); decResvB -= q.decOutB; q.decOutB = 0; }
      if (q.decLane) {
        decLanesUsed--; q.decLane = false;
        while (decLaneQ.length && decLanesUsed < plan.decLanes) decLaneTake(decLaneQ.shift());
      }
    }
    let nextTrace = 0; const nsKey = { v: 0 };
    const scratch = new Float64Array(S);

    // ---------- trees (one per concurrency lane)
    // A tree replays one trace: the root chain plus every child stream it spawns. Per tree:
    //   ended[local req]  requests that have ENDed (or that precede t* and so count as completed history)
    //   waiters           request -> barrier-gated requests waiting for it to END
    //   pend / waiting    join counters and root turns waiting on them
    function newTree(lane, trace, tstar, now) {
      const rep = reps.reduce((a, b) => (b.trees < a.trees ? b : a), reps[0]);
      rep.trees++;
      const tree = { lane, trace, rep, ns: rep.pool ? rep.pool.allocNs(trace) : 0, dns: decPool ? decPool.allocNs(trace) : 0, key: nsKey.v++, live: 0, waiting: new Map(), pend: new Map(),
        r0: TR.tr_req0[trace], ended: TR.hasPred ? new Uint8Array(TR.trReqs[trace]) : null, waiters: new Map() };
      const s0 = TR.tr_st0[trace], ns = TR.trStreams[trace];
      tree.s0 = s0;
      const mainFirst = TR.st_first[s0], mainCnt = TR.st_cnt[s0];
      if (tstar === null) { // recycled: start at turn 0 with fresh join counters
        initPend(tree);
        dispatch(tree, mainFirst, now);
        return tree;
      }
      // initial tree: AIPerf TrajectorySource._snapshot_for / _replay_resume_boundaries at t* (ai-dynamo/aiperf @ c78644090ff,
      // timing/trajectory_source.py). A branch whose start is after t* is spawned later by its spawn turn unless that turn
      // started before t*; every stream resumes at its first request at/after t* (its previous one is the primer); every
      // request before t* is completed history for the barriers; t* after the last root start leaves a rootless lane.
      const nextAt = (first, cnt) => { for (let c = first; c < first + cnt; c++) if (TR.req_t[c] >= tstar) return c; return -1; };
      const seed = (from, to) => { if (tree.ended) for (let r = from; r < to; r++) tree.ended[r - tree.r0] = 1; };
      const rootNext = nextAt(mainFirst, mainCnt), rootIdx = rootNext >= 0 ? rootNext - mainFirst : -1;
      const prof = [], primers = [], live = new Set(); // prof: [req, offset, isRoot]
      for (let s = s0 + 1; s < s0 + ns; s++) {
        const sp = TR.st_spawn[s] - mainFirst, first = TR.st_first[s], cnt = TR.st_cnt[s], j = TR.st_join[s];
        if (tstar < TR.st_t0[s]) { // branch not started at t*: spawned later unless its spawn turn already completed
          const notCompleted = tstar < TR.req_t[mainFirst + sp] || (rootIdx >= 0 && rootIdx <= sp);
          if (notCompleted) { if (j >= 0) tree.pend.set(j, (tree.pend.get(j) || 0) + 1); continue; }
        }
        const c = nextAt(first, cnt);
        if (c < 0) { seed(first, first + cnt); continue; } // no request at/after t*: completed history
        seed(first, c);
        if (c - 1 >= first) primers.push(c - 1);
        prof.push([c, Math.max(0, TR.req_t[c] - tstar), false]);
        if (j >= 0) { tree.pend.set(j, (tree.pend.get(j) || 0) + 1); live.add(j); }
      }
      if (rootNext >= 0) {
        seed(mainFirst, rootNext);
        if (rootNext - 1 >= mainFirst) primers.push(rootNext - 1);
        prof.push([rootNext, Math.max(0, TR.req_t[rootNext] - tstar), true]);
      } else seed(mainFirst, mainFirst + mainCnt); // t* after the last root turn: no root state, no root primer
      // AIPerf returns no snapshot when nothing is dispatchable (empty, or only a root gated on a join) and falls back
      // to the legacy turn-index split of the root (below)
      const dispatchable = prof.some(([r, , isRoot]) => !isRoot || !live.has(r));
      if (!dispatchable) return legacyTree(tree, trace, mainFirst, mainCnt);
      for (const r of primers) primer(tree, r);
      tree.prof = prof;
      return tree;
    }
    // AIPerf _build_trajectory_for_lane fallback (timestamp-less split): warm root turn k, resume at k+1 at once.
    // Unreachable with tr_dur = last recorded start (t* < that start, so some stream is always live at t*). AIPerf seeds
    // only root turns 0..k as completed here; this treats every request that is not replayed as completed history.
    function legacyTree(tree, trace, mainFirst, mainCnt) {
      const n = mainCnt;
      if (n <= 1) { st.skippedTraces++; tree.rep.trees--; return null; } // unspawnable: the sampler draws the next trace
      st.legacyStarts++;
      const kMin = Math.min(Math.floor(cfg.startMin * n), n - 2), kMax = Math.max(kMin, Math.min(Math.floor(cfg.startMax * n), n - 2));
      const k = kMin + Math.floor(rnd() * (kMax - kMin + 1));
      tree.pend = new Map(); tree.waiting = new Map();
      if (tree.ended) { tree.ended.fill(1); for (let r = mainFirst + k + 1; r < mainFirst + mainCnt; r++) tree.ended[r - tree.r0] = 0; }
      const s0 = tree.s0, ns = TR.trStreams[trace];
      for (let s = s0 + 1; s < s0 + ns; s++) {
        if (TR.st_spawn[s] - mainFirst <= k) continue; // spawned by an already-played turn: not replayed
        const j = TR.st_join[s]; if (j >= 0) tree.pend.set(j, (tree.pend.get(j) || 0) + 1);
        if (tree.ended) for (let r = TR.st_first[s]; r < TR.st_first[s] + TR.st_cnt[s]; r++) tree.ended[r - tree.r0] = 0;
      }
      primer(tree, mainFirst + k);
      tree.prof = [[mainFirst + k + 1, 0, true]];
      return tree;
    }
    function primer(tree, r) {
      primersLeft++; st.primers++;
      const q = mkReq(tree, r, true); tree.live++; ev.push(0, { e: EV_READY, q });
    }
    function mkReq(tree, r, isPrimer) {
      return { r, tree, primer: isPrimer, tReady: 0, tStart: -1, tDone: 0, hit: 0, host: 0, rem: 0, pos: 0, lane: -1, slot: -1, started: false, fetchedAt: 0, laneIdx: -1 };
    }
    function dispatch(tree, r, t) {
      const q = mkReq(tree, r, false); tree.live++; ev.push(t, { e: EV_READY, q });
    }
    function startProfiling(t) {
      t0 = t; tEnd = t + cfg.duration; warmDone = true; st.warmupS = t;
      for (const tr of trees) {
        if (!tr || !tr.prof) continue;
        for (const [r, off, isMain] of tr.prof) {
          if (isMain && (tr.pend.get(r) || 0) > 0) { tr.waiting.set(r, true); continue; }
          dispatch(tr, r, t + off);
        }
        tr.prof = null;
        if (tr.live === 0) recycle(tr, t);
      }
    }
    function recycle(tree, t) {
      tree.rep.trees--;
      const trace = nextTrace++ % TR.nT;
      trees[tree.lane] = newTree(tree.lane, trace, null, t);
    }
    // ---------- request lifecycle
    // AIPerf ReplayBarrierCoordinator (timing/replay_dependencies.py @ c78644090ff; edges from
    // infer_cross_stream_predecessors per replay scope, see prep_traffic.py): a due request is retained until its
    // recorded cross-stream predecessors in the same tree have ENDed (history before t* counts as ended); it is issued
    // the moment the last one ends. Primers bypass it (AIPerf activates barriers after cache priming). Retained requests
    // hold their tree (tree.live) but are not in flight (idle warp) until issued; TTFT counts from the issue.
    function onReady(q) {
      const tree = q.tree, r = q.r;
      if (!q.primer && tree.ended) {
        let need = 0;
        for (let i = TR.pred_head[r], e = TR.pred_head[r + 1]; i < e; i++) {
          const p = TR.pred_list[i];
          if (tree.ended[p - tree.r0]) continue;
          need++;
          const w = tree.waiters.get(p); if (w) w.push(q); else tree.waiters.set(p, [q]);
        }
        if (need > 0) { q.need = need; q.gatedAt = now; if (warmDone) st.gated++; return; }
      }
      issue(q);
    }
    function issue(q) {
      const tree = q.tree, rep = tree.rep, r = q.r;
      q.tReady = now; inflightReqs++;
      // overlap-path subagents are dispatched when the spawning main turn is issued
      if (!q.primer && TR.req_stream[r] === tree.s0) {
        for (let s = TR.spawnHead[r]; s >= 0; s = TR.spawnNext[s]) if (TR.st_ovl[s]) dispatch(tree, TR.st_first[s], now + TR.st_off[s]);
      }
      if (!q.primer) { // decode KV first (FIFO behind requests already waiting for it)
        if (decQ.length || !decAdmit(q)) { q.decWaitAt = now; decQ.push(q); return; }
      }
      admit(q);
    }
    // the request may start: look up its prefix (fetching host / SSD parts first), then queue it for prefill
    function admit(q) {
      const tree = q.tree, rep = tree.rep, r = q.r;
      if (rep.pool) {
        const pool = rep.pool; const n = pool.walk(tree.ns, TR.req_leaf[r] - TR.tr_pc0[tree.trace]);
        pool.hit(n);
        const dev = pool.hDev, host = pool.hHost, ssd = pool.hSsd;
        pool.refresh(n, now); // LRU-refresh the resident part of the path; tryStart re-checks what is left
        q.hitRaw = dev + host + ssd; q.host = host + ssd;
        if (host + ssd > 0 && offload) { // fetch the host DRAM / SSD part before it can be admitted
          const t = offFetch(rep, host, ssd); q.fetchedAt = t;
          ev.push(t, { e: EV_FETCHED, q }); return;
        }
      }
      if (rep.off && !q.primer) { // static slots: the stream's KV is in host DRAM / on SSD, not in a slot -> read it back
        const key = slotKey(q);
        if (!rep.slots.of.has(key) && offHas(rep, key)) { const t = slotFetch(q, rep, key); q.fetchedAt = t; ev.push(t, { e: EV_FETCHED, q }); return; }
      }
      enqueue(q);
    }
    function enqueue(q) {
      const rep = q.tree.rep;
      rep.queue.push(q);
      pump(rep);
    }
    function hitTokens(q, rawBlocks) {
      let h = rawBlocks * B;
      const tot = TR.req_blocks[q.r] * B;
      if (h >= tot) h = tot - B; // always recompute the last block (logits)
      if (cfg.reqPad === 'chunk' && !cfg.unaligned) { const a = Math.floor(h / cfg.chunk) * cfg.chunk; st.alignLoss += q.primer || !warmDone ? 0 : h - a; h = a; }
      return Math.max(0, h);
    }
    function tryStart(q, rep) { // acquire lane/slot, compute hit; false if blocked
      const tree = q.tree, r = q.r;
      if (rep.slots) {
        const streamKey = slotKey(q);
        const a = rep.slots.acquire(streamKey, now); if (!a) return false;
        q.slot = a.slot; q.laneCap = cfg.slotLen;
        if (rep.slots.evictEnd > (q.readyAt || 0)) q.readyAt = rep.slots.evictEnd; // previous occupant still writing back
        // SSD tier: read back at READY, or now if the slot was evicted while the request was queued
        if (!a.warm && !q.ssdBlocks && offHas(rep, streamKey)) q.readyAt = Math.max(q.readyAt || 0, slotFetch(q, rep, streamKey));
        const lp = a.warm ? TR.req_lcp_prev[r] : q.ssdBlocks || 0;
        q.hitTok = hitTokens(q, lp);
      } else {
        const need = TR.req_blocks[r] * B;
        if (pinHits) { // the pool must hold every in-progress request in full (pinned prefix + new KV)
          if (rep.nProg > 0 && rep.progTok + need > plan.poolTok) return false;
          rep.progTok += need; rep.nProg++; q.prog = need;
        }
        if (rrPool) { /* lanes are taken per turn (laneGet) */ } else {
          if (plan.lanes !== Infinity && rep.lanesUsed >= plan.lanes) return false;
          if (cfg.laneArena && cfg.cache === 'pool') { if (rep.arenaUsed + need > plan.arena) return false; rep.arenaUsed += need; q.arena = need; }
          if (plan.lanes !== Infinity) rep.lanesUsed++;
          q.lane = 1;
        }
        // dense ring-joint scans the whole lane: fixed lanes are laneLen; arena lanes / paged kernels are request-sized
        q.laneCap = cfg.cache === 'pool' && !cfg.laneArena ? cfg.laneLen : need;
        // re-check the hit: the READY-time refresh does not pin, so pages may have been evicted (lost) or demoted to
        // host (need a PCIe fetch) while the request was queued. Blocks fetched at READY are staged on device.
        const pool = rep.pool; const n = pool.walk(tree.ns, TR.req_leaf[r] - TR.tr_pc0[tree.trace]); pool.hit(n);
        const devNow = pool.hDev, offNow = pool.hHost + pool.hSsd, fetched = q.host || 0;
        const hitB = Math.min(q.hitRaw, devNow + Math.max(offNow, fetched));
        const extra = offload ? Math.max(0, hitB - devNow - fetched) : 0;
        if (extra > 0) { const h = Math.min(extra, pool.hHost); q.readyAt = offFetch(rep, h, extra - h); }
        q.hitTok = hitTokens(q, hitB);
        if (pinHits) { q.pins = pool.pinPrefix(n, q.hitTok / B); q.pinTok = Math.min(pool.pinB * B, q.hitTok); }
        if (cfg.cache === 'paging' && plan.poolTok !== Infinity) { q.resv = (TR.req_blocks[r] * B - q.hitTok) / B; pool.reserve(q.resv); }
      }
      q.started = true; q.tStart = now;
      q.pos = q.hitTok; q.rem = TR.req_blocks[r] * B - q.hitTok; q.first = true;
      q.waitS = now - (q.fetchedAt || q.tReady); // queue wait before admission; counted with the completion
      return true;
    }
    function queueOrder(rep) {
      if (cfg.policy === 'srpt' && rep.queue.length > 1) {
        const mw = cfg.srptMaxWait;
        rep.queue.sort((a, b) => {
          const aw = now - a.tReady > mw, bw = now - b.tReady > mw;
          if (aw !== bw) return aw ? -1 : 1;
          if (aw) return a.tReady - b.tReady;
          return estNew(a, rep) - estNew(b, rep);
        });
      }
    }
    function estNew(q, rep) { // expected new blocks: pool hit known at READY; slots hit if the stream still owns its slot
      if (!rep.slots) return TR.req_blocks[q.r] - (q.hitRaw || 0);
      const key = q.tree.key * 4096 + (TR.req_stream[q.r] - q.tree.s0);
      return TR.req_blocks[q.r] - (rep.slots.of.has(key) ? TR.req_lcp_prev[q.r] : 0);
    }
    // pull segments into one chunk
    function formChunk(rep) {
      const segs = []; let T = 0; const C = cfg.chunk;
      const fixed = cfg.reqPad === 'chunk';
      // chunk token budget: batching with tile padding packs up to the budget; chunk padding packs whole chunks
      const budget = !fixed ? cfg.budget : cfg.batch ? Math.max(cfg.budget, C) : C;
      // owner placement: rows per SP rank in this chunk; a rank holds at most plan.rankCap
      const own = plan.owner, load = own ? new Array(plan.sp).fill(0) : null;
      const room = () => segs.length < plan.maxR && (fixed ? budget - T >= C : own ? load.some((l) => plan.rankCap - l >= 32) : budget - T >= plan.gran); // space for one more segment
      const cands = rep.active;
      // owner placement: every rank is padded to the busiest one, so the chunk costs SP x its rows
      const done = () => ({ segs, T: own ? plan.sp * Math.max(...load) : T, load });
      // one segment per request per chunk (a request's tokens in a chunk form one attention call, whatever the
      // padding): chunk padding = a whole number of chunks, tile padding = any multiple of 32*SP tokens (32, owner)
      const take = (q, maxTok) => {
        let n, npad;
        if (fixed) { const units = Math.floor(maxTok / C); if (units < 1) return false; n = Math.min(q.rem, units * C); npad = Math.ceil(n / C) * C; }
        else { const cap = maxTok; if (cap < plan.gran) return false; n = Math.min(q.rem, cap - (cap % plan.gran)); if (n <= 0) return false; npad = Math.ceil(n / plan.gran) * plan.gran; }
        let rows;
        if (own) { // trim to what the ranks it lands on can still hold
          const m = ownerWalk(load, q.pos, npad, plan.rankCap, plan.blk, plan.sp, false);
          if (m <= 0) return false;
          n = Math.min(n, m); npad = Math.ceil(n / 32) * 32;
          rows = new Array(plan.sp).fill(0); ownerWalk(load, q.pos, npad, plan.rankCap, plan.blk, plan.sp, true, rows);
        }
        segs.push({ q, n, npad, k: q.pos, first: q.first, last: n === q.rem, rows });
        q.first = false; q.pos += n; q.rem -= n; T += npad;
        return true;
      };
      if (rr) {
        // admit every waiting request that gets a slot/lane (oldest first; static slots skip a stream whose slot is busy) to
        // the back of the round-robin queue (rep.active), then serve it from the front. A request that is not done
        // goes to the back, behind everything waiting now. Batched, each popped request takes as many of the
        // remaining chunks as it can fill, as one attention call, so it is never split into two runs in one batch.
        if (rep.queue.length) {
          const keep = []; let blocked = false;
          for (const q of rep.queue) {
            if (!blocked && tryStart(q, rep)) rep.active.push(q);
            else { keep.push(q); if (!rep.slots) blocked = true; }
          }
          rep.queue = keep;
        }
        const again = [];
        if (cfg.batch) { // rounds over the queue, up to L units per request per round (batchChunksPerRequest)
          const unit = fixed ? C : plan.gran;
          const L = fixed ? (cfg.batchChunksPerRequest > 0 ? cfg.batchChunksPerRequest : Infinity)
            : cfg.batchTokensPerRequest > 0 ? Math.max(1, Math.floor(cfg.batchTokensPerRequest / unit)) : Infinity;
          const cand = [], need = [], alloc = [];
          // owner placement: the budget is per rank (sl = rows allocated so far on each rank), not a token total
          let left = own ? Infinity : Math.floor(budget / unit);
          const sl = own ? new Array(plan.sp).fill(0) : null;
          const fit = (q, off, a, apply) => (own ? ownerWalk(sl, q.pos + off * unit, a * unit, plan.rankCap, plan.blk, plan.sp, apply) / unit : a);
          // first round: requests join in queue order while the budget lasts and they get a lane
          for (const q of rep.active) {
            if (left <= 0 || cand.length >= plan.maxR) break;
            const n = Math.ceil(q.rem / unit), a = fit(q, 0, Math.min(n, L, left), false);
            if (a <= 0) break; // its first rank is full
            if (rrPool && !laneGet(q, rep)) break; // no lane free: a lane release re-pumps
            fit(q, 0, a, true);
            cand.push(q); need.push(n); alloc.push(a); left -= a;
          }
          // further rounds among them while budget is left and someone needs more
          for (let more = true; left > 0 && more;) {
            more = false;
            for (let i = 0; i < cand.length && left > 0; i++) {
              const a = fit(cand[i], alloc[i], Math.min(need[i] - alloc[i], L, left), true);
              if (a > 0) { alloc[i] += a; left -= a; more = true; }
            }
          }
          cand.forEach((q, i) => {
            rep.active.shift();
            take(q, alloc[i] * unit);
            if (rrPool) Object.assign(segs[segs.length - 1], q.seg);
            if (q.rem > 0) again.push(q);
          });
          for (const q of again) rep.active.push(q);
          return segs.length ? done() : null;
        }
        while (rep.active.length && room()) { // unbatched: one chunk of the front request
          const q = rep.active[0];
          if (rrPool && !laneGet(q, rep)) break; // no lane free: a lane release re-pumps
          rep.active.shift();
          take(q, budget - T);
          if (rrPool) Object.assign(segs[segs.length - 1], q.seg);
          if (q.rem > 0) again.push(q);
          break;
        }
        for (const q of again) rep.active.push(q);
        return segs.length ? done() : null;
      }
      // 1) continue active requests (they hold lanes), in start order
      for (let i = 0; i < cands.length && room(); i++) {
        if (cands[i].rem > 0) take(cands[i], budget - T);
        if (!cfg.batch && segs.length) break;
      }
      // 2) start new requests from the queue (only while the chunk has room for them)
      if (room() && (cfg.batch || segs.length === 0)) {
        queueOrder(rep);
        let i = 0;
        while (i < rep.queue.length && room()) {
          const q = rep.queue[i];
          if (!tryStart(q, rep)) { if (rep.slots) { i++; continue; } break; }
          rep.queue.splice(i, 1); rep.active.push(q);
          take(q, budget - T);
          if (!cfg.batch) break;
        }
      }
      rep.active = rep.active.filter((q) => q.rem > 0);
      return segs.length ? done() : null;
    }
    // round robin on the pool: a lane for one turn -> q.seg {lane, arena, miss}; false if none is free. The request's
    // own lane is reused while it still holds its context (also while its previous segment is still copying out);
    // otherwise a free lane, preferring one that holds nothing, else the least recently used. miss = copy the
    // whole context in.
    function laneGet(q, rep) {
      if (cfg.laneArena) {
        const a = TR.req_blocks[q.r] * B; if (rep.arenaUsed + a > plan.arena) return false;
        rep.arenaUsed += a; q.seg = { lane: -1, arena: a, miss: true }; return true;
      }
      const L = rep.laneTab;
      let pick = q.laneIdx >= 0 && L[q.laneIdx].owner === q ? q.laneIdx : -1;
      const miss = pick < 0;
      if (pick < 0) {
        for (let i = 0; i < L.length; i++) {
          if (L[i].busy) continue;
          if (!L[i].owner) { pick = i; break; }
          if (pick < 0 || L[i].t < L[pick].t) pick = i;
        }
        if (pick < 0) return false;
        if (L[pick].owner) L[pick].owner.laneIdx = -1;
        L[pick].owner = q; q.laneIdx = pick;
      }
      L[pick].busy++;
      q.seg = { lane: pick, arena: 0, miss };
      return true;
    }
    function pump(rep) {
      while (true) {
        while (rep.exits.length && rep.exits[0] <= now + 1e-12) heapPopNum(rep.exits);
        if (rep.free[0] > now + 1e-12) { schedPump(rep, rep.free[0]); return; }
        if (rep.exits.length >= plan.maxInflight) { schedPump(rep, rep.exits[0]); return; }
        if (!rep.active.length && !rep.queue.length) return;
        const ch = formChunk(rep);
        if (!ch) return; // blocked on lanes: a lane release re-pumps
        runChunk(rep, ch);
      }
    }
    function schedPump(rep, t) { if (rep.pumpAt >= 0 && rep.pumpAt <= t + 1e-12 && rep.pumpAt >= now) return; rep.pumpAt = t; ev.push(t, { e: EV_PUMP, rep }); }
    function runChunk(rep, ch) {
      const { segs } = ch;
      // batchDynShape off: batched chunks have one static shape (the budget), so a partly filled batch still pays
      // the full budget in every token-proportional op and in the stage-to-stage send; routed MoE ops keep
      // trimming to the real tokens (padding_config actual_isl), as they do for a padded chunk tail today
      const T = cfg.batch && !cfg.batchDynShape ? Math.max(ch.T, cfg.reqPad === 'chunk' ? Math.max(cfg.budget, cfg.chunk) : cfg.budget) : ch.T;
      let cs = segs.map((s) => ({ n: s.npad, na: s.n, k: s.k, cap: s.q.laneCap || plan.laneCap }));
      let trl;
      if (plan.owner) {
        // uneven ranks: a per-request attention call waits for the segment's busiest rank (nl); a fused call for the
        // rank with the most attention work over all segments (nlf: that rank's rows of each segment, work ~ rows x
        // context); routed MoE tokens scale with the busiest rank's rows
        let rb = 0, wb = -1, pad = 0, real = 0;
        for (let r = 0; r < plan.sp; r++) { let w = 0; for (const s of segs) w += s.rows[r] * (s.k + s.n / 2); if (w > wb) { wb = w; rb = r; } }
        cs.forEach((c, i) => { c.nl = Math.max(...segs[i].rows); c.nlf = segs[i].rows[rb]; pad += c.n; real += c.na; });
        trl = Math.max(...ch.load) * real / pad;
      }
      if (cfg.attn === 'chunk' && cfg.reqPad === 'chunk' && cs.some((s) => s.n > cfg.chunk)) { // one attention call per chunk
        const C = cfg.chunk;
        cs = cs.flatMap((s) => Array.from({ length: s.n / C }, (_, u) => ({ n: C, na: Math.max(0, Math.min(C, s.na - u * C)), k: s.k + u * C, cap: s.cap })));
      }
      chunkStageMs(plan, T, cs, scratch, trl);
      // pool copy-in (first segment of a request) / copy-out (last) per stage, DRAM bound (read + write at 50%)
      const copies = rep.pool && cfg.cache === 'pool' && plan.poolTok !== Infinity;
      if (copies) {
        let copyTok = 0;
        for (const s of segs) {
          // round robin: copy the segment's new KV out, and the whole context so far in unless the lane holds it
          if (rrPool) {
            const ci = s.miss ? s.k : 0;
            copyTok += ci + s.n; s.copyS = plan.copyMs(ci + s.n) / 1e3;
            if (warmDone && !s.first) { st.rrSegs++; if (!s.miss) st.rrReuse++; }
            if (warmDone) st.rrCopyIn += ci;
            continue;
          }
          if (s.first) { copyTok += s.q.hitTok; s.q.copyInS = plan.copyMs(s.q.hitTok) / 1e3; }
          if (s.last) { const nt = TR.req_blocks[s.q.r] * B - s.q.hitTok; copyTok += nt; s.q.copyOutS = plan.copyMs(nt) / 1e3; }
        }
        if (copyTok > 0) {
          for (let s = 0; s < S; s++) {
            const ms = copyTok * plan.stages[s].n * plan.kvbL / plan.P * 2 / (HW.dram * 0.5) * 1e3;
            scratch[s] += cfg.copyMode === 'sequential' ? ms : ms * cfg.copyContention;
          }
        }
      }
      const blk = plan.blockMs(T) / 1e3, hop = plan.hopMs(T) / 1e3;
      // a request whose prefix had to be re-fetched from host at admission cannot start before the fetch lands
      let arr = now;
      for (const s of segs) if (s.first && s.q.readyAt > arr) arr = s.q.readyAt;
      let end = arr, end0 = arr;
      for (let s = 0; s < S; s++) {
        const dt = scratch[s] / 1e3;
        const start = Math.max(rep.free[s], arr); end = start + dt;
        rep.free[s] = end + blk;
        if (warmDone) rep.busy[s] += Math.max(0, Math.min(end, tEnd) - Math.max(start, t0)); // busy inside the window
        if (s === 0) end0 = end;
        arr = end + hop;
      }
      heapPushNum(rep.exits, end);
      if (opts.onChunk) opts.onChunk(now, segs); // test hook: the segments of every chunk, in issue order
      if (warmDone && end >= t0 && end <= tEnd) { st.chunks++; st.segs += segs.length; st.processed += T; st.sendBlock += blk; }
      for (const s of segs) {
        if (rrPool) { // every segment frees its lane (held for its copies, as below); the last one also inserts the KV
          let hold = 0;
          if (copies && cfg.copyMode === 'double') hold = s.copyS || 0;
          else if (copies && cfg.copyMode === 'overlap3') hold = 2 * (scratch[0] / 1e3 + blk);
          ev.push((cfg.laneScope === 'stage' ? end0 : end) + hold, { e: EV_SEG, q: s.q, rep, upto: s.k + s.n, last: s.last, arena: s.arena, lane: s.lane });
          if (s.last) ev.push(end, { e: EV_DONE, q: s.q });
          continue;
        }
        if (s.last) {
          ev.push(end, { e: EV_DONE, q: s.q });
          // lane release: per-stage lanes when stage 0 finishes the last chunk, global lanes at prefill completion,
          // then held longer while copies overlap compute (see DEFAULTS.copyMode)
          if (cfg.cache !== 'slots') {
            let hold = 0;
            if (copies && cfg.copyMode === 'double') hold = (s.q.copyOutS || 0) + (s.q.copyInS || 0);
            else if (copies && cfg.copyMode === 'overlap3') hold = 2 * (scratch[0] / 1e3 + blk);
            ev.push((cfg.laneScope === 'stage' ? end0 : end) + hold, { e: EV_LANE, q: s.q, rep });
          }
        }
      }
      schedPump(rep, rep.free[0]);
    }
    function releaseLane(q, rep) {
      if (q.lane < 0) return;
      if (plan.lanes !== Infinity) rep.lanesUsed--;
      if (q.arena) { rep.arenaUsed -= q.arena; q.arena = 0; }
      q.lane = -1; pump(rep);
    }
    // copy-out: the request's KV enters the pool. With per-stage lanes that happens when the lane is freed (stage 0
    // finished the last chunk; later stages follow in FIFO order), otherwise at prefill completion.
    function insertKV(q, rep) {
      if (q.inserted) return; q.inserted = true;
      const pool = rep.pool;
      if (q.resv) { pool.unreserve(q.resv); q.resv = 0; }
      const n = pool.walk(q.tree.ns, TR.req_leaf[q.r] - TR.tr_pc0[q.tree.trace]); pool.touch(n, now);
      if (q.pins) { pool.unpin(q.pins); q.pins = null; }
      if (q.prog) { rep.progTok -= q.prog; rep.nProg--; q.prog = 0; }
    }
    function onLaneFree(q, rep) { releaseLane(q, rep); insertKV(q, rep); }
    // round robin on the pool: a segment's lane is free (it still holds the request's context until handed to another
    // request); its new KV is now in the pool (reserved, i.e. pinned, until the whole request is inserted), beyond the
    // part of the cached prefix that is pinned in place
    function onSegFree(p) {
      const q = p.q, rep = p.rep;
      if (p.lane >= 0) {
        const l = rep.laneTab[p.lane]; l.busy--; l.t = now;
        if (p.last && l.owner === q) { l.owner = null; q.laneIdx = -1; }
      }
      if (p.arena) rep.arenaUsed -= p.arena;
      if (p.last) insertKV(q, rep);
      else if (!q.inserted && rep.pool && plan.poolTok !== Infinity) {
        const tgt = Math.max(0, p.upto - (q.pinTok || 0)) / B;
        if (tgt > (q.resv || 0)) { rep.pool.reserve(tgt - (q.resv || 0)); q.resv = tgt; }
      }
      pump(rep);
    }
    function onDone(q) { // prefill complete (TTFT)
      const tree = q.tree, rep = tree.rep, r = q.r;
      q.tDone = now;
      if (rep.slots) { rep.slots.release(q.slot, now, TR.req_blocks[r]); pump(rep); }
      else if (cfg.laneScope !== 'stage') insertKV(q, rep); // global lanes: KV visible at completion, lane freed by EV_LANE
      const inTok = TR.req_blocks[r] * B;
      if (q.primer) { decWarm(q); st.primerTok += inTok - q.hitTok; primersLeft--; if (primersLeft === 0 && !warmDone) startProfiling(now); ev.push(now, { e: EV_END, q }); return; }
      if (warmDone && now >= t0 && now <= tEnd) {
        const useful = (TR.req_blocks[r] - TR.req_lcp_best[r]) * B;
        st.done++; st.laneWait += q.waitS || 0; st.useful += Math.min(useful, inTok - q.hitTok); st.newTok += inTok - q.hitTok; st.hitTok += q.hitTok; st.inTok += inTok; st.outTok += TR.req_out[r];
        st.infHitTok += TR.req_lcp_best[r] * B; st.reprefill += Math.max(0, (inTok - q.hitTok) - useful);
        if (cfg.logRequests) (st.reqLog || (st.reqLog = [])).push(r); // opt-in: request ids completed in the window
        q.ttftWin = true; pendTtft.add(q); // TTFT is taken when decode begins (the first token)
      }
      q.prefillEnd = now;
      decStep(q);
    }
    // decode begins once the request's decode KV is on the device (read-backs and write-backs landed) and, with global
    // lanes, once it holds a lane (copied in); EV_DSTART re-enters here when the wait is over
    function decStep(q) {
      if (q.decReadyAt > now + 1e-12) { ev.push(q.decReadyAt, { e: EV_DSTART, q }); return; }
      if (q.parked) { // past prefill without a slot: wait for one (FIFO)
        if (slotQ.length || !decUnpark(q)) { q.laneWaited = true; slotQ.push(q); }
        return;
      }
      if (decGlobal && q.decSlot && !q.decLane) {
        if (decLanesUsed >= plan.decLanes || decLaneQ.length) { q.laneWaited = true; decLaneQ.push(q); return; }
        decLaneTake(q); return;
      }
      decBegin(q);
    }
    function decBegin(q) {
      if (q.ttftWin) {
        q.ttftWin = false; pendTtft.delete(q); st.ttft.push(now - q.tReady);
        st.decStarts++; st.decStartDelay += now - q.prefillEnd; if (q.laneWaited) st.decLaneWaited++;
      }
      const ctx = TR.req_blocks[q.r] * B, speed = decodeSpeed(cfg, ctx); // its context: the prompt (output adds ~1k)
      if (!q.decSlot) { ev.push(now + TR.req_out[q.r] / speed, { e: EV_END, q }); return; }
      q.decBeginT = now;
      q.decBase = speed; q.decCopySpeed = decStage ? 1 / (1 / speed + plan.decCopyTrip(ctx)) : speed;
      if (cfg.decodeStages > 0) {
        decAdvance(); decTick(); decoding++; decRerate();
        q.decSpeed = decCopying ? q.decCopySpeed : speed; decSpeedSum += q.decSpeed;
        ringPush({ thr: decV + TR.req_out[q.r] / q.decSpeed, q }); decSchedule(); return;
      }
      q.decSpeed = speed;
      decTick(); decoding++; decSpeedSum += speed;
      ev.push(now + TR.req_out[q.r] / speed, { e: EV_END, q });
    }
    function onEnd(q) {
      const tree = q.tree, r = q.r;
      inflightReqs--; tree.live--;
      if (q.decSlot) {
        // decode speed this session got, output tokens / decode time (the decode-speed SLO, tsuP10)
        if (warmDone && now >= t0 && now <= tEnd && TR.req_out[r] > 0) st.tsu.push(TR.req_out[r] / Math.max(1e-9, now - q.decBeginT));
        decTick(); decHeld--; decoding--; decSpeedSum -= q.decSpeed || 0; q.decSlot = false;
        if (cfg.decodeStages > 0) decRerate();
        decRelease(q); decDrain();
      }
      if (tree.ended) {
        tree.ended[r - tree.r0] = 1;
        const w = tree.waiters.get(r);
        if (w) {
          tree.waiters.delete(r);
          for (const g of w) if (--g.need === 0) { if (warmDone) st.gateWait += now - g.gatedAt; issue(g); }
        }
      }
      if (!q.primer) {
        const s = TR.req_stream[r], first = TR.st_first[s], cnt = TR.st_cnt[s];
        if (s === tree.s0) {
          for (let x = TR.spawnHead[r]; x >= 0; x = TR.spawnNext[x]) if (!TR.st_ovl[x]) dispatch(tree, TR.st_first[x], now + TR.st_off[x]);
          if (r + 1 < first + cnt) {
            if ((tree.pend.get(r + 1) || 0) > 0) tree.waiting.set(r + 1, true);
            else dispatch(tree, r + 1, now + Math.min(cfg.gapCap, TR.req_delay[r + 1]));
          }
        } else {
          if (r + 1 < first + cnt) dispatch(tree, r + 1, now + Math.min(cfg.gapCap, TR.req_delay[r + 1]));
          else {
            const j = TR.st_join[s];
            if (j >= 0) {
              const left = (tree.pend.get(j) || 0) - 1; tree.pend.set(j, left);
              if (left <= 0 && tree.waiting.get(j)) { tree.waiting.delete(j); dispatch(tree, j, now); }
            }
          }
        }
      }
      if (tree.live === 0 && warmDone && !tree.prof && trees[tree.lane] === tree) recycle(tree, now);
    }
    function initPend(tree) {
      const s0 = tree.s0, ns = TR.trStreams[tree.trace];
      for (let s = s0 + 1; s < s0 + ns; s++) { const j = TR.st_join[s]; if (j >= 0) tree.pend.set(j, (tree.pend.get(j) || 0) + 1); }
    }
    // ---------- start
    const trees = [];
    for (let l = 0; l < cfg.concurrency; l++) {
      let tree = null;
      for (let tries = 0; !tree && tries < TR.nT; tries++) { // unspawnable traces are skipped (AIPerf _build_trajectories)
        const trace = nextTrace++ % TR.nT;
        const tstar = (cfg.startMin + (cfg.startMax - cfg.startMin) * rnd()) * TR.tr_dur[trace];
        tree = newTree(l, trace, tstar, 0);
      }
      trees.push(tree);
    }
    if (primersLeft === 0) startProfiling(0);
    // ---------- event loop
    let evCount = 0;
    const maxEv = opts.maxEvents || 5e7;
    while (ev.size) {
      let t = ev.peekT();
      if (warmDone && inflightReqs === 0 && t > now + cfg.idleCap) { ev.shiftAll(t - now - cfg.idleCap); st.idleWarps++; t = ev.peekT(); }
      if (t > tEnd) break;
      if (!warmDone && t > cfg.maxWarmup) { st.warmupTimeout = true; break; }
      if (warmDone && t > now) { // why stage 0 has nothing to issue in [now, t), see pfStarved* in the results
        const a = Math.max(now, t0), b = Math.min(t, tEnd);
        if (b > a) for (const rp of reps) {
          if (rp.free[0] > now + 1e-12 || rp.queue.length || rp.active.some((q) => q.rem > 0)) continue; // busy or has work
          if (decQ.length) st.pfSlot += b - a; else st.pfNone += b - a;
        }
      }
      now = Math.max(now, t);
      const p = ev.pop(); evCount++;
      if (evCount > maxEv) { st.eventCap = true; break; }
      switch (p.e) {
        case EV_READY: onReady(p.q); break;
        case EV_FETCHED: enqueue(p.q); break;
        case EV_DONE: onDone(p.q); break;
        case EV_END: onEnd(p.q); break;
        case EV_DEC: onDecodeEvent(p.gen); break;
        case EV_DSTART: decStep(p.q); break;
        case EV_PUMP: if (p.rep.pumpAt === t) p.rep.pumpAt = -1; pump(p.rep); break;
        case EV_LANE: onLaneFree(p.q, p.rep); break;
        case EV_SEG: onSegFree(p); break;
      }
    }
    // ---------- results
    const D = warmDone ? Math.min(cfg.duration, Math.max(1e-9, now - t0)) : 1;
    decTick();
    // requests whose decode had not begun when the window closed: TTFT at least up to the close
    const tCut = Math.max(now, Math.min(tEnd, ev.size ? ev.peekT() : now));
    for (const q of pendTtft) st.ttft.push(tCut - q.tReady);
    // sessions still decoding on the ring when the window closed: the speed they got so far (the slowest are the
    // ones still running, so leaving them out would flatter tsuP10)
    if (warmDone) { decAdvance(); for (const e of ring) { const out = TR.req_out[e.q.r], dt = now - e.q.decBeginT; if (out > 0 && dt > 1e-6) st.tsu.push((out - Math.max(0, e.thr - decV) * e.q.decSpeed) / dt); } }
    const tsuS = Float64Array.from(st.tsu).sort(), tsuPct = (p) => (tsuS.length ? tsuS[Math.min(tsuS.length - 1, Math.floor(p * (tsuS.length - 1)))] : NaN);
    const tt = Float64Array.from(st.ttft).sort();
    const pct = (p) => (tt.length ? tt[Math.min(tt.length - 1, Math.floor(p * (tt.length - 1)))] : NaN);
    const util = []; for (let s = 0; s < S; s++) { let u = 0; for (const rp of reps) u += rp.busy[s]; util.push(u / (D * reps.length)); }
    return {
      plan: planSummary(plan), cfg,
      usefulTps: st.useful / D, processedTps: st.processed / D, newTps: st.newTok / D, reqPerS: st.done / D,
      // input tokens of the requests completed in the window: cached (prefix hit) + new (prefilled, incl. re-prefill)
      inTps: st.inTok / D, hitTps: st.hitTok / D, outTps: st.outTok / D, // + output tokens they will decode
      ttftP50: pct(0.5), ttftP90: pct(0.9), ttftP99: pct(0.99), ttftMean: tt.length ? tt.reduce((a, b) => a + b, 0) / tt.length : NaN,
      hitRate: st.inTok ? st.hitTok / st.inTok : NaN, infHitRate: st.inTok ? st.infHitTok / st.inTok : NaN,
      reprefillFrac: st.newTok ? st.reprefill / st.newTok : 0, padFrac: st.processed ? 1 - st.newTok / st.processed : 0,
      alignLossFrac: st.inTok ? st.alignLoss / st.inTok : 0,
      avgChunkTok: st.chunks ? st.processed / st.chunks : 0, avgSegsPerChunk: st.chunks ? st.segs / st.chunks : 0,
      stageUtil: util, maxUtil: Math.max(...util), done: st.done, warmupS: st.warmupS, primers: st.primers, primerTok: st.primerTok,
      laneWaitMean: st.done ? st.laneWait / Math.max(1, st.done) : 0, hostTok: st.hostTok, ssdTok: st.ssdTok, idleWarps: st.idleWarps,
      // offload traffic (tokens read back per second from host DRAM / SSD) and link busy shares
      hostReadTps: st.hostTok / D, ssdReadTps: st.ssdTok / D,
      pcieH2DUtil: st.h2dFreeBusy / (D * reps.length), pcieD2HUtil: st.d2hFreeBusy / (D * reps.length), ssdUtil: st.ssdFreeBusy / (D * reps.length),
      // decode slots: held on average and at most (admission to decode end), decoding on average, and the requests
      // that waited for a slot (count per second, mean wait)
      decSlotsMean: st.decArea / D, decSlotsMax: st.decMax, decodingMean: st.decingArea / D,
      decWaitPerS: st.decWaited / D, decWaitMean: st.decWaited ? st.decWait / st.decWaited : 0, decQueuedAtEnd: decQ.length,
      // decode: mean speed per decoding session (tokens/s/u: decodeTps, unless the ring is oversubscribed or the
      // speed follows the context) and the
      // share of the window it carried decodeStages or more sessions
      decodeTpsMean: st.decingArea > 0 ? st.decTokArea / st.decingArea : cfg.decodeTps, ringFullFrac: cfg.decodeStages > 0 ? st.ringFull / D : 0,
      // output tokens decoded per second in the window (what output revenue bills; outTps counts them at prefill end)
      outDecTps: st.decTokArea / D,
      // decode KV (decodeCache): share of the context already cached on the decode side at admission (on device or read
      // back from its host DRAM / SSD), so not migrated from prefill; tokens migrated per second (not modelled as a
      // cost); read-backs and the decode galaxies' link shares; pool tokens pinned or reserved by sessions (mean);
      // mean time from prefill end to the first token, and the share of requests that waited for a global lane
      decHitRate: st.decCtxTok ? st.decHitTok / st.decCtxTok : NaN, migTps: (st.decCtxTok - st.decHitTok) / D,
      decHostReadTps: st.decHostTok / D, decSsdReadTps: st.decSsdTok / D,
      decPcieH2DUtil: st.dH2dBusy / D, decPcieD2HUtil: st.dD2hBusy / D, decSsdUtil: st.dSsdBusy / D,
      decPoolTokMean: decPool ? st.decPoolArea / D : NaN,
      decStartDelayMean: st.decStarts ? st.decStartDelay / st.decStarts : 0, decLaneWaitFrac: st.decStarts ? st.decLaneWaited / st.decStarts : 0,
      // decode speed each session got (output tokens / its decode time, tokens/s/u): 10th percentile and median
      tsuP10: tsuPct(0.1), tsuP50: tsuPct(0.5),
      // queue backpressure: requests parked on the decode SSDs per second and on average
      decParkPerS: st.parked / D, decParkedMean: st.parkArea / D,
      // share of the window prefill's first stage had nothing to issue (no queued request, no started request with
      // tokens left, stage free), by cause: requests waiting for a decode KV slot, else no demand. sendBlockFrac:
      // share of time a stage is held after its compute by the synchronous handoff to the next stage
      pfStarvedSlotFrac: st.pfSlot / (D * reps.length), pfStarvedIdleFrac: st.pfNone / (D * reps.length),
      sendBlockFrac: st.sendBlock / (D * reps.length),
      gated: st.gated, gateWaitMean: st.gated ? st.gateWait / st.gated : 0, legacyStarts: st.legacyStarts, skippedTraces: st.skippedTraces,
      gatedAtEnd: trees.reduce((a, tr) => { const u = new Set(); if (tr) for (const w of tr.waiters.values()) for (const g of w) u.add(g); return a + u.size; }, 0),
      warmupTimeout: !!st.warmupTimeout, eventCap: !!st.eventCap, events: evCount, duration: D,
      slotEvictions: reps.reduce((a, r) => a + (r.slots ? r.slots.evictions : 0), 0),
      // round robin on the pool: share of continuation turns that found their context still in their lane, and the
      // copy-in rate (tokens copied pool -> lane per second)
      rrLaneReuse: st.rrSegs ? st.rrReuse / st.rrSegs : NaN, rrCopyInTps: st.rrCopyIn / D,
      reqLog: st.reqLog,
    };
  }
  function heapPushNum(h, x) { let i = h.length; h.push(x); while (i > 0) { const j = (i - 1) >> 1; if (h[j] <= x) break; h[i] = h[j]; i = j; } h[i] = x; }
  function heapPopNum(h) { const top = h[0], last = h.pop(), n = h.length; if (n) { let i = 0; while (true) { let l = 2 * i + 1; if (l >= n) break; if (l + 1 < n && h[l + 1] < h[l]) l++; if (h[l] >= last) break; h[i] = h[l]; i = l; } h[i] = last; } return top; }

  function planSummary(plan) {
    return {
      S: plan.S, mesh: [plan.sp, plan.tp], counts: plan.counts, capTok: plan.capTok, nSlots: plan.nSlots, poolTok: plan.poolTok,
      lanes: plan.lanes, arena: plan.arena, hostTok: plan.hostTok, ssdTok: plan.ssdTok, hostKvGB: +plan.hostBudget.kv.toFixed(1),
      decLanes: plan.decLanes, decPoolTok: plan.decPoolTok, decHostTok: plan.decHostTok, decSsdTok: plan.decSsdTok,
      errors: plan.errors, kvbL: plan.kvbL,
      weightsGBperChip: plan.stages.map((s) => +s.weightsGBperChip.toFixed(2)),
    };
  }

  // ------------------------------------------------------------------------------------------------------
  // Matrix-cell replay (validation vs the #57827 tables): U users stream (cached, new) requests
  // ------------------------------------------------------------------------------------------------------
  function matrixCell(cal, cfgIn, cached, nnew, users, reqsPerUser) {
    // the #57827 matrix runs used bf16 index_k (M3_INDEX_CACHE_BF16=1)
    const plan = makePlan(Object.assign({ cache: 'inf', batch: false, reqPad: 'chunk', idxBf16: true }, cfgIn), cal);
    const S = plan.S, C = plan.cfg.chunk, out = new Float64Array(S);
    const cap = cached + 51200;
    const cachedA = Math.floor(cached / C) * C;
    const nCh = Math.max(1, Math.ceil(nnew / C));
    const push = (free, j, arrT) => {
      const na = Math.min(C, nnew - j * C);
      chunkStageMs(plan, C, [{ n: C, na, k: cachedA + j * C, cap }], out);
      const blk = plan.blockMs(C), hop = plan.hopMs(C);
      let arr = arrT, end = arrT;
      for (let s = 0; s < S; s++) { const st = Math.max(free[s], arr); end = st + out[s]; free[s] = end + blk; arr = end + hop; }
      return end;
    };
    // idle TTFT: one request on an empty pipeline (h2d ~1 ms)
    let free = new Float64Array(S); let e = 0;
    for (let j = 0; j < nCh; j++) e = push(free, j, 1);
    const idle = e;
    // loaded: the producer streams all users' requests back to back, round-robin at chunk granularity
    free = new Float64Array(S);
    const exits = []; let tNow = 0;
    const total = users * reqsPerUser * nCh;
    for (let i = 0; i < total; i++) {
      const j = Math.floor(i / users) % nCh;
      const arrive = Math.max(tNow, free[0]);
      const x = push(free, j, arrive); exits.push(x); tNow = arrive;
    }
    exits.sort((a, b) => a - b);
    const drop = Math.min(16, Math.floor(exits.length / 4));
    const span = exits[exits.length - 1 - drop] - exits[drop];
    const period = span / (exits.length - 1 - 2 * drop);
    const processed = C / period * 1e3;
    return { idleTtftMs: idle, periodMs: period, processedTps: processed, newTps: processed * nnew / (nCh * C) };
  }

  // Fit handoff block/hop against the measured tables (needs the pipeline fit first)
  function fitHandoff(cal, data) {
    // single-chunk cells: measured loaded chunk period - modelled bottleneck stage = blocking handoff per chunk;
    // (idle TTFT - sum of stages) / 16 = per-hop latency
    const pts = { 5120: [], 2048: [] };
    for (const run of ['A', 'B', 'C']) {
      const r = data.pipeline[run]; const T = r.chunk; const counts = r.layers.split(',').map(Number);
      const plan = makePlan({ chunk: T, split: counts, stages: 16, cache: 'inf', idxBf16: true }, cal); // runs used bf16 index_k
      for (const row of data.tables[run]) {
        if (row.new > T || row.loaded_steady_processed_tps == null) continue;
        const cell = (r.cells || []).find((c) => c.cached === row.cached && c.new === row.new);
        const periodMeas = cell && cell.period_ms ? cell.period_ms : T / row.loaded_steady_processed_tps * 1e3;
        const o = new Float64Array(16);
        chunkStageMs(plan, T, [{ n: T, na: row.new, k: Math.floor(row.cached / T) * T, cap: row.cached + 51200 }], o);
        const sum = o.reduce((a, b) => a + b, 0), mx = Math.max(...o);
        pts[T].push({ block: periodMeas - mx, hop: (row.idle_ttft_ms_median - 1 - sum) / 15 });
      }
    }
    const med = (a, k) => { const v = a.map((p) => p[k]).sort((x, y) => x - y); return v[Math.floor(v.length / 2)]; };
    const b5 = Math.max(0, med(pts[5120], 'block')), b2 = Math.max(0, med(pts[2048], 'block'));
    const h5 = Math.max(0, med(pts[5120], 'hop')), h2 = Math.max(0, med(pts[2048], 'hop'));
    const s = (y5, y2) => { const b1 = (y5 - y2) / ((5120 - 2048) / 1000); return [y5 - b1 * 5.12, b1]; };
    cal.pipe.block = s(b5, b2); cal.pipe.hop = s(h5, h2);
    cal.fit.block = { b5, b2 }; cal.fit.hop = { h5, h2 };
  }

  function calibrateAll(data) { const cal = calibrate(data); fitHandoff(cal, data); return cal; }

  // List prices, USD per million tokens: MiniMax's own endpoint for minimax/minimax-m3 on OpenRouter (provider
  // "Minimax", minimax/fp8; also the model-level price), read 2026-10-07 from
  // openrouter.ai/api/v1/models/minimax/minimax-m3/endpoints. A cache miss (re-prefill
  // included) is billed at the input price, a prefix hit at the cache-read price; there is no cache-write charge.
  // Prefill revenue is input tokens only; output tokens are decode's, and count only in the margin (economics()),
  // which also pays for the decode galaxies.
  const PRICE = { inUsdPerM: 0.30, cachedUsdPerM: 0.06, outUsdPerM: 1.20 };
  // Operating cost: a rough USD per galaxy-hour, and the decode galaxies paired with the prefill: one M3 decode
  // instance, 64 sessions (decodeSlots) on 16 galaxies.
  const COST = { galaxyUsdPerH: 12, decodeGalaxies: 16 };

  // Hourly volume of one sweep point: input tokens of the completed requests, split into new (prefilled) and cached
  // (prefix hit), requests completed, the input revenue they bill at `price` (USD per hour), and their output tokens
  // and output revenue (NaN for points that predate outTps). Points from studies run before inTps/hitTps existed are
  // derived from newTps and hitRate (newTok = inTok - hitTok, hitRate = hitTok / inTok), which is exact.
  function hourly(p, price) {
    if (!p) return null;
    const pr = Object.assign({}, PRICE, price);
    const inTps = p.inTps != null ? p.inTps : p.newTps != null && p.hitRate < 1 ? p.newTps / (1 - p.hitRate) : NaN;
    const hitTps = p.hitTps != null ? p.hitTps : inTps * p.hitRate;
    const newTok = 3600 * (inTps - hitTps), cachedTok = 3600 * hitTps;
    const newUsd = newTok * pr.inUsdPerM / 1e6, cachedUsd = cachedTok * pr.cachedUsdPerM / 1e6;
    // output tokens decoded in the window (points from before outDecTps: those of the requests completed)
    const outTok = p.outDecTps != null ? 3600 * p.outDecTps : p.outTps != null ? 3600 * p.outTps : NaN;
    return { inTok: 3600 * inTps, newTok, cachedTok, req: 3600 * p.reqPerS, usd: newUsd + cachedUsd, newUsd, cachedUsd, outTok, outUsd: outTok * pr.outUsdPerM / 1e6 };
  }

  // Hourly margin of one sweep point on `galaxies` prefill galaxies plus the decode galaxies: input + output revenue
  // minus the galaxy-hours of both (USD per hour).
  function economics(p, galaxies, price, cost) {
    const h = hourly(p, price);
    if (!h) return null;
    const c = Object.assign({}, COST, cost);
    const revenue = h.usd + h.outUsd, prefillUsd = galaxies * c.galaxyUsdPerH, decodeUsd = c.decodeGalaxies * c.galaxyUsdPerH;
    const costUsd = prefillUsd + decodeUsd;
    return { revenue, inUsd: h.usd, outUsd: h.outUsd, cost: costUsd, prefillUsd, decodeUsd, margin: revenue - costUsd, marginFrac: (revenue - costUsd) / revenue };
  }

  const API = { M3, HW, DEFAULTS, PRICE, COST, M3_DECODE_TSU, decodeSpeed, calibrate: calibrateAll, makePlan, splitLayers, chunkStageMs, loadTraffic, simulate, matrixCell, planSummary, hourly, economics, layerMs, roofTok, roofSeg, ownerWalk };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.M3SIM = API;
})(typeof self !== 'undefined' ? self : this);
