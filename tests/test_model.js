#!/usr/bin/env node
// Regression checks for the cost model, plan validation and a few replay invariants. Run: node tests/test_model.js
'use strict';
const assert = require('assert');
const path = require('path');
const SIM = require('../sim_core.js');
const { loadAll } = require('../run.js');
const P = require('../lib/paths.js');

const { TR, cal } = loadAll(P.DATA);
const plan = (cfg) => SIM.makePlan(Object.assign({ chunk: 2048 }, cfg), cal);

// 1. MoE layer cost is non-decreasing in chunk size (the 1/T pipeline term is clamped below 2048)
for (const opEff of [0, 0.5, 1]) {
  const p = plan({ opEff });
  let prev = 0;
  for (let T = 64; T <= 32768; T *= 2) {
    const t = p.layer('moe', T, [{ n: T, na: T, k: 100000, cap: 0 }]);
    assert.ok(t >= prev - 1e-9, `moe layer not monotone at T=${T}, opEff=${opEff}`);
    prev = t;
  }
}

// 2. fused multi-user attention is never slower than per-request attention
for (const kind of ['moe', 'dense']) {
  const pf = plan({ attn: 'fused', batch: true, reqPad: 'tile', budget: 16384 }), ps = plan({ attn: 'request', batch: true, reqPad: 'tile', budget: 16384 });
  for (const segs of [[{ n: 8192, k: 500000 }, { n: 256, k: 1000 }], [{ n: 640, k: 50000 }, { n: 640, k: 60000 }, { n: 1600, k: 1e5 }]]) {
    const T = segs.reduce((a, s) => a + s.n, 0);
    const s2 = segs.map((s) => Object.assign({ na: s.n, cap: 0 }, s));
    assert.ok(pf.layer(kind, T, s2) <= ps.layer(kind, T, s2) + 1e-9, `fused > seq for ${kind}`);
  }
}

// 3. at opEff 0 every op runs exactly at its measured efficiency (no cap at the target)
const e0 = plan({ opEff: 0 }).eff;
assert.strictEqual(e0.moe.qkv, cal.effs['2x4'].moe.qkv);

// 4. buffers that must hold a whole request are rejected when too small
assert.ok(plan({ cache: 'pool', laneArena: true, arenaTokens: 5e5 }).errors.some((e) => e.includes('arena')));
assert.ok(plan({ cache: 'slots', slotLen: 262144 }).errors.some((e) => e.includes('slots')));
// batching works with every cache; static slots run out of memory when requests per batch x stages > slots
const oom = (cfg) => plan(cfg).errors.some((e) => e.startsWith('out of memory'));
assert.ok(oom({ cache: 'slots', batch: true, budget: 8192 }), '4 per batch x 16 stages > 20 slots');
assert.ok(!oom({ cache: 'slots' }) && !oom({ cache: 'slots', galaxies: 8, stages: 32 }), 'unbatched slots fit');
assert.ok(!oom({ cache: 'slots', batch: true, budget: 8192, stages: 4, mesh: [8, 4] }), '4 per batch x 4 stages <= 22 slots');
assert.ok(oom({ cache: 'slots', batch: true, budget: 8192, reqPad: 'tile', stages: 4, mesh: [8, 4] }), 'tile padding: budget / 32SP per batch');
for (const cache of ['pool', 'paging', 'inf']) assert.ok(!plan({ cache, batch: true }).errors.length, cache);

// 5. replay: paging with an unbounded pool behaves exactly like the infinite cache
const base = { concurrency: 24, duration: 600, chunk: 2048 };
const a = SIM.simulate(TR, cal, Object.assign({ cache: 'inf' }, base));
const b = SIM.simulate(TR, cal, Object.assign({ cache: 'paging', reserveGB: -1e6 }, base)); // absurd capacity
assert.strictEqual(a.done, b.done); assert.ok(Math.abs(a.usefulTps - b.usefulTps) < 1e-6 * a.usefulTps + 1e-9);

// 6. replay: chunk-padded batching packs one segment per request per chunk, so chunks carry several requests
const r = SIM.simulate(TR, cal, Object.assign({ cache: 'inf', batch: true, budget: 16384, reqPad: 'chunk' }, base, { concurrency: 64 }));
assert.ok(r.avgSegsPerChunk >= 1 && r.done > 0 && r.maxUtil <= 1 + 1e-9);

// 7. replay: offload tiers behind static slots keep evicted slots' KV, so reads happen and the hit rate improves; the
//    infinite cache has no tier
{
  const cfg = Object.assign({ cache: 'slots', chunk: 2048, unaligned: true }, base, { concurrency: 64 });
  const a = SIM.simulate(TR, cal, cfg), b = SIM.simulate(TR, cal, Object.assign({ hostTier: true }, cfg));
  assert.ok(SIM.makePlan(Object.assign({ hostTier: true }, cfg), cal).hostTok > 0 && b.hostTok > 0, 'no host reads behind slots');
  assert.ok(b.hitRate > a.hitRate, `slots + tiers hit ${b.hitRate} <= ${a.hitRate}`);
  const inf = SIM.makePlan(Object.assign({ hostTier: true }, cfg, { cache: 'inf' }), cal);
  assert.ok(inf.hostTok === 0 && inf.ssdTok === 0);
}

// 7b. offload tiers: the host DRAM share left for KV is DRAM minus the reserves; with a small host tier, pool pages
//     cascade device -> host -> SSD and are read back from both; without host DRAM, evictions go straight to SSD
{
  const p = SIM.makePlan(Object.assign({ cache: 'pool', hostTier: true }, base), cal), h = p.hostBudget;
  assert.ok(Math.abs(h.kv - (576 * 0.9 - 16 - 32 - 32 - 32 * 1.073741824 - h.weights)) < 1e-9 && h.weights > 0 && p.ssdTok > p.hostTok, `host kv ${h.kv}`);
  const cfg = Object.assign({ cache: 'pool', chunk: 2048, hostTier: true, hostDramGBPerGalaxy: 260 }, base, { concurrency: 256 }); // about 50 GB of KV
  const r = SIM.simulate(TR, cal, cfg), n = SIM.simulate(TR, cal, Object.assign({}, cfg, { hostDramGBPerGalaxy: 0 }));
  assert.ok(r.hostTok > 0 && r.ssdTok > 0 && r.pcieH2DUtil > 0 && r.ssdUtil > 0, `reads host ${r.hostTok} ssd ${r.ssdTok}`);
  assert.ok(n.hostTok === 0 && n.ssdTok > 0, 'no host tier: SSD only');
}

// 8. unaligned resume (tt-metal #57636) is the default: no cached tokens are lost to chunk rounding unless
//    unaligned=false asks for the old behaviour
{
  assert.strictEqual(SIM.DEFAULTS.unaligned, true);
  const cfg = Object.assign({ cache: 'slots', chunk: 2048 }, base);
  const u = SIM.simulate(TR, cal, cfg), a = SIM.simulate(TR, cal, Object.assign({ unaligned: false }, cfg));
  assert.ok(u.alignLossFrac === 0 && a.alignLossFrac > 0, `align loss ${u.alignLossFrac} / ${a.alignLossFrac}`);
}

// 9. hourly volume: input = new + cached, and the fallback for older study points (derived from newTps and hitRate)
//    gives the same numbers as the simulator's own inTps / hitTps
{
  const r = SIM.simulate(TR, cal, Object.assign({ cache: 'slots', chunk: 2048 }, base, { concurrency: 64 }));
  const h = SIM.hourly(r), { inTps, hitTps, ...old } = r, g = SIM.hourly(old);
  assert.ok(Math.abs(r.inTps - r.newTps - r.hitTps) < 1e-6 * r.inTps && Math.abs(h.req - 3600 * r.reqPerS) < 1e-9);
  for (const k of ['inTok', 'newTok', 'cachedTok', 'req', 'usd']) assert.ok(Math.abs(h[k] - g[k]) < 1e-6 * h[k], `hourly fallback ${k}`);
  // revenue: new tokens at the input price, cached at the cache-read price; a partial override keeps the other price
  const { inUsdPerM, cachedUsdPerM } = SIM.PRICE;
  assert.ok(Math.abs(h.usd - (h.newTok * inUsdPerM + h.cachedTok * cachedUsdPerM) / 1e6) < 1e-9 * h.usd && h.usd === h.newUsd + h.cachedUsd);
  const o = SIM.hourly(r, { inUsdPerM: 2 * inUsdPerM });
  assert.ok(Math.abs(o.newUsd - 2 * h.newUsd) < 1e-9 * h.newUsd && o.cachedUsd === h.cachedUsd);
  // output tokens: those of the requests completed in the window; points without outTps have no output revenue
  const l = SIM.simulate(TR, cal, Object.assign({ cache: 'slots', chunk: 2048 }, base, { concurrency: 64, logRequests: true }));
  let out = 0; for (const q of l.reqLog) out += TR.req_out[q];
  const { outTps, ...noOut } = old;
  assert.ok(Math.abs(l.outTps - out / l.duration) < 1e-9 * l.outTps && Number.isNaN(SIM.hourly(noOut).outUsd));
  // margin: input + output revenue minus prefill and decode galaxy-hours
  const e = SIM.economics(r, 8, null, { galaxyUsdPerH: 10, decodeGalaxies: 16 });
  assert.ok(Math.abs(e.margin - (h.usd + h.outUsd - 240)) < 1e-9 && e.cost === 240 && e.decodeUsd === 160 && h.outUsd > 0);
}

// 10. decode slots: without a limit nothing waits and the results match the default exactly; with a limit, slots held
//     never exceed it, requests wait for one, and fewer requests complete
{
  const cfg = Object.assign({ cache: 'pool', chunk: 512, batch: true, budget: 8192 }, base, { concurrency: 128 });
  const d = SIM.simulate(TR, cal, cfg), u = SIM.simulate(TR, cal, Object.assign({ decodeSlots: 0 }, cfg));
  const l = SIM.simulate(TR, cal, Object.assign({ decodeSlots: 8 }, cfg));
  assert.ok(d.usefulTps === u.usefulTps && d.ttftP90 === u.ttftP90 && u.decWaitPerS === 0 && u.decSlotsMax > 8, `unlimited ${u.decSlotsMax}`);
  assert.ok(u.decodingMean > 0 && u.decodingMean <= u.decSlotsMean, `decoding ${u.decodingMean} / held ${u.decSlotsMean}`);
  assert.ok(l.decSlotsMax === 8 && l.decSlotsMean <= 8 && l.decWaitPerS > 0 && l.decWaitMean > 0 && l.reqPerS < u.reqPerS,
    `limit 8: max ${l.decSlotsMax} mean ${l.decSlotsMean} waits ${l.decWaitPerS} req ${l.reqPerS} vs ${u.reqPerS}`);
}

// 10b. decode ring (decodeStages): every session past prefill decodes, each at decodeTps x min(1, stages / N).
//      A ring that never fills matches the fixed-speed model; an oversubscribed one slows every session, never
//      decodes more than stages x decodeTps tokens/s in total, and serves fewer requests
{
  const cfg = Object.assign({ cache: 'pool', chunk: 512, batch: true, budget: 8192, decodeSlots: 16, decodeCurve: 'flat' }, base, { concurrency: 128 });
  const u = SIM.simulate(TR, cal, cfg), w = SIM.simulate(TR, cal, Object.assign({ decodeStages: 1e6 }, cfg));
  const c = SIM.simulate(TR, cal, Object.assign({ decodeStages: 4 }, cfg));
  assert.ok(Math.abs(w.usefulTps - u.usefulTps) < 1e-3 * u.usefulTps && Math.abs(w.decodeTpsMean - 180) < 1e-6 && w.ringFullFrac === 0, `wide ring ${w.usefulTps} vs ${u.usefulTps}`);
  assert.ok(c.decodeTpsMean < 180 && c.ringFullFrac > 0 && c.decodingMean > 4 && c.decodingMean * c.decodeTpsMean <= 4 * 180 * (1 + 1e-9) && c.reqPerS < u.reqPerS,
    `ring 4: speed ${c.decodeTpsMean} full ${c.ringFullFrac} decoding ${c.decodingMean}`);
  // decodeCurve 'm3': decodeTps is the speed at 100k context, within 0.6% of the measured curve's shape; longer
  // contexts decode slower, so AgentX sessions average below decodeTps; a ring that never fills changes nothing
  const m3 = Object.assign({}, SIM.DEFAULTS, { decodeCurve: 'm3' });
  assert.ok(Math.abs(SIM.decodeSpeed(m3, 100000) - 180) < 1e-9 && SIM.decodeSpeed(m3, 550000) < SIM.decodeSpeed(m3, 8000) && SIM.decodeSpeed(Object.assign({}, SIM.DEFAULTS, { decodeCurve: 'flat' }), 550000) === 180 && SIM.DEFAULTS.decodeCurve === 'm3');
  for (const [x, t] of SIM.M3_DECODE_TSU) assert.ok(Math.abs(SIM.decodeSpeed(m3, x) * 98 / 180 / t - 1) < 0.006, `curve at ${x}`);
  const cu = SIM.simulate(TR, cal, Object.assign({ decodeCurve: 'm3' }, cfg)), cw = SIM.simulate(TR, cal, Object.assign({ decodeCurve: 'm3', decodeStages: 1e6 }, cfg));
  assert.ok(cu.decodeTpsMean < 180 && cu.decodeTpsMean > 100 && Math.abs(cw.usefulTps - cu.usefulTps) < 1e-3 * cu.usefulTps, `m3 mean speed ${cu.decodeTpsMean}`);
  // starvation causes: shares of the window, summing to at most 1; none from slots without a slot limit
  const n = SIM.simulate(TR, cal, Object.assign({}, cfg, { decodeSlots: 0 }));
  for (const r of [u, c, n]) {
    const f = [r.pfStarvedSlotFrac, r.pfStarvedIdleFrac, r.sendBlockFrac];
    assert.ok(f.every((x) => x >= 0 && x <= 1) && f[0] + f[1] <= 1 + 1e-9, `starved ${f}`);
  }
  assert.ok(c.pfStarvedSlotFrac > 0 && n.pfStarvedSlotFrac === 0 && n.sendBlockFrac > 0, 'slot limits starve prefill');
}

// 10c. batchChunksPerRequest L: rounds over the queue, up to L chunks per request per round.
//      L = 1 is an even split: within a batch, every request not finishing in it holds the largest share (within one
//      chunk), and a request alone takes the whole budget. L = 4: at most one request per batch gets fewer than 4
//      chunks without finishing (the one the budget ran out on). L = 0 (default) is greedy.
{
  const C = 512, cfg = Object.assign({ cache: 'inf', batch: true, budget: 8192 }, base, { concurrency: 256, chunk: C });
  let bad = 0, multi = 0, alone = 0;
  const one = SIM.simulate(TR, cal, Object.assign({ batchChunksPerRequest: 1 }, cfg), { onChunk: (t, segs) => {
    const tot = segs.reduce((a, s) => a + s.npad, 0), mx = Math.max(...segs.map((s) => s.npad));
    if (tot > 8192) bad++;
    if (segs.length > 1) { multi++; for (const s of segs) if (!s.last && s.npad < mx - C) bad++; }
    else if (!segs[0].last && segs[0].npad === 8192) alone++;
  } });
  let cut = 0;
  SIM.simulate(TR, cal, Object.assign({ batchChunksPerRequest: 4 }, cfg), { onChunk: (t, segs) => {
    if (segs.filter((s) => !s.last && s.npad < 4 * C).length > 1) cut++;
  } });
  const g = SIM.simulate(TR, cal, cfg), z = SIM.simulate(TR, cal, Object.assign({ batchChunksPerRequest: 0 }, cfg));
  assert.ok(bad === 0 && multi > 0 && alone > 0 && cut === 0 && one.avgSegsPerChunk > g.avgSegsPerChunk, `L=1 bad ${bad} multi ${multi} alone ${alone}; L=4 cut ${cut}`);
  assert.ok(g.usefulTps === z.usefulTps && SIM.makePlan(Object.assign({}, cfg, { batchChunksPerRequest: 1.5 }), cal).errors.length > 0);
}

// 11. auto layer split: the DP finds the least bottleneck (checked against every split for a few stages), and on the
//     real costs it is no worse than the old auto split anywhere, matches the hand split [2,3,4x13,3] on 16x[2,4]
//     (dense layers sharing stages) and keeps 32x[2,4] at its old bottleneck
{
  const L = 60, D = 3;
  const worstOf = (c, m) => { let w = 0, st = 0; c.forEach((n, s) => { const nd = Math.max(0, Math.min(D, st + n) - st); st += n; w = Math.max(w, nd * m.td + (n - nd) * m.tm + m.ov + (s === 0 ? m.embed : 0)); }); return w; };
  const rnd = (() => { let a = 7; return () => { a = (a * 16807) % 2147483647; return a / 2147483647; }; })();
  for (let t = 0; t < 6; t++) for (const S of [2, 3, 4]) {
    const m = { tm: 5 + 10 * rnd(), td: 2 + 60 * rnd(), ov: 2 * rnd(), embed: 3 * rnd() };
    const c = SIM.splitLayers({ split: 'auto' }, S, m);
    assert.ok(c.length === S && c.every((n) => n >= 1) && c.reduce((x, y) => x + y, 0) === L, `split ${c}`);
    let best = Infinity;
    const rec = (pre, left) => {
      if (pre.length === S - 1) { best = Math.min(best, worstOf(pre.concat([left]), m)); return; }
      for (let n = 1; n <= left - (S - 1 - pre.length); n++) rec(pre.concat([n]), left - n);
    };
    rec([], L);
    assert.ok(worstOf(c, m) <= best * (1 + 1e-9), `S=${S}: auto ${worstOf(c, m)} > best ${best}`);
  }
  const worst = (cfg) => { const p = plan(Object.assign({ cache: 'inf' }, cfg)), o = []; SIM.chunkStageMs(p, 2048, [{ n: 2048, na: 2048, k: 131072, cap: 0 }], o); return Math.max(...o); };
  const rep = (n, x) => Array(n).fill(x);
  const old = { // the old auto splits (each dense layer on its own stage)
    '8x2,4': [2, 4].concat(rep(6, 9)), '16x2,4': [1, 1, 1].concat(rep(8, 4), rep(5, 5)), '24x2,4': [1, 1, 1].concat(rep(6, 2), rep(15, 3)),
    '32x2,4': [1, 1, 1, 1].concat(rep(28, 2)), '8x4,2': [4].concat(rep(7, 8)), '16x4,2': [2, 3, 3].concat(rep(13, 4)),
    '24x4,2': [1, 1, 1].concat(rep(6, 2), rep(15, 3)), '32x4,2': [1, 1, 1, 1].concat(rep(28, 2)),
  };
  for (const [key, split] of Object.entries(old)) {
    const [S, mesh] = [Number(key.split('x')[0]), key.split('x')[1].split(',').map(Number)];
    const cfg = { galaxies: S / 4, stages: S, mesh };
    assert.ok(worst(cfg) <= worst(Object.assign({ split }, cfg)) + 1e-9, `${key}: auto ${worst(cfg)} > old ${worst(Object.assign({ split }, cfg))}`);
  }
  const g16 = { galaxies: 4, stages: 16, mesh: [2, 4] }, hand = [2, 3].concat(rep(13, 4), [3]);
  assert.ok(worst(g16) <= worst(Object.assign({ split: hand }, g16)) + 1e-9 && worst(g16) < 0.85 * worst(Object.assign({ split: old['16x2,4'] }, g16)), `16x[2,4] ${worst(g16)}`);
  assert.ok(plan(g16).counts[0] >= 2, 'dense layers share a stage on 16x[2,4]');
}

// 12. owner placement (tile padding): tokens go to the SP rank that owns their KV row under the block-cyclic layout
{
  const W = (load, p, m, cap, blk, sp, apply) => { const t = SIM.ownerWalk(load, p, m, cap, blk, sp, apply); return [t, load]; };
  // any C-length window gives every rank C/SP rows (#57636's mid-slab chunk); shorter ones are uneven
  assert.deepStrictEqual(W([0, 0, 0, 0], 5088, 2048, 1e9, 512, 4, true), [2048, [512, 512, 512, 512]]);
  assert.deepStrictEqual(W([0, 0, 0, 0], 0, 1600, 1e9, 512, 4, true), [1600, [512, 512, 512, 64]]);
  // a full rank stops the segment (it is contiguous); without apply the loads are untouched
  assert.deepStrictEqual(W([480, 0, 0, 0], 0, 1024, 512, 512, 4, false), [32, [480, 0, 0, 0]]);
  // no all-to-all: an owner-placed layer costs what chunk padding does for the same segments
  const segs = [{ n: 2048, na: 2000, k: 30000, cap: 0 }];
  const po = plan({ reqPad: 'tile', placement: 'owner' }), pf = plan({ reqPad: 'chunk' }), pe = plan({ reqPad: 'tile' });
  for (const kind of ['moe', 'dense']) {
    assert.strictEqual(po.layer(kind, 2048, segs), pf.layer(kind, 2048, segs), kind);
    assert.ok(pe.layer(kind, 2048, segs) > po.layer(kind, 2048, segs), `${kind}: even split pays the all-to-all`);
  }
  // tile padding fixes the KV slab at 128*SP whatever chunk is asked for (MSA needs 128-row KV blocks per rank)
  assert.strictEqual(plan({ reqPad: 'tile', chunk: 5120 }).cfg.chunk, 128 * 2);
  // replay: padding (rank imbalance included) between the even split (none) and chunk padding
  const pad = (cfg) => SIM.simulate(TR, cal, Object.assign({ cache: 'inf' }, base, { concurrency: 64 }, cfg)).padFrac;
  const [pF, pE, pO] = [pad({ reqPad: 'chunk' }), pad({ reqPad: 'tile' }), pad({ reqPad: 'tile', placement: 'owner' })];
  assert.ok(pE < pO && pO < pF, `pad even ${pE} owner ${pO} fixed ${pF}`);
  const rb = SIM.simulate(TR, cal, Object.assign({ cache: 'inf', batch: true, budget: 16384, reqPad: 'tile', placement: 'owner' }, base, { concurrency: 64 }));
  assert.ok(rb.done > 0 && rb.avgChunkTok <= 16384 && rb.maxUtil <= 1 + 1e-9);
}

console.log('test_model: all checks passed');
