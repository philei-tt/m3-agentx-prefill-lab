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
  const pf = plan({ attn: 'fused', batch: true, layout: 'var', budget: 16384 }), ps = plan({ attn: 'seq', batch: true, layout: 'var', budget: 16384 });
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
assert.ok(oom({ cache: 'slots', batch: true, budget: 8192, layout: 'var', stages: 4, mesh: [8, 4] }), 'var layout: budget / 32SP per batch');
for (const cache of ['pool', 'paging', 'inf']) assert.ok(!plan({ cache, batch: true }).errors.length, cache);

// 5. replay: paging with an unbounded pool behaves exactly like the infinite cache
const base = { concurrency: 24, duration: 600, chunk: 2048 };
const a = SIM.simulate(TR, cal, Object.assign({ cache: 'inf' }, base));
const b = SIM.simulate(TR, cal, Object.assign({ cache: 'paging', reserveGB: -1e6 }, base)); // absurd capacity
assert.strictEqual(a.done, b.done); assert.ok(Math.abs(a.usefulTps - b.usefulTps) < 1e-6 * a.usefulTps + 1e-9);

// 6. replay: fixed-layout batching packs one segment per request per chunk, so chunks carry several requests
const r = SIM.simulate(TR, cal, Object.assign({ cache: 'inf', batch: true, budget: 16384, layout: 'fixed' }, base, { concurrency: 64 }));
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

// 10b. decodeConcurrency: never more requests decoding than the cap; requests wait for a position holding their
//      slot; slots <= concurrency changes nothing
{
  const cfg = Object.assign({ cache: 'pool', chunk: 512, batch: true, budget: 8192 }, base, { concurrency: 128 });
  const a = SIM.simulate(TR, cal, Object.assign({ decodeSlots: 8 }, cfg)), b = SIM.simulate(TR, cal, Object.assign({ decodeSlots: 8, decodeConcurrency: 8 }, cfg));
  const c = SIM.simulate(TR, cal, Object.assign({ decodeSlots: 16, decodeConcurrency: 4 }, cfg));
  assert.ok(a.usefulTps === b.usefulTps && b.runWaitPerS === 0, 'slots <= concurrency is unchanged');
  assert.ok(c.decodingMean <= 4 + 1e-9 && c.runWaitPerS > 0 && c.runWaitMean > 0 && c.decSlotsMax <= 16, `decoding ${c.decodingMean} runWait ${c.runWaitPerS}`);
  // starvation causes: shares of the window, summing to at most 1; none attributed to decode without decode limits
  const u = SIM.simulate(TR, cal, cfg);
  for (const r of [a, c, u]) {
    const f = [r.pfStarvedSlotFrac, r.pfStarvedDecodeFrac, r.pfStarvedIdleFrac, r.sendBlockFrac];
    assert.ok(f.every((x) => x >= 0 && x <= 1) && f[0] + f[1] + f[2] <= 1 + 1e-9, `starved ${f}`);
  }
  assert.ok(a.pfStarvedSlotFrac > 0 && c.pfStarvedDecodeFrac + c.pfStarvedSlotFrac > 0, 'decode limits starve prefill');
  assert.ok(u.pfStarvedSlotFrac === 0 && u.pfStarvedDecodeFrac === 0 && u.sendBlockFrac > 0);
}

// 11. batchMaxChunks: a request never takes more than that many chunk units of one batch; 0 is the default (no
//     limit) and leaves results unchanged; with a cap of 1, unloaded TTFT matches no batching closely
{
  const cfg = Object.assign({ cache: 'inf', chunk: 2048, batch: true, budget: 8192 }, base, { concurrency: 16 });
  let maxSeg = 0;
  const one = SIM.simulate(TR, cal, Object.assign({ batchMaxChunks: 1 }, cfg), { onChunk: (t, segs) => { for (const s of segs) maxSeg = Math.max(maxSeg, s.n); } });
  const d = SIM.simulate(TR, cal, cfg), z = SIM.simulate(TR, cal, Object.assign({ batchMaxChunks: 0 }, cfg));
  const nb = SIM.simulate(TR, cal, Object.assign({}, cfg, { batch: false }));
  assert.ok(maxSeg === 2048 && d.ttftP90 === z.ttftP90 && d.usefulTps === z.usefulTps, `max segment ${maxSeg}`);
  assert.ok(one.ttftP90 < d.ttftP90 && Math.abs(one.ttftP90 - nb.ttftP90) < 0.1 * nb.ttftP90, `p90 cap1 ${one.ttftP90} unlimited ${d.ttftP90} no batch ${nb.ttftP90}`);
  assert.ok(SIM.makePlan(Object.assign({}, cfg, { batchMaxChunks: 1.5 }), cal).errors.length > 0);
}

console.log('test_model: all checks passed');
