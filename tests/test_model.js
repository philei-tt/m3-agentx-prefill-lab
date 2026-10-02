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

// 7. replay: an SSD tier behind static slots keeps evicted slots' KV, so reads happen and the hit rate improves; the
//    infinite cache has no tier
{
  const cfg = Object.assign({ cache: 'slots', chunk: 2048, unaligned: true }, base, { concurrency: 64 });
  const a = SIM.simulate(TR, cal, cfg), b = SIM.simulate(TR, cal, Object.assign({ hostTier: true }, cfg));
  assert.ok(SIM.makePlan(Object.assign({ hostTier: true }, cfg), cal).hostTok > 0 && b.hostTok > 0, 'no SSD reads behind slots');
  assert.ok(b.hitRate > a.hitRate, `slots + SSD hit ${b.hitRate} <= ${a.hitRate}`);
  assert.strictEqual(SIM.makePlan(Object.assign({ hostTier: true }, cfg, { cache: 'inf' }), cal).hostTok, 0);
}

console.log('test_model: all checks passed');
