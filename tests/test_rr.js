#!/usr/bin/env node
// Round-robin scheduler (policy 'rr', tt-d-gen PrefillQueue semantics). Run: node tests/test_rr.js
'use strict';
const assert = require('assert');
const SIM = require('../sim_core.js');
const { loadAll } = require('../run.js');
const P = require('../lib/paths.js');

const { TR, cal } = loadAll(P.DATA);
// record every chunk as a list of [request id, tokens, first, last, copy-in (round robin on the pool), lane];
// ids are per request object
function trace(cfg) {
  const ids = new Map(), chunks = [];
  const id = (q) => { if (!ids.has(q)) ids.set(q, ids.size); return ids.get(q); };
  const r = SIM.simulate(TR, cal, Object.assign({ concurrency: 128, duration: 300, chunk: 2048 }, cfg), {
    onChunk: (t, segs) => chunks.push(segs.map((s) => [id(s.q), s.n, s.first, s.last, s.miss, s.lane])),
  });
  assert.ok(!r.error && r.done > 0, `${JSON.stringify(cfg)}: ${r.error || 'no request finished'}`);
  return { r, chunks };
}
// a request interleaved with others: X ... Y ... X with X unfinished in between
function interleaved(chunks) {
  const open = new Set(); let n = 0, prev = -1;
  for (const ch of chunks) for (const [q, , first, last] of ch) {
    if (!first && prev !== q && open.has(q)) n++;
    prev = q; if (last) open.delete(q); else open.add(q);
  }
  return n;
}
// round-robin fairness: between two consecutive turns of a request, no other request has two turns (segments in
// pop order: a batch's segments are in the order the scheduler took them from the queue)
function fair(chunks) {
  const seen = new Map(); // request -> requests served since its last turn
  for (const ch of chunks) for (const [q, , , last] of ch) {
    for (const [x, since] of seen) if (x !== q) { if (since.has(q)) return false; since.add(q); }
    if (last) seen.delete(q); else seen.set(q, new Set());
  }
  return true;
}
const once = (chunks) => chunks.every((ch) => new Set(ch.map((s) => s[0])).size === ch.length);
// requests in progress (started, not finished) at every chunk
function maxOpen(chunks) {
  const open = new Set(); let m = 0;
  for (const ch of chunks) {
    for (const [q] of ch) open.add(q);
    m = Math.max(m, open.size);
    for (const [q, , , last] of ch) if (last) open.delete(q);
  }
  return m;
}

// 1. unbatched 'rtc' runs every request to completion; round robin gives one chunk per turn and rotates
for (const cache of ['slots', 'inf']) {
  const f = trace({ cache }), r = trace({ cache, policy: 'rr' });
  assert.strictEqual(interleaved(f.chunks), 0, `rtc/${cache} interleaves requests`);
  assert.ok(interleaved(r.chunks) > 0, `rr/${cache} never interleaves`);
  assert.ok(r.chunks.every((ch) => ch.length === 1 && ch[0][1] <= 2048), `rr/${cache}: one chunk per turn`);
  assert.ok(fair(r.chunks), `rr/${cache} is not round robin`);
}

// 2. batched round robin: a popped request fills as many C-units as it can, as one run (never split within a batch),
//    and the rotation is fair
{
  const C = 1024, budget = 8192;
  const { chunks } = trace({ cache: 'pool', policy: 'rr', batch: true, chunk: C, budget, layout: 'fixed', unaligned: true });
  assert.ok(once(chunks), 'a request appears twice in one batch');
  assert.ok(chunks.every((ch) => ch.reduce((a, s) => a + Math.ceil(s[1] / C) * C, 0) <= budget), 'batch over budget');
  assert.ok(chunks.some((ch) => ch.length > 1) && fair(chunks), 'batched rr is not round robin');
}

// 3. round robin on pool lanes
const pool = { cache: 'pool', laneScope: 'stage', copyMode: 'double', policy: 'rr' };
const batched = Object.assign({ batch: true, chunk: 1024, budget: 8192, layout: 'fixed' }, pool);
// 3a. lane count: auto = stages without batching, chunk units per batch with batching; rrLanes sets it
assert.strictEqual(SIM.makePlan(Object.assign({ chunk: 2048 }, pool), cal).lanes, 16);
assert.strictEqual(SIM.makePlan(batched, cal).lanes, 8);
assert.strictEqual(SIM.makePlan(Object.assign({}, batched, { rrLanes: 3 }), cal).lanes, 3);
// a variable-layout batch has no chunk units: the lane count (the most requests per batch) must be given
const varBatched = Object.assign({}, batched, { layout: 'var' });
assert.ok(SIM.makePlan(varBatched, cal).errors.some((e) => e.includes('rrLanes')), 'var-layout batching without rrLanes');
assert.ok(!SIM.makePlan(Object.assign({}, varBatched, { rrLanes: 5 }), cal).errors.length);
assert.ok(trace(Object.assign({}, varBatched, { rrLanes: 5 })).chunks.every((ch) => ch.length <= 5), 'var layout: more requests per batch than lanes');
assert.strictEqual(SIM.makePlan(Object.assign({ lanes: 5 }, pool, { policy: 'rtc' }), cal).lanes, 5, 'other policies keep `lanes`');
// 3b. the lane count bounds the requests per batch, not the requests in progress (the partial KV is in the pool);
//     each request of a batch has its own lane
{
  const { chunks } = trace(Object.assign({}, batched, { rrLanes: 3 }));
  assert.ok(chunks.every((ch) => ch.length <= 3 && new Set(ch.map((x) => x[5])).size === ch.length), 'lanes per batch');
  assert.ok(maxOpen(chunks) > 3, 'in-progress requests capped by lanes');
  assert.ok(once(chunks) && fair(chunks));
}
// 3c. copy-in is skipped only when the lane still holds the request's context (the request used it last); a
//     request's first turn always copies its cached prefix in
for (const cfg of [Object.assign({ chunk: 2048 }, pool), Object.assign({}, batched, { rrLanes: 4 })]) {
  const m = trace(Object.assign({}, cfg, { concurrency: 160 }));
  const holder = new Map(), laneOf = new Map(); let reused = 0, missed = 0;
  for (const ch of m.chunks) for (const [q, , first, , miss, lane] of ch) {
    if (first) assert.ok(miss, 'first turn without a copy-in');
    if (!miss) { assert.ok(laneOf.get(q) === lane && holder.get(lane) === q, 'copy-in skipped on a lane that does not hold the context'); reused++; } else if (!first) missed++;
    holder.set(lane, q); laneOf.set(q, lane);
  }
  assert.ok(reused > 0 && missed > 0, `lane reuse ${reused}, misses ${missed}`);
  assert.ok(m.r.rrLaneReuse > 0 && m.r.rrLaneReuse < 1 && m.r.rrCopyInTps > 0);
}

// 4. the default policy is unchanged by the round-robin code (same results as without the option set)
{
  const a = SIM.simulate(TR, cal, { concurrency: 64, duration: 300, chunk: 2048 });
  const b = SIM.simulate(TR, cal, { concurrency: 64, duration: 300, chunk: 2048, policy: 'rtc', rrLanes: 4 });
  assert.strictEqual(a.usefulTps, b.usefulTps); assert.strictEqual(a.events, b.events);
}

console.log('test_rr: all checks passed');
