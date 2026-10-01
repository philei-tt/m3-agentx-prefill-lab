#!/usr/bin/env node
// Round-robin scheduler (policy 'rr', tt-d-gen PrefillQueue semantics). Run: node tests/test_rr.js
'use strict';
const assert = require('assert');
const SIM = require('../sim_core.js');
const { loadAll } = require('../run.js');
const P = require('../lib/paths.js');

const { TR, cal } = loadAll(P.DATA);
// record every chunk as a list of [request id, tokens, first, last]; ids are per request object
function trace(cfg) {
  const ids = new Map(), chunks = [];
  const id = (q) => { if (!ids.has(q)) ids.set(q, ids.size); return ids.get(q); };
  const r = SIM.simulate(TR, cal, Object.assign({ concurrency: 128, duration: 300, chunk: 2048 }, cfg), {
    onChunk: (t, segs) => chunks.push(segs.map((s) => [id(s.q), s.n, s.first, s.last])),
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

// 1. unbatched FCFS runs every request to completion; round robin gives one chunk per turn and rotates
for (const cache of ['slots', 'inf']) {
  const f = trace({ cache }), r = trace({ cache, policy: 'rr' });
  assert.strictEqual(interleaved(f.chunks), 0, `fcfs/${cache} interleaves requests`);
  assert.ok(interleaved(r.chunks) > 0, `rr/${cache} never interleaves`);
  assert.ok(r.chunks.every((ch) => ch.length === 1 && ch[0][1] <= 2048), `rr/${cache}: one chunk per turn`);
  assert.ok(fair(r.chunks), `rr/${cache} is not round robin`);
}

// 2. batched round robin: a popped request fills as many C-units as it can, as one run (never split within a batch),
//    and the rotation is fair
{
  const C = 1024, budget = 8192;
  const { chunks } = trace({ cache: 'slots', policy: 'rr', batch: true, chunk: C, budget, layout: 'fixed', unaligned: true });
  assert.ok(once(chunks), 'a request appears twice in one batch');
  assert.ok(chunks.every((ch) => ch.reduce((a, s) => a + Math.ceil(s[1] / C) * C, 0) <= budget), 'batch over budget');
  assert.ok(chunks.some((ch) => ch.length > 1) && fair(chunks), 'batched rr is not round robin');
}

// 3. pool lanes: 'keep' admits at most `lanes` requests in progress; 'release' frees the lane between turns, so more
//    requests are in progress but at most `lanes` share one batch
{
  const pool = { cache: 'pool', lanes: 3, laneScope: 'stage', copyMode: 'double', policy: 'rr', batch: true, chunk: 1024, budget: 8192, layout: 'fixed' };
  const k = trace(pool), rel = trace(Object.assign({}, pool, { rrLanes: 'release' }));
  assert.ok(maxOpen(k.chunks) <= 3, `keep: ${maxOpen(k.chunks)} requests in progress with 3 lanes`);
  assert.ok(rel.chunks.every((ch) => ch.length <= 3), 'release: more requests in one batch than lanes');
  assert.ok(maxOpen(rel.chunks) > 3, 'release: in-progress requests still capped by lanes');
  assert.ok(once(rel.chunks) && fair(rel.chunks));
  const cap = trace(Object.assign({}, pool, { rrLanes: 'release', rrMaxActive: 5 }));
  assert.ok(maxOpen(cap.chunks) <= 5, 'rrMaxActive not enforced');
}

// 4. the default policy is unchanged by the round-robin code (same results as without the option set)
{
  const a = SIM.simulate(TR, cal, { concurrency: 64, duration: 300, chunk: 2048 });
  const b = SIM.simulate(TR, cal, { concurrency: 64, duration: 300, chunk: 2048, policy: 'fcfs', rrLanes: 'release' });
  assert.strictEqual(a.usefulTps, b.usefulTps); assert.strictEqual(a.events, b.events);
}

console.log('test_rr: all checks passed');
