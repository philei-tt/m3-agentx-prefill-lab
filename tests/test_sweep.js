#!/usr/bin/env node
// Unit checks for lib/sweep.js (concurrency sweep + goodput rule). Run: node tests/test_sweep.js
'use strict';
const assert = require('assert');
const S = require('../lib/sweep.js');

// synthetic cliff: throughput rises until C=1800 then collapses; p90 crosses 10 s at C=1700
const cliff = (c) => ({ conc: c, usefulTps: c < 1800 ? c * 90 : c * 40, ttftP90: c < 1700 ? 2 + c / 1000 : 30 });
const p1 = S.sweep([256, 512, 1024, 1536, 2048, 3072], 10, cliff);
const c1 = p1.map((p) => p.conc);
assert.strictEqual(new Set(c1).size, c1.length, 'no concurrency simulated twice');
assert.ok(S.summarize(p1, 10).at.conc >= 1600 && S.summarize(p1, 10).at.conc < 1700, 'bisection lands next to the crossing');

// never failing: the grid is extended and the result is flagged as a lower bound
const flat = (c) => ({ conc: c, usefulTps: c * 10, ttftP90: 1 });
const p2 = S.sweep([256, 512, 1024], 10, flat);
assert.strictEqual(p2.length, 6);
assert.ok(S.summarize(p2, 10).atEdge);

// missing TTFT (no request finished) never passes
assert.strictEqual(S.summarize([{ conc: 8, usefulTps: 5, ttftP90: null }, { conc: 16, usefulTps: 3, ttftP90: NaN }], 10).goodput, 0);

// a passing low-throughput point between the best point and the first failure must not stall the bisection
const dip = (c) => ({ conc: c, usefulTps: c === 1200 ? 50 : c * 10, ttftP90: c >= 1500 ? 50 : 5 });
const c3 = S.sweep([1000, 1200, 2000], 10, dip).map((p) => p.conc);
assert.strictEqual(new Set(c3).size, c3.length);

// interpolation to the SLO crossing only when the failing point has higher throughput
const g = S.summarize([{ conc: 100, usefulTps: 100, ttftP90: 5 }, { conc: 200, usefulTps: 200, ttftP90: 20 }], 10).goodput;
assert.ok(g > 100 && g < 200);
assert.strictEqual(S.summarize([{ conc: 100, usefulTps: 100, ttftP90: 5 }, { conc: 200, usefulTps: 50, ttftP90: 20 }], 10).goodput, 100);

// decode-speed SLO (tsuMin): a point must also have tsuP10 >= tsuMin; no interpolation towards a point that fails it
{
  const pts = [{ conc: 100, usefulTps: 100, ttftP90: 5, tsuP10: 80 }, { conc: 200, usefulTps: 200, ttftP90: 8, tsuP10: 40 }, { conc: 300, usefulTps: 250, ttftP90: 20, tsuP10: 30 }];
  assert.strictEqual(S.summarize(pts, 10).goodput > 200, true);
  assert.strictEqual(S.summarize(pts, 10, 50).goodput, 100);
  assert.ok(!S.passes(pts[1], 10, 50) && S.passes(pts[1], 10) && !S.passes({ conc: 1, usefulTps: 1, ttftP90: 1 }, 10, 50));
  const fall = (c) => ({ conc: c, usefulTps: c, ttftP90: 1, tsuP10: 200 - c / 10 });
  const r = S.summarize(S.sweep([256, 512, 1024, 2048], 10, fall, { tsuMin: 50 }), 10, 50);
  assert.ok(r.at.conc <= 1500 && r.at.conc >= 1024, `tsu bisection at ${r.at.conc}`);
}

console.log('test_sweep: all checks passed');
