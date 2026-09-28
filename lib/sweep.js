/* sweep.js - concurrency sweep and goodput rule, shared by the study (lib/pool.js) and the web page (inlined by
 * build_artifact.js). One copy on purpose: the page and the study must measure goodput the same way.
 *
 * sweep(concs, slo, run, opts) -> points
 *   1. ascending grid, early stop once well past the knee (p90 > stopFactor x SLO and throughput no longer growing)
 *   2. extend the grid (x1.5, up to `extend` times) while its last point still meets the SLO, so a configuration that
 *      is still under the SLO at the top of the grid (typically an infinite cache) is not cut off
 *   3. bisect the SLO crossing (`refine` runs, log concurrency) between the highest passing concurrency below the
 *      first failure and that failure; never re-runs a concurrency
 * summarize(points, slo) -> {goodput, at, peak, atEdge}
 *   goodput = best useful tok/s among points with finite p90 TTFT <= SLO, linearly interpolated (in log p90) to the
 *   SLO crossing when the next point has higher throughput. atEdge = the goodput point is the highest concurrency
 *   simulated and still passes (goodput is a lower bound).
 */
(function (root) {
  'use strict';
  const passes = (p, slo) => !!p && !p.error && Number.isFinite(p.ttftP90) && p.ttftP90 <= slo;
  const ok = (p) => !!p && !p.error && Number.isFinite(p.usefulTps);

  function summarize(points, slo) {
    let best = null, peak = null, g = 0;
    const pts = points.filter(ok).sort((a, b) => a.conc - b.conc);
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (!peak || p.usefulTps > peak.usefulTps) peak = p;
      if (!passes(p, slo)) continue;
      if (!best || p.usefulTps > best.usefulTps) best = p;
      g = Math.max(g, p.usefulTps);
      const q = pts[i + 1];
      if (q && Number.isFinite(q.ttftP90) && q.ttftP90 > slo && q.usefulTps > p.usefulTps) {
        const lp = Math.log(Math.max(1e-3, p.ttftP90));
        const w = (Math.log(slo) - lp) / (Math.log(q.ttftP90) - lp);
        g = Math.max(g, p.usefulTps + Math.min(1, Math.max(0, w)) * (q.usefulTps - p.usefulTps));
      }
    }
    const top = pts.length ? pts[pts.length - 1] : null;
    return { goodput: g, at: best, peak, atEdge: !!(best && top && best === top) };
  }

  function sweep(concs, slo, run, opts) {
    const o = Object.assign({ stopFactor: 4, refine: 4, extend: 3, extendFactor: 1.5, maxConc: 16384 }, opts || {});
    const pts = [];
    const seen = new Map();
    const one = (c, tag) => {
      if (seen.has(c)) return seen.get(c);
      const m = run(c, tag); seen.set(c, m); pts.push(m); return m;
    };
    // 1. grid with early stop
    let over = 0, prevU = -1, stopped = false;
    for (const c of concs) {
      const m = one(c, 'grid');
      if (!ok(m)) { stopped = true; break; }
      if (m.ttftP90 > o.stopFactor * slo && m.usefulTps <= prevU * 1.02) over++; else over = 0;
      prevU = Math.max(prevU, m.usefulTps);
      if (over >= 2) { stopped = true; break; }
    }
    // 2. extend while the top point still passes
    for (let e = 0; e < o.extend && !stopped; e++) {
      const top = pts.filter(ok).sort((a, b) => b.conc - a.conc)[0];
      if (!passes(top, slo)) break;
      const c = Math.round(top.conc * o.extendFactor / 8) * 8;
      if (c > o.maxConc || c <= top.conc) break;
      if (!ok(one(c, 'extend'))) break;
    }
    // 3. bisect the SLO crossing
    for (let i = 0; i < o.refine; i++) {
      const s = pts.filter(ok).sort((a, b) => a.conc - b.conc);
      let best = null;
      for (const p of s) if (passes(p, slo) && (!best || p.usefulTps >= best.usefulTps)) best = p;
      if (!best) break;
      const hi = s.find((p) => p.conc > best.conc && !passes(p, slo));
      if (!hi) break;
      let lo = best;
      for (const p of s) if (p.conc > lo.conc && p.conc < hi.conc && passes(p, slo)) lo = p;
      const mid = Math.round(Math.sqrt(lo.conc * hi.conc) / 8) * 8;
      if (mid <= lo.conc || mid >= hi.conc || seen.has(mid)) break;
      if (!ok(one(mid, 'refine'))) break;
    }
    return pts;
  }

  const API = { passes, summarize, sweep };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.M3SWEEP = API;
})(typeof self !== 'undefined' ? self : this);
