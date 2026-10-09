/* sweep.js - concurrency sweep and goodput rule, shared by the study (lib/pool.js) and the web page (inlined by
 * build_artifact.js). One copy on purpose: the page and the study must measure goodput the same way.
 *
 * sweep(concs, slo, run, opts) -> points
 *   1. ascending grid, early stop once well past the knee (p90 > stopFactor x SLO and throughput no longer growing)
 *   2. extend the grid (x1.5, up to `extend` times) while its last point still meets the SLO, so a configuration that
 *      is still under the SLO at the top of the grid (typically an infinite cache) is not cut off
 *   3. bisect the SLO crossing (`refine` runs, log concurrency) between the highest passing concurrency below the
 *      first failure and that failure; never re-runs a concurrency
 * summarize(points, slo, tsuMin) -> {goodput, at, peak, atEdge}
 *   goodput = best useful tok/s among points with finite p90 TTFT <= SLO, linearly interpolated (in log p90) to the
 *   SLO crossing when the next point has higher throughput. atEdge = the goodput point is the highest concurrency
 *   simulated and still passes (goodput is a lower bound).
 * tsuMin (optional, 0 = none; sweep: opts.tsuMin): a decode-speed SLO. A point also needs 90% of its decode sessions
 *   at or above tsuMin tokens/s/u (tsuP10 >= tsuMin); no interpolation towards a point that fails it.
 */
(function (root) {
  'use strict';
  const tsuOk = (p, tsuMin) => !(tsuMin > 0) || (Number.isFinite(p.tsuP10) && p.tsuP10 >= tsuMin);
  const passes = (p, slo, tsuMin) => !!p && !p.error && Number.isFinite(p.ttftP90) && p.ttftP90 <= slo && tsuOk(p, tsuMin);
  const ok = (p) => !!p && !p.error && Number.isFinite(p.usefulTps);

  function summarize(points, slo, tsuMin) {
    let best = null, peak = null, g = 0;
    const pts = points.filter(ok).sort((a, b) => a.conc - b.conc);
    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      if (!peak || p.usefulTps > peak.usefulTps) peak = p;
      if (!passes(p, slo, tsuMin)) continue;
      if (!best || p.usefulTps > best.usefulTps) best = p;
      g = Math.max(g, p.usefulTps);
      const q = pts[i + 1];
      if (q && Number.isFinite(q.ttftP90) && q.ttftP90 > slo && tsuOk(q, tsuMin) && q.usefulTps > p.usefulTps) {
        const lp = Math.log(Math.max(1e-3, p.ttftP90));
        const w = (Math.log(slo) - lp) / (Math.log(q.ttftP90) - lp);
        g = Math.max(g, p.usefulTps + Math.min(1, Math.max(0, w)) * (q.usefulTps - p.usefulTps));
      }
    }
    const top = pts.length ? pts[pts.length - 1] : null;
    return { goodput: g, at: best, peak, atEdge: !!(best && top && best === top) };
  }

  function sweep(concs, slo, run, opts) {
    const o = Object.assign({ stopFactor: 4, refine: 4, extend: 3, extendFactor: 1.5, maxConc: 16384, tsuMin: 0 }, opts || {});
    const pass = (p, s) => passes(p, s, o.tsuMin);
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
      if (!pass(top, slo)) break;
      const c = Math.round(top.conc * o.extendFactor / 8) * 8;
      if (c > o.maxConc || c <= top.conc) break;
      if (!ok(one(c, 'extend'))) break;
    }
    // 3. bisect the SLO crossing
    for (let i = 0; i < o.refine; i++) {
      const s = pts.filter(ok).sort((a, b) => a.conc - b.conc);
      let best = null;
      for (const p of s) if (pass(p, slo) && (!best || p.usefulTps >= best.usefulTps)) best = p;
      if (!best) break;
      const hi = s.find((p) => p.conc > best.conc && !pass(p, slo));
      if (!hi) break;
      let lo = best;
      for (const p of s) if (p.conc > lo.conc && p.conc < hi.conc && pass(p, slo)) lo = p;
      const mid = Math.round(Math.sqrt(lo.conc * hi.conc) / 8) * 8;
      if (mid <= lo.conc || mid >= hi.conc || seen.has(mid)) break;
      if (!ok(one(mid, 'refine'))) break;
    }
    // 4. bisect around the throughput peak when goodput is the peak, not the SLO crossing: explicitly for "no SLO"
    //    sweeps (refinePeak), and automatically (refinePeakAuto) when the best passing point is not the highest
    //    passing concurrency, i.e. throughput already falls before p90 reaches the SLO. Without this, a looser SLO
    //    moves the crossing bisection past the peak and the peak is sampled only on the coarse grid, so a re-run with
    //    a looser SLO could report lower goodput. Try the log midpoints between the best point and each neighbour.
    const s0 = pts.filter((p) => ok(p) && pass(p, slo)).sort((a, b) => a.conc - b.conc);
    const b0 = s0.reduce((bi, p, j) => (p.usefulTps > s0[bi].usefulTps ? j : bi), 0);
    // the peak can also sit just below the best passing point when that point is the highest passing one but the
    // first failing point above it has lower throughput (throughput already falls where p90 crosses the SLO)
    const bestP = s0[b0], failUp = bestP && pts.filter(ok).sort((a, b) => a.conc - b.conc).find((p) => p.conc > bestP.conc && !pass(p, slo));
    const peakInside = s0.length > 1 && (b0 < s0.length - 1 || (failUp && failUp.usefulTps < bestP.usefulTps));
    const nPeak = Math.max(o.refinePeak || 0, peakInside ? (o.refinePeakAuto != null ? o.refinePeakAuto : 2) : 0);
    for (let i = 0; i < nPeak; i++) {
      const s = pts.filter((p) => ok(p) && pass(p, slo)).sort((a, b) => a.conc - b.conc);
      if (!s.length) break;
      const k = s.reduce((bi, p, j) => (p.usefulTps > s[bi].usefulTps ? j : bi), 0);
      let added = false;
      for (const nb of [s[k - 1], s[k + 1]]) {
        if (!nb) continue;
        const mid = Math.round(Math.sqrt(s[k].conc * nb.conc) / 8) * 8;
        if (mid === s[k].conc || mid === nb.conc || seen.has(mid)) continue;
        if (ok(one(mid, 'peak'))) added = true;
      }
      if (!added) break;
    }
    return pts;
  }

  const API = { passes, summarize, sweep };
  if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.M3SWEEP = API;
})(typeof self !== 'undefined' ? self : this);
