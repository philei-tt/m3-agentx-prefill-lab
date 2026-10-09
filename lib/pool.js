// Worker pool: evaluates configurations over an ascending concurrency grid (early stop past the knee).
'use strict';
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const path = require('path'), os = require('os');

const KEEP = ['usefulTps', 'processedTps', 'newTps', 'inTps', 'hitTps', 'reqPerS', 'ttftP50', 'ttftP90', 'ttftP99', 'ttftMean', 'hitRate', 'infHitRate',
  'reprefillFrac', 'padFrac', 'alignLossFrac', 'avgChunkTok', 'avgSegsPerChunk', 'maxUtil', 'done', 'warmupS', 'laneWaitMean',
  'hostTok', 'ssdTok', 'hostReadTps', 'ssdReadTps', 'pcieH2DUtil', 'pcieD2HUtil', 'ssdUtil', 'slotEvictions', 'error', 'warmupTimeout', 'eventCap',
  'outTps', 'decSlotsMean', 'decSlotsMax', 'decodingMean', 'decWaitPerS', 'decWaitMean', 'decodeTpsMean', 'ringFullFrac', 'pfStarvedSlotFrac', 'pfStarvedIdleFrac', 'sendBlockFrac',
  'outDecTps', 'decHitRate', 'migTps', 'decHostReadTps', 'decSsdReadTps', 'decPcieH2DUtil', 'decPcieD2HUtil', 'decSsdUtil', 'decPoolTokMean', 'decStartDelayMean', 'decSlotWaitFrac', 'tsuP10', 'tsuP50', 'decParkPerS', 'decParkedMean'];

// goodput rule and concurrency sweep live in lib/sweep.js (shared with the web page)
const { summarize, sweep } = require('./sweep.js');

class Pool {
  constructor(n, data) {
    this.n = n || Math.min(32, os.cpus().length); this.data = data || require('./paths.js').DATA;
    this.workers = []; this.queue = []; this.idle = [];
    for (let i = 0; i < this.n; i++) {
      const w = new Worker(__filename, { workerData: { data: this.data } });
      w.on('message', (m) => { if (m.ready) { this.idle.push(w); this.drain(); return; } const cb = w._cb; w._cb = null; this.idle.push(w); cb(m); this.drain(); });
      w.on('error', (e) => { console.error('worker error', e); });
      this.workers.push(w);
    }
  }
  drain() { while (this.idle.length && this.queue.length) { const w = this.idle.pop(); const t = this.queue.shift(); w._cb = t.cb; w.postMessage(t.job); } }
  run(job) { return new Promise((res) => { this.queue.push({ job, cb: res }); this.drain(); }); }
  // job: {id, cfg, concs, slo, stopFactor, opts} (opts: extra lib/sweep.js options, e.g. {refinePeak: 3})
  evalCfg(id, cfg, concs, slo = 10, stopFactor = 4, opts = {}) { return this.run({ id, cfg, concs, slo, stopFactor, opts }); }
  close() { for (const w of this.workers) w.terminate(); }
}

if (!isMainThread) {
  const SIM = require('../sim_core.js');
  const { loadAll } = require('../run.js');
  const { TR, cal } = loadAll(workerData.data);
  parentPort.on('message', (job) => {
    const t0 = Date.now();
    const one = (c) => {
      let r;
      try { r = SIM.simulate(TR, cal, Object.assign({}, job.cfg, { concurrency: c })); } catch (e) { r = { error: String(e && e.stack || e) }; }
      const m = { conc: c };
      for (const k of KEEP) if (r[k] !== undefined) m[k] = r[k];
      return m;
    };
    const pts = sweep(job.concs, job.slo, one, Object.assign({ stopFactor: job.stopFactor }, job.opts || {}));
    const plan = SIM.planSummary(SIM.makePlan(job.cfg, cal));
    parentPort.postMessage({ id: job.id, cfg: job.cfg, points: pts, plan, wallMs: Date.now() - t0 });
  });
  parentPort.postMessage({ ready: true });
}

module.exports = { Pool, summarize, sweep };
