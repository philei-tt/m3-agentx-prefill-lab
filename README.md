# MiniMax-M3 prefill: AgentX traffic roofline

Traffic-level model of the M3 pipeline-prefill system on Blackhole galaxies. It replays the **AgentX 1M corpus** the way
AIPerf's `inferencex-agentx-mvp` scenario does and runs every request through a **per-op roofline** calibrated to the
16-stage hardware runs of #57827. The goal is to rank features and hyperparameters by what they do for the whole system
on realistic traffic, not by single-cell benchmarks.

* Repository: https://github.com/philei-tt/m3-agentx-prefill-lab (also mirrored in tt-metal branch `philei/m3-traffic-sim`, `models/demos/minimax_m3/traffic_sim/`)
* Hosted copy of the web app: https://claude.ai/artifact/3Yresb3qbWJQpvpQMabaq6
* Tracking issue: https://github.com/tenstorrent/tt-metal/issues/57827

## Quick start: run the web app

Needs only Node.js ≥ 18 (no `npm install`, no dependencies). The traffic, calibration and study results are in the repository.

```bash
git clone https://github.com/philei-tt/m3-agentx-prefill-lab && cd m3-agentx-prefill-lab
node server.js                      # builds dist/index.html on first start, serves http://127.0.0.1:8765
node server.js --port 9000          # another port; --host 0.0.0.0 to listen on all interfaces
```

On a remote machine, forward the port from your laptop and open http://localhost:8765:

```bash
ssh -L 8765:localhost:8765 <remote-host>     # then, on the remote host: node server.js
```

The page is self-contained: the simulator runs in your browser (Web Workers), so the server only serves one file.
Fonts come from Google Fonts when reachable, with system fallbacks otherwise.
**Copy link** (Explore tab) gives a URL such as `http://localhost:8765/?galaxies=8&batch=1&budget=5120#explore` holding every setting that differs from the defaults; opening it loads that configuration and runs the sweep.
**Metrics at** (Explore tab, or a click on a chart) shows the tiles (useful tok/s, TTFT, hit rate, input tokens and requests per hour, wasted compute, requests per batch) at any concurrency instead of the goodput point. A concurrency the sweep did not run is simulated on demand; it is added to the charts but never moves goodput. The link keeps it as `at=<concurrency>`.
`node server.js --rebuild` forces a rebuild; at startup the server also rebuilds when any input is newer than `dist/index.html`.

## Metric

**Goodput = useful tok/s at p90 TTFT ≤ 10 s**, maximised over concurrency. The sweep and the goodput rule live in one file, `lib/sweep.js`, used by the study and inlined into the web page:
* **Grid:** an ascending concurrency grid, stopped early once well past the knee.
* **Extension:** the grid is extended ×1.5 (up to 3 times) while its top point still meets the SLO. A result that still passes at the top is flagged as a lower bound, shown as "≥" on the page.
* **Cliff bisection:** the cache cliff is steep (neighbouring grid points can differ by 10%+), so the sweep bisects the SLO crossing 4 times in log concurrency, without re-running any point.
* **Interpolation:** when the first failing point has higher throughput, goodput is interpolated to the SLO crossing.

A point where no request finished (TTFT undefined) never passes. Useful tokens per request are capped at the tokens it actually prefilled.
*Useful* tokens are the tokens an infinite prefix cache would still have to prefill (`in - 64·lcp_best`). Re-prefilled tokens
(evicted, misaligned, never materialised) and padding count as processed but not useful. Each result also reports the same
configuration with an infinite cache, TTFT p50/p90, the hit rate against the ∞-cache hit rate, and the split of processed
tokens into useful, re-prefill and padding. At the goodput point it also reports hourly volume (`hourly()` in `sim_core.js`): input tokens per hour of the requests completed, split into new (prefilled, re-prefill included) and cached (prefix hit), requests completed per hour, and the revenue
those input tokens bill per hour.

**Revenue per hour** prices new tokens (cache misses, re-prefill included) at the input price and prefix hits at the
cache-read price. The defaults (`SIM.PRICE` in `sim_core.js`) are MiniMax's own endpoint for `minimax/minimax-m3` on
OpenRouter (provider "Minimax", Oct 7 2026): **$0.30 / M input, $0.06 / M cache read, $1.20 / M output**, no cache-write
charge. That is also the model-level price and most providers' (the cheapest third-party provider is about 20% lower). Override it with `--price-in` / `--price-cached` (USD per
million tokens) on `run.js`, `analyze.js` and `tools/feature_table.js`, or with the Revenue & cost fields on the web page.
* **Output tokens are not counted in revenue.** This system only prefills; decode generates the output tokens, so their
  revenue ($1.20 / M) counts only in the margin below, next to the decode galaxies' cost. AgentX requests average about
  1.1k output tokens against 120k–150k input tokens at the goodput points, so output adds roughly 10–15% to a bill.
* **A better cache bills less per token** (a hit costs a fifth of a miss); it raises revenue only through the extra
  requests the same hardware can then serve.

**Net revenue per hour** (the page's Net Revenue tile, `economics()` in `sim_core.js`) is input + output revenue of the requests completed per hour, minus
(prefill galaxies + decode galaxies) × USD per galaxy-hour. Defaults (`SIM.COST`): **$12 per galaxy-hour** (a rough
operating cost) and **16 decode galaxies**, one M3 decode instance of 64 sessions, so pair it with `decodeSlots: 64`
(unlimited slots overstate what 16 decode galaxies serve). Output revenue bills `outDecTps`, the output tokens decoded
in the window. Runs also report `outTps`, the output tokens of the requests completed in the window, which is what runs
before `outDecTps` bill. The two agree in steady state, but `outTps` runs ahead while a decode backlog grows. Study
points from before both have no margin. Override with `--price-out`, `--galaxy-usd` and
`--decode-galaxies` on `run.js`, or the Revenue & cost fields on the page.

## Files

| file | what |
|---|---|
| `sim_core.js` | **The whole model** (no dependencies): per-op roofline + calibration, stage plan and memory, KV residency (static slots / paged pool with lanes / host DRAM + SSD offload tiers / paging / ∞), and the closed-loop replay. The artifact inlines it and node requires it. |
| `prep_traffic.py` | Corpus → `traffic.bin` + `traffic.json`: agent-chain split, DAG, end-to-start delays, spawn/join, replay barriers, prefix-tree pieces. Takes 23 s on 4 cores. |
| `collect_calib.py` | Measured data → `calib_data.json` (committed): zone profiles, per-rank per-chunk-position medians of runs A/B/C, matrix tables. |
| `validate.js` | Calibration report: fitted efficiencies, per-rank stage times, and all 105 matrix cells, model vs measured. |
| `run.js` | One configuration or a concurrency sweep from the CLI (`--preset`, `--set key=value`, `--conc`, `--price-in` / `--price-cached`). |
| `study.js` | Greedy feature roadmap, leave-one-out, topology/budget/lane grid and sensitivity, over {4, 8} galaxies × {today's kernels, roofline kernels}. |
| `analyze.js`, `tools/study_detail.js` | Print study results. |
| `lib/pool.js`, `lib/sweep.js` | Worker-thread pool; the concurrency sweep and the goodput rule (shared with the page). |
| `lib/scope.js` | Complexity bin and one-line summary per feature (Roadmap and README tables). |
| `tests/` | `node tests/test_sweep.js`, `node tests/test_model.js` (or `npm test`): sweep/goodput rule and cost-model/replay regression checks. |
| `build_artifact.js`, `artifact/template.html` | Build the single-file page: the core, calibration, 4 MB of traffic as base64, the study summary and the presets. `--standalone` wraps it as a full HTML document for `server.js`. |
| `server.js`, `package.json` | Zero-dependency local web server (see Quick start); `npm start` / `npm run build` / `npm run study` are shortcuts. |
| `feature_details.js` | The per-feature explanations shown when a feature is expanded in the Roadmap table. |
| `lib/price.js` | `--price-in` / `--price-cached` / `--price-out` / `--galaxy-usd` / `--decode-galaxies` overrides of `SIM.PRICE` and `SIM.COST` for the CLIs. |
| `lib/paths.js` | Where data and results are read from: env `M3SIM_DATA` / `M3SIM_RESULTS`, else `./data` and `./results`, else the exabox scratch copies. |
| `on_node.sh` | `JOB=<slurm id> ./on_node.sh <cmd>` runs on the compute node with soft ulimits raised to the hard limits. |
| `tools/` | Helpers: `feature_table.js` (README tables), `study_detail.js`, `grid_by_topology.js`, `dump_cells.js` (per-cell stage medians), `traffic_stats.js`, `smoke.sh` (every feature path), `investigate.sh`, `decode_layout_ab.js` (decode KV layouts vs revenue). |

Data:
* `data/traffic.bin` + `traffic.json`: the preprocessed corpus, 5 MB, derived from the Apache-2.0 HF dataset.
* `data/zones/`: `parse_zone_perf.py` outputs of the per-op profiles, the inputs to `collect_calib.py`.
* `results/study.json`: the full study, which the page's Roadmap tab and presets are built from.

`dist/` (the built page) is not committed.

## How to run the model and the study

Any machine with Node ≥ 18 works. On exabox use a cpu_only Slurm allocation, never the login node; `JOB=<id> ./on_node.sh <cmd>` runs a command there with raised ulimits.

```bash
node validate.js                                                  # calibration report (model vs 105 measured cells)
node run.js --preset today-C --conc 16,64,256                     # one config over a few concurrencies
node run.js --study-preset g8_k0 --set decodeTps=90 --conc 512,1024
node study.js --workers 36                                        # full study -> results/study.json (about 23 min with `--workers 13` on 14 cores; about 8 min on 36 threads before the offload tiers)
node analyze.js && node tools/feature_table.js                    # print results / README tables
node tools/feature_table.js --price-in 0.6 --price-cached 0.06     # revenue at other prices
node run.js --preset today-C --set decodeSlots=64 --galaxy-usd 12 --decode-galaxies 16   # margin per hour
node build_artifact.js --standalone                               # rebuild dist/index.html (server.js does this itself)
# regenerate the inputs (needs numpy, the AgentX traces and the #57827 run directories):
python3 prep_traffic.py --traces <traces.jsonl> --procs 32        # data/traffic.bin + traffic.json
python3 collect_calib.py --matrix <run dir with runA/runB/runC>   # calib_data.json
```

One simulation of 1800 s of traffic at C=1024 takes 0.3–0.5 s. The full study (about 420 configurations and several thousand runs with cliff refinement) takes about 23 minutes on a 14-core laptop with `--workers 13`.

## Traffic replay (AIPerf `inferencex-agentx-mvp` semantics)

The replay rules below were ported from the AIPerf source (`ai-dynamo/aiperf` @ `c78644090ff`: `timing/strategies/agentic_replay.py`, `timing/trajectory_source.py`, `timing/branch_orchestrator.py`, `timing/replay_dependencies.py`, `dataset/loader/weka_trace.py`, `dataset/loader/weka_agent_chains.py`). `prep_traffic.py` reproduces AIPerf's streams, spawn/join/offsets, replay scopes and barrier predecessors exactly on all 393 traces (checked against AIPerf's own loader code), and the sim's t* snapshots match `TrajectorySource._snapshot_for`.

* **Corpus.** `semianalysisai/cc-traces-weka-062126`, full variant: 393 sessions, 98,827 requests, 21.6B input tokens, contexts up to 990k. The ∞-cache hit rate is 98.3%.
* **Lanes.** `concurrency` = N lanes, each replaying one tree: the root chain plus its child streams. Children are the sub-agents, each split into its own agent chains by hash-id LCP detection, and the flattened agent chains that the same detection splits off the top-level requests (`WEKA_SPLIT_FLATTENED_AGENTS`, 49% of top-level requests). 9,843 streams in all.
  * **Start.** t* is uniform between the first and the last recorded request start (AgentX uses a 0–1 ratio). Every stream live at t* resumes at its first request at/after t* and gets a primer: its last request before t*, as a real prefill with `max_tokens=1`. A branch that starts after t* is spawned later by its spawn turn. When t* is after the last root start the lane is rootless (its children drain, then it recycles). Profiling starts when all primers are done.
  * **Recycle.** A drained tree recycles to the next session, sequentially, from turn 0 with a new cache-bust. Trees share nothing.
* **Delays.** Each next request waits `max(0, t_k − t_{k−1} − api_time_{k−1})` after the live end of the previous one. The delay is **not capped**. Root-chain gaps have a median of 4.5 s, a mean of 285 s and a p99 of 51 min. The only cap is AIPerf's 10 s system-idle cap. `gapCap` exists as a knob, default off.
* **Children.** Sub-agents (and, separately, flat chains) that share a spawn turn and a join turn form one branch. The branch starts at its earliest marker; it overlaps its spawn turn (dispatch at the spawn turn's issue instead of its return) when that start is before the spawn turn's recorded end. Each child's first request is offset from the spawn turn's issue (overlap) or from the branch start. The join turn fires immediately once the last child ends.
* **Barriers.** AIPerf's cross-stream replay barriers: within a replay scope (the root with its flat chains, or one sub-agent with its chains) a request waits, on every other stream, for the latest request that had completed at its recorded start. 62% of requests have such predecessors (76k edges, `pred_head`/`pred_list` in `traffic.bin`, optional). Requests before t* count as completed; primers are not gated.
* **Decode.** A request ends when decode has generated its `out` tokens. By default its speed follows its own context along the measured M3 curve, 180 tokens/s/u at 100k (`decodeCurve`, see Decode speed vs context), and an oversubscribed decode ring slows every session (`decodeStages`, see Decode ring).
* **Decode slots (backpressure).** `decodeSlots` (default 0 = unlimited) caps the requests decode holds KV for. A request
  takes a slot before it may start prefill (waiting in FIFO order, inside its TTFT, while all are held) and frees it when
  decode ends, so prefill only runs requests decode has room for. Taking the slot at admission, not at hand-off, is
  what tt-d-gen does: it needs the decode slot up front to start KV migration eagerly. Every run reports the slots held (mean and max over
  the window), how many of them are decoding, and the share of requests that waited and their mean wait. M3 decode
  today holds about 64 sessions, one per pipeline stage (tt-blaze #4220); batched decode (m = 8) targets about 504.
* **Decode KV layouts** (`decodeCache`; page: "Decode KV layout"). `decodeSlots` is the decode KV memory in 1M-token
  slots (0 = unlimited). Every layout reserves a request's decode KV when it is admitted to prefill and releases it when
  its decode ends; the released KV stays cached.
  * `'slots'` (default, today): one 1M slot per request. A released slot keeps its stream's KV, LRU over idle slots
    (as prefill's static slots).
  * `'hybrid'`: `decodeLanes` 1M lanes, with the rest of the memory a content-addressed paged pool (prefill's pool).
    A request's pages (and its output's) are pinned from admission, with shared prefixes counted once. Admission waits
    in FIFO order while the pinned and reserved pages would not fit. `decodeLaneScope`:
    * `'global'` (default): a session takes a lane when its decode starts (default `decodeStages` lanes, × batches once
      decode is batched). It copies its context in once (every stage in parallel, about 0.3 ms at 100k) and unpins its
      pages. At most `decodeLanes` sessions decode; the rest wait for a lane in FIFO order. That wait delays the first
      token, so it counts in TTFT (`decLaneWaitFrac`, `decStartDelayMean`).
    * `'stage'`: `decodeLanes` lanes per stage (default 1). Once more sessions decode than there are lanes, every token
      copies the session's whole context pool → lane on every stage, as prefill's per-stage lanes do for every chunk.
      At 340 B per token per layer per chip, read and written at 50% of DRAM bandwidth, that adds 60 × context × 680 B
      / 256 GB/s to each token's trip: about 16 ms at 100k, against a 5.6 ms trip at 180 tokens/s/u.
  * `'paging'`: an ideal paged decode kernel; the whole memory is the pool.
  * `decodeHostTier`: released slots and unpinned pages go on to host DRAM, then SSD, of the decode galaxies
    (`hostTier` specs × `decodeGalaxies`, default 16). They are read back over the decode side's PCIe and SSDs (before
    decode starts) when a later request needs that prefix. KV migration is not modelled, so decode-side hits
    (`decHitRate`; `migTps` = tokens still migrated per second) change no throughput. A read-back delays decode start
    only if it has not landed by the end of prefill.
  * Runs also report `decPoolTokMean` (pool pages pinned or reserved by sessions) and `decPcieH2DUtil`,
    `decPcieD2HUtil`, `decSsdUtil`. TTFT runs to the start of decode: prefill, plus any wait for a lane or a read-back.
* **Decode ring.** `decodeTps` is the speed per user (TSU) while the ring has room: one token per trip through the
  decode pipeline, so 1 / TSU is the trip time. `decodeStages` (default 0 = unlimited) is how many sessions the ring
  carries at once, one token per stage: 64 for a 64-stage ring, m × 64 with m-row batched decode. Every request past
  prefill decodes at once; with N decoding, each runs at TSU × min(1, decodeStages / N) (processor sharing), so an
  oversubscribed ring slows every session, its requests hold their KV slots longer, and total decode is at most
  decodeStages × TSU tokens/s. The KV slots (`decodeSlots`) are the only hard limit: the ring's memory holds about 85
  slots of 1M tokens per stage (tt-blaze: K/V sharded by head over the 4 mesh rows and replicated over the 2 columns,
  index-K split over the columns, bf8, 340 B per token per chip). Runs report `decodeTpsMean` (mean speed per
  decoding session) and `ringFullFrac` (share of the time the ring carried decodeStages or more sessions).
* **Decode speed vs context** (page: "Decode TSU", constant / adapt to KV length). `decodeCurve: 'm3'` makes each session's speed follow its context. The "Minimax M3
  Updates" deck measures 116 / 105 / 98 / 92 / 72 / 55 tokens/s/u at 8k / 60k / 100k / 140k / 310k / 550k context (64
  users); time per token is linear in context (8.48 ms + 0.0175 ms per 1k tokens, within 0.6% of every point), so
  the model uses that line, scaled so `decodeTps` is the speed at 100k (the deck's targets are quoted @100k): 180
  gives about 214 at 8k, 168 at 140k, 132 at 310k, 101 at 550k and 71 at 990k. Ring sharing applies on top. Default
  `'m3'` at `decodeTps` 180, the decode target (180 tokens/s/u @100k) with the measured shape; `decodeTps` 98
  reproduces the measured table, and `'flat'` gives every session `decodeTps`. The study in `results/study.json` runs
  the default (`'m3'`, 180 @100k); its "decode flat 180" sensitivity row is the old flat speed.
* **KV migration to decode is not modelled.** It streams layer by layer while prefill runs, so only the last layer
  of the last chunk is left when prefill ends, and that overlaps the first token's trip through the decode ring.
* **Why prefill starves.** Runs report the share of the window prefill's first stage had nothing to issue (stage
  free, nothing queued, no started request with tokens left), by cause: `pfStarvedSlotFrac` (requests were waiting
  for a decode KV slot) and `pfStarvedIdleFrac` (no demand). Separately,
  `sendBlockFrac` is the share of time a stage is held after its compute by the synchronous handoff to the next stage
  (about 25% of a saturated stage; `asyncHandoff` removes it).
* **Window.** The profiling window is 1800 s.

**Consequence:** an AgentX lane is mostly idle. Saturating a pipeline therefore takes hundreds to thousands of lanes. The KV working set then grows with concurrency, and throughput is limited by a **cache cliff**: past it, evictions turn into re-prefill, which raises TTFT, which leaves more KV idle and evicted.

## Cost model

Each layer is decomposed into the ops the implementation runs.

* **MoE/MSA layer:** 2× norm all-gather, qkv, index branch, all-gathers of the K/V and index_k prefix, indexer, sparse SDPA, o_proj + RS, shared expert (+RS), router, dispatch, experts, combine, moe_reduce (+RS), misc.
* **Dense layer:** ring-joint SDPA = max(compute ∝ T·(k+T/2), gather of the valid prefix ∝ kv_len/SP) + a fixed cost per call, plus the MLP and its CCLs. The op bounds the gather to [0, kv_len) since tt-metal #47539 (M3 passes `kv_actual_isl` and `logical_n`). Only the gather *buffer* is capacity-sized. `boundedDense: false` models the old whole-lane gather.

**Roofline.** Each op gets a roofline time from FLOPs, DRAM bytes and ethernet bytes on the stage mesh. The Blackhole numbers come from tt-moe-nappkin `lib/system.py`: 608/304 TF (LoFi/HiFi2), 512 GB/s, 100/50 GB/s CCL bi/uni.

**Calibration.**
1. **Zone profiles.** `eff = roofline / (zone time − latency floor)`, using the zone profiles (chunk 5120, 51k cached) for [2,4], [8,4] and [4,2]. [4,4] is the geometric midpoint of [2,4] and [8,4].
2. **Pipeline fit.** A fit on 7,840 per-rank, per-chunk-position medians of the 16×[2,4] runs A/B/C produces:
   * an MoE multiplier of about 1.06 at chunk 5120 and 1.10 at 2048 (pipeline vs single-stage profile, 2D fabric);
   * dense ring-joint: compute 33%, kv_len gather 1.35% (about 100 ms per 1M context tokens per chunk, which dominates at chunk 2048), and 2.75 ms fixed per call.
     * In runs A/B/C the capacity is cached + 51,200, so a capacity term and a kv term look alike across cells.
     * A joint regression on the single-dense-layer stages settles it: about 210 ms per 1M kv vs 19 ms per 1M capacity at chunk 5120, and 100 vs 2 at 2048.
     * Before Sep 29, the fit used a whole-capacity scan. That made today's 1M slots look far too slow and produced a phantom ×3 "bounded dense gather" feature.
   * embedding 1.5 ms;
   * a blocking send of 18 ms at 5120 and 7.5 ms at 2048;
   * a hop latency of 16 ms at 5120 and 14 ms at 2048.

   Two findings came out of the fit:
   * **Padded chunk tails are cheaper.** Routed MoE ops use *actual* tokens (the `actual_isl` trim), so a 640-token request in a 5120 chunk is 18% cheaper than a full chunk.
   * **Today's expert kernel behaves like weight read + compute,** not the max of the two.
3. **Torus rings (`torus`).** A galaxy is two 4×4 tori. Every measurement used line (2D mesh) collectives, so the fitted efficiencies are line efficiencies. On an axis that runs as a ring, the link terms are scaled by textbook ratios:
   * all-gather/reduce-scatter ×(N−1)/N;
   * all-to-all (dispatch/combine, even-split KV write) ×0.5;
   * the ring-joint KV pass ×0.5;
   * the collective latency floor ×0.5.

   There are three modes. `'full'` (default) gives rings only to [4,4] stages, i.e. a whole torus. `'axes'` also gives them to any 4-long axis spanning a torus row or column: [2,4] TP, [4,2] SP, [8,4] TP. `'off'` is the measured setup. `tools/torus_ab.js` compares the modes on every topology.
4. **Validation.** Replaying all 105 matrix cells gives |error| of 5.5% median, 11.0% p90 and 15.9% max (loaded new tok/s and idle TTFT). The old capacity-scan fit scored 4.2 / 8.9 / 15.4%, but for the wrong reason; the dense stages themselves now fit 3× better (log rms 0.033 vs 0.100). The largest misses are the deepest cells, 7–10% under on the dense stages at 553k cached. Run `validate.js` to see it.

**`opEff` knob.** It moves each op geometrically from its measured efficiency to a target (70% matmul, 80% DRAM/link, overlapped expert weight reads) and lowers the latency floors. 0 means today's kernels; 1 means roofline kernels.

**Batching.** Batching works with every cache. With static slots a request holds its 1M slot for its whole trip through the pipeline (today's `slot_id`), so a full pipeline needs (requests per batch) × (stages) slots: the chunks per batch with chunk padding, budget / (32·SP) with tile padding, 1 without batching. `makePlan` reports out of memory when that exceeds the slots that fit. At 4 galaxies, 16 × [2,4] stages hold 20 slots, so batching on slots does not fit; 4 × [8,4] stages fit up to 4 requests per batch. In the study, whose base config uses slots, `batch` requires `pool`. Several requests share one chunk: MoE and projections run on the padded total, and attention runs per request (`seq`) or once per chunk (`fused`), with a 110-core wave-quantization factor. Tile padding (`reqPad: 'tile'`) pads each request's tokens in a batch to 32·SP and adds an all-to-all KV write per layer.

**Memory per stage.**
* weights: bf4 experts, bf8 attention, bf16 shared/dense MLP, embedding on stage 0, LM head on the last stage;
* `reserveGB` per chip plus activation buffers;
* KV: 1088 B/token/layer for K/V (bf8), plus index_k at 128 × (2 B bf16 | 1.0625 B bf8) × (TP replicas | 1).

KV capacity is the minimum over stages. Defaults:
* bf8 index_k replicated ×4, as deployed (1632 B/token/layer);
* a 1 GB reserve, calibrated so that 16×[2,4] with an even split fits 35 × 1M slots, as measured on hardware (Sep 30 2026, 4-MoE-layer stages bind);
* the default auto split ([2,3,4×13,3] on 16×[2,4]) also puts at most 4 layers on a stage and fits the same (the split before Oct 8, [1,1,1,4×8,5×5], fit 28);
* the #57827 calibration replays keep bf16 index_k, as those runs used.

Every buffer that must hold a whole request (slots, fixed lanes, the lane arena) must be at least 990,016 tokens, the largest AgentX request; `makePlan` rejects smaller ones.

## KV residency

* **`slots`** (today): one 1M slot per stream; LRU over idle slots. The hit is the prefix shared with the stream's previous request, resumed at any 32-token boundary (`unaligned`, default on since tt-metal #57636 merged; `--set unaligned=false` floors it to a chunk multiple, the old behaviour).
  * With `hostTier` (the offload tiers, below): a reclaimed slot's KV (its stream's latest request) is written to host DRAM over PCIe, and the new occupant waits for that write-back; host DRAM overflows to SSD. The stream's next request reads back the prefix it shares with it before admission (or at start, if its slot was reclaimed while it was queued). One copy per stream, LRU per tier, index_k stored once. In the Oct 8 study, at 4 galaxies with today's kernels, the tiers take goodput from 3.8k to 18.5k (p90 9.8 s, hit 95.5%), and adding the pool (one lane per stage) takes it to 25.5k. The tiers do not require the pool, and they are the first roadmap step in every scenario.
* **`pool`**: lanes (fixed 1M per stage, or request-sized in an arena) plus a content-addressed paged pool with LRU over prefix-tree pieces.
  * **Lanes per stage** are derived: with a per-stage lane table a stage works on one chunk at a time, so it needs one lane per request in the batch it is processing (1 without batching, the chunks per batch with chunk-padded batching), times the buffers of `copyMode` (sequential 1, double 2, overlap3 3). A global lane table (`laneScope: 'global'`) holds a lane for the request's whole trip through the pipeline, so it derives stages × requests per batch, and it allows only sequential copies (buffering a lane reserved for the whole trip gains nothing). `lanesOverride: true` sets the count to `lanes` instead; it is allowed only with batching on the pool. Tile-padded batching has no chunks, so it must override. Every request in a batch holds its own lane, so the count caps the requests per batch, under every policy. Each extra 1M lane comes out of the pool: at 4 galaxies without the SSD tier, one lane beats three by 2–4% and eight lanes cost 13–22% with batching. The study's batched stacks override (its pool feature uses 4 lanes; the grid tries 2/4/6); its unbatched pool steps derive 1 lane.
  * The prefix tree is compressed into pieces cut at branch points and request ends, so every request touches whole pieces. LRU over pieces matches LRU over 64-token pages except that a piece is evicted whole. Against a brute-force page LRU, total hits agree within 0.15%.
  * The hit is looked up when the request becomes ready (which refreshes it in the LRU, but does not pin it), and re-checked when the request starts. Pages evicted meanwhile are recomputed; pages demoted to host meanwhile are fetched over PCIe before the first chunk.
  * The new KV enters the pool when the lane is freed: with per-stage lanes, when stage 0 finishes the request's last chunk (later stages follow in FIFO order); with global lanes, at prefill completion.
  * Copy-in/out is DRAM-bound per stage. By default (`copyMode: 'sequential'`) the full copy time is charged on the stage. `'double'` overlaps copies with compute and charges 25% of their time for DRAM contention; it was the default before Oct 1; the study now uses the sequential default. `'overlap3'` is static triple buffering.
  * With `hostTier` (the **KV offload tiers**), device evictions move to host DRAM, host DRAM evicts to SSD, SSD drops (exclusive LRU tiers over prefix-tree pieces). Hits are read back before admission. Offloaded copies store index_k once (no TP replicas; re-broadcast on fetch).
  * Known approximation: KV fetched from host at READY is staged on device without being counted against capacity until the request starts.
* **`paging`**: an ideal paged kernel (no lanes, no copies). The pages a request is writing are reserved in the pool from start to completion. The offload tiers also work behind paging.
* **KV offload tiers** (`hostTier`): device DRAM → host DRAM → SSD, per galaxy host. Defaults, from the [Tenstorrent Galaxy Blackhole Server User Guide v1.8](https://docs.tenstorrent.com/_downloads/3086863c42126fd0d63b01baccf8432e/galaxy-blackhole.pdf) unless noted:
  * **Host DRAM** `hostDramGBPerGalaxy` 576 GB (6 × 96 GB DDR5-4800 RDIMM). The share left for KV is derived in `makePlan` (`plan.hostBudget`), and these reserves are estimates:

    | reserve | default | key |
    |---|---|---|
    | headroom | 10% (58 GB) | `hostHeadroom` |
    | OS and services | 16 GB | `hostOsGB` |
    | model runtime (tt-metal, inference server, program caches) | 32 GB | `hostRuntimeGB` |
    | staging for KV migration to decode | 32 GB | `hostKvStagingGB` |
    | pinned PCIe host memory: 1 GiB hugepage channel per chip (tt-metal/UMD picks min(4, chips per MMIO device) = 1 on a Galaxy, where every chip has its own PCIe link) | 34 GB | `hostPinnedGBPerChip` |
    | this galaxy's share of the model weights (host copies while loading / reloading) | 33 GB at 8 galaxies, 67 GB at 4 | `hostStageWeights` |

    That leaves about **371 GB of KV per galaxy at 8 galaxies** (40M tokens per pipeline at 73.4 KB/token off device, next to 77M of device pool) and 337 GB at 4 galaxies (18M tokens, next to 29M). Set `hostDramGBPerGalaxy: 0` to drop the host tier. The AgentX rules cap host DRAM per system; check that cap against this budget.
  * **SSD** `ssdTBPerGalaxy` 16 TB (planned hardware; 4 E1.S bays, shipped systems carry 4 × 7.68 TB Samsung PM9D3a, MZTL67T6HBLC = 30.7 TB): 1.7B tokens at 8 galaxies. Bandwidth `ssdReadGBsPerGalaxy` 31.5, `ssdWriteGBsPerGalaxy` 27.2 GB/s: the [PM9D3a](https://semiconductor.samsung.com/ssd/datacenter-ssd/pm9d3a/) reads up to 12,000 MB/s and writes up to 6,800 MB/s at PCIe 5.0 x4, but Rev C chassis run the E1.S links at Gen4 x4 (a BIOS setting in the Exabox provisioning runbook), which caps a read at 7.88 GB/s per drive. Reads and writes share the drives.
  * **PCIe** `pcieGBsPerGalaxy` 63 GB/s each way, full duplex (fetches and write-backs do not share it). Per tray, one chip has a Gen5 x8 host link and seven have x1 links that train at Gen4 (user guide: "PCIe Gen5 1x8 and 7x1"; Quanta S7TK product spec: "7x1 PCIe G4"). KV is spread evenly over the chips and each chip moves its own share over its own link, so the x1 chips set the pace: 32 × 1.97 GB/s. Relaying through each tray's x8 chip over Ethernet would allow up to 4 × (31.5 + 7 × 1.97) = 181 GB/s; no software does that today.
  * Not modelled: host DRAM bandwidth (6 channels of DDR5-4800, about 230 GB/s, above the links' combined peak), SSD endurance and per-IO latency. A device page counts as free as soon as its write-back is queued. Runs report `hostReadTps`, `ssdReadTps` and the busy share of each link (`pcieH2DUtil`, `pcieD2HUtil`, `ssdUtil`).
* **`inf`**: infinite cache.

## Scheduling (`policy`)

* **`rtc`** (used by the study and the configs built from it; formerly `fcfs`, which is still accepted): run to completion, oldest first. A started request is never preempted: every chunk continues the oldest started request; a new request starts only when no started request has tokens left. With batching, started requests are continued first and new ones fill the leftover room.
* **`srpt`**: also run to completion, with the waiting queue sorted shortest-new-first (30 s aging). Despite the name it does not preempt.
* **`rr`** (default): round robin, as in tt-d-gen (`PrefillQueue` + `PrefillWriter::step`). A request joins the back of the queue when it is admitted, i.e. when it gets a slot or lane. Each turn the front request takes one chunk and goes to the back if it has chunks left. With batching, requests are popped from the front and each takes as many of the batch's remaining chunks as it can fill (all it has left, if that is less), as one attention call. Unfinished requests go to the back behind everything waiting, so a request is never split into two runs within one batch. On static slots the in-flight cap is tt-d-gen's ChunkFifo, max(8, 4 × slots).
  * **`batchChunksPerRequest`** L (default 0 = no limit): batches are filled in rounds over the queue, each request getting up to L more chunks per round, until the budget is used or no request needs more. L = 0 is greedy (the front request takes all it can, then the next), L = 1 an even split, and a request alone fills the batch whatever L is.
  * **`batchMaxReqs`** (default 0 = no limit): the most requests in one batch. With static slots it bounds the slots in flight to `batchMaxReqs` × stages, which is what lets tile-padded batching fit on slots; on fixed pool lanes it sets the derived lane count, so no lane override is needed.
    * Every request in a batch makes its own attention call and reads its own cached prefix, so many requests per batch cost throughput. At 8 galaxies (pool + tiers, chunk 512, budget 8k = 16 chunks): L = 0 reaches 72.4k goodput with 2.6 requests per batch; L = 8 70.3k with 3.2 (TTFT p50 7.0 → 5.6 s); L = 4 62.1k; L = 2 45.3k; L = 1 31.6k with 5.6. With a 16k budget, L = 16 ties L = 0 (72.3k vs 72.1k).
  * **On the pool** (`cache: 'pool'`): a *lane* is a per-stage KV slot that the attention kernels run on (1M tokens, or request-sized in an arena). A request takes a lane for each turn, and every turn copies the segment's new KV out to the pool. The partial KV stays pinned in the pool until the request finishes, so a lane never holds the only copy and can be handed to another request at any turn. The lane count bounds the requests per batch, not the requests in progress.
  * The lane count is the same as for the other policies (see **Lanes per stage** under KV residency). A request takes a lane for each turn, so the count caps the requests per batch but not the requests in progress.
  * Copy-in: a request whose lane still holds its context, i.e. no other request has used that lane since its last turn, skips the copy-in. On a miss it takes a free lane (an empty one first, else the least recently used) and copies its whole context so far in. Arena lanes always copy in. Results report `rrLaneReuse` (share of continuation turns that skipped the copy-in) and `rrCopyInTps`. Copying in every turn instead made no measurable difference to goodput (≤0.5%, even with sequential copies), so it is not an option.
  * Round robin on the pool, and paging under rr, pin the cached prefix of every request in progress. A request is admitted only while the pool can hold all in-progress requests in full; until then it waits in the queue. This matches tt-d-gen, where admission needs a free slot, a full pool queues the request rather than rejecting it, and in-flight slots are never evicted.

## Findings (study of Oct 8 2026, third run: decode speed follows each session's context along the measured M3 curve, 180 tokens/s/u at 100k; with the new auto split, deployed memory defaults (bf8 index_k, 1 GB reserve), KV offload tiers (host DRAM + 16 TB SSD per galaxy) behind static slots, round-robin scheduling, sequential pool copies, derived lane counts, unaligned resume in the baseline; decode ring and decode slots unlimited, AIPerf-exact replay, `results/study.json`)

Goodput in useful tok/s at p90 TTFT ≤ 10 s. "Today" = 16×[2,4] (or 32×[2,4]), chunk 2048, auto split, static 1M slots, round-robin scheduling, unaligned resume. Decode is unlimited in every study configuration except the decode-limited sensitivity rows: the study ranks prefill features assuming decode keeps up (takeaway 8).

**Changes from the second Oct 8 run: decode speed follows context** (`decodeCurve: 'm3'`, the page's "Decode TSU: adapt to KV length"). The second run decoded every session at a flat 180 tokens/s/u. In this run the time per token is linear in the session's context, along the measured M3 curve, scaled so 180 tokens/s/u is the speed at 100k: about 214 tokens/s/u at 8k, 168 at 140k and 101 at 550k. Decode is now also a processor-sharing ring with the KV slots as its only hard limit, but the defaults (`decodeStages` 0 and `decodeSlots` 0, unlimited) leave the study base unchanged; only the context curve moves the numbers.
* Today's baselines: 6.2k / 8.7k / 13.0k / 16.4k → 6.1k / 8.7k / 12.5k / 16.2k (4 gx today / 4 gx roofline / 8 gx today / 8 gx roofline).
* Greedy full stacks: 65.2k / 126k / 131k / 285k → 65.4k / 122k / 132k / 274k.
* Best grid configs: 73.9k / 137.8k / 150.3k / 294.4k → 74.2k / 133.3k / 150.8k / 284.4k: within 0.5% with today's kernels, 3% lower with roofline kernels.
  * 4 gx today: still 16×[4,2] at budget 16k, now with a 4M arena (the 2M arena is 0.04% behind).
  * 8 gx roofline: 32×[2,4] at budget 4k, 0.7% ahead of 32×[4,2] at the same budget (282.5k, 4M arena), which is within seed noise. 32×[4,2] at budget 8k, the second run's winner, gives 281.1k (4M arena).
* ∞-cache references: 74.5k / 193.2k / 151.0k / 367.4k → 75.0k / 203.8k / 151.5k / 315.4k. The 8-gx roofline reference drops because it re-runs the new winner (32×[2,4] at budget 4k). The best configs now reach 99% / 65% / 99.5% / 90% of it (second run: 99% / 71% / 99.5% / 80%).
* The greedy order is unchanged in all four scenarios, and so are the tiers (P0/P1/P2). In the 8-gx-today ranking column every feature moves by at most 0.09; fused attention and the MSA indexer (both ×1.04) swap places within P2.
* New sensitivity rows: decode 90 / 360 tokens/s/u @100k (they replace 90 / 360 flat), decode flat 180 (no context curve), decode as measured (98 @100k), and two decode-limited rows: 85 slots with a 64-stage ring (today's M3 decode) and 150 slots with a 128-stage ring (batched decode, m = 2). See takeaway 8.
* The `tools/*.js` diagnostics were re-run against this run's best configs. With today's kernels most numbers move by 1–2% (paging + tiers on today's config the most: 30.4k → 31.5k). With roofline kernels the SSD wall costs more at 4 gx and less at 8 gx (takeaway 10), and the 8-gx roofline diagnostics now run on 32×[2,4].

**Changes in the second Oct 8 run, from the first: the auto split** (`splitLayers` in `sim_core.js`). The old auto split gave each dense layer its own stage and scored a split by the mean of per-chunk worst stages at 60k–550k context. It now minimises the busiest stage's mean time per chunk over the contexts AgentX chunks actually see (deciles, mean about 140k), and dense layers may share stages with each other and with MoE layers. 16×[2,4] moves from [1,1,1,4×8,5×5] to [2,3,4×13,3]; 32-stage splits keep their bottleneck. With batching the split is computed for the batch budget, so it varies by config (e.g. [2,2,4×14] for 16×[4,2] at budget 16k).
* Today's 4-gx baselines rise with the extra slots: no stage holds 5 layers any more, so 36 slots fit instead of 28. 3.8k → 6.2k with today's kernels, 6.5k → 8.7k with roofline kernels. At 8 gx (32 stages) they stay at 13.0k and 16.4k.
* Best grid configs: 60.3k / 131k / 150k / 293k → 73.9k / 138k / 150k / 294k.
  * 4 gx today +23%: still 16×[4,2], now with a 2M arena instead of 6 fixed lanes.
  * 4 gx roofline +5%, with a new winner: 8×[4,4] (2M arena, budget 8k).
  * The 8-gx winners stay 32×[4,2].
* Topologies with 4–16 stages gain most (best per topology, old → new):
  * 4 gx today: 16×[2,4] 53.1k → 66.8k, 8×[4,4] 54.5k → 62.9k, 4×[8,4] 30.5k → 38.4k.
  * 8 gx today: 16×[2,4]×2 97.6k → 129k, 16×[4,4] 103k → 129k, 8×[4,4]×2 111k → 128k.
  * 32-stage topologies move by under 1%.
* Feature gains shift at 4 gx:
  * Today's kernels: the pool's step gain drops (×1.38 → ×1.22) but its leave-one-out loss rises (×1.90 → ×2.35). Batching rises (×1.16 → ×1.33), and the arena turns from a loss into a gain (×0.92 → ×1.04, ×1.11 leave-one-out).
  * Roofline kernels: the pool rises (×1.18 → ×1.34) and batching drops (×1.34 → ×1.08).
  * The 8-gx columns move by at most 1%, so the tiers (P0/P1/P2, ranked on 8 gx today) are unchanged.
* With roofline kernels the best configs reached 71% / 80% of their own ∞-cache goodput (first run: 86% / 90%). The ∞-cache reference rose (4 gx 152k → 193k with the [4,4] winner, 8 gx 326k → 367k at budget 8k instead of 4k), while the real cache stayed bound by tier reads.
* The `tools/*.js` diagnostics were re-run against the second run's best configs and splits. At 8 gx they moved by under 1%; at 4 gx (new configs) they changed.

**Changes in the first Oct 8 run** (from Oct 7: host DRAM + 32 TB SSD, bf16 index_k, 3 GB reserve per chip; Oct 1: one 1 TB SSD tier at 64 GB/s):
* **Memory defaults match the deployment.**
  * index_k is bf8, the dtype the runner deploys; bf16 is rejected. "index_k bf8" is no longer a roadmap feature; the bf16 sensitivity row shows what it would cost.
  * The per-chip reserve is 1 GB instead of 3. That reproduces the hardware slot fit: 35 × 1M slots on 16×[2,4] with an even split (Sep 30 2026), 70 on 8 galaxies.
  * Today's auto split (up to 5 layers per stage) fits 28 slots on 4 galaxies, where the old defaults gave 20.
  * The #57827 calibration replays keep bf16 index_k, as those runs used. Validation is unchanged.
  * Today's baselines rise with the extra slots: 3.4k → 3.8k and 11.1k → 12.9k with today's kernels, 4.8k → 6.5k and 15.0k → 16.4k with roofline kernels.
* **The KV offload tiers replace the Oct 1 1 TB SSD tier:** device evictions go to host DRAM (about 337 GB of KV per galaxy at 4 galaxies, 371 GB at 8) over PCIe (63 GB/s each way), and host DRAM evicts to a 16 TB SSD (31.5 / 27.2 GB/s read / write). The Oct 7 study ran the same tiers with 32 TB of SSD.
* **The best grid configs are within 2% of Oct 7:** 60.7k / 129k / 150k / 289k → 60.3k / 131k / 150k / 293k. Against Oct 1 (45.1k / 82.6k / 104k / 161k) that is ×1.34–1.81. The greedy full stacks are 48.8k / 125k / 131k / 284k (Oct 1: 39.8k / 78.3k / 94.8k / 158k).
  * **16 TB of SSD per galaxy is enough:** 64 TB changes nothing in any scenario. With today's kernels 4 TB is also enough (1 TB costs 18–25%); with roofline kernels 4 TB costs 8–11% and 1 TB 36–42%.
  * bf8 index_k is worth 7–8% with roofline kernels (the bf16 row) and the 1 GB reserve 1–2%; with today's kernels both are within 0.7%.
  * Re-prefill in the full stack is 2–4% of prefilled tokens (Oct 1: 18–59%). The best configs reach 97–99% of their own ∞-cache goodput with today's kernels and 86–90% with roofline kernels (Oct 1: 40–75%).
* **The tiers are the first step everywhere,** ×3.5–7.4 when added (Oct 7: ×4.0–9.9, from lower baselines; Oct 1: ×3.7–6.7) and ×2.1–2.7 leave-one-out (Oct 1: ×1.6–1.9).
* **With capacity no longer binding, compute features gain** (as on Oct 7). Batching is P0 (×1.42 / ×1.62 at 8 gx today), request padding P1 (×1.15 / ×1.17) and the arena P1 (×1.08 leave-one-out at 8 gx today, but ×0.92 at 4 gx today). index_k stored once is P2: ×1.00 with today's kernels, ×1.04–1.13 with roofline kernels, where tier bandwidth binds.
* **At 4 gx with roofline kernels the best topology is 4×[8,4]** (131.3k vs 129.1k for 16×[4,2]), now at budget 32k. The other three best configs are [4,2] stages, as on Oct 1.
* The sensitivity runs vary the tiers and the memory defaults: no host DRAM tier, 1 / 4 / 64 TB of SSD per galaxy, PCIe 181 GB/s per galaxy (x8 relay), half SSD bandwidth, a 3 GB reserve per chip, bf16 index_k (Oct 1 varied 0.5 / 2 TB and 16 / 256 GB/s of SSD).

**Changes in the Oct 1 study (from Sep 29 b):**
* **Unaligned resume is part of today's baseline** (tt-metal #57636, merged): a conversation resumes at any 32-token boundary instead of rounding its cached prefix down to a chunk multiple. It is no longer a roadmap feature.
  * On top of round robin, today's baselines move 3.3k → 3.4k and 10.9k → 11.1k with today's kernels, 4.6k → 4.8k (15.0k unchanged) with roofline kernels: on static slots the goodput point is set by capacity.
  * Request padding loses the part of its gain that was the rounding loss: at 4 gx today ×1.20 (step #3) → ×1.08 (step #6). It drops to P2, and batching to P1 (×1.19 / ×1.24 at 8 gx today, just under the 1.25 cut).
  * The best stacks use tile padding, which unaligned resume does not touch: full stacks, grids and sensitivity rows are unchanged.
  * Still open on the tt-metal side: #57636 was validated on the first 5 layers (8x4, against a one-pass prefill); the end-to-end two-turn 60-layer PCC check against the CPU reference has not been run yet.
* **Round-robin scheduling is the base** (the simulator's default, as tt-d-gen) instead of run to completion.
  * Today's baselines rise: 3.1k → 3.3k and 8.2k → 10.9k with today's kernels, 4.0k → 4.6k and 11.4k → 15.0k with roofline kernels.
  * Shortest-first run to completion is now ×0.97–1.02 on top of it, where shortest-first scored ×1.01–1.08 over run to completion. Round robin already keeps short requests from waiting behind long ones.
* **The SSD tier works behind static slots** (a reclaimed slot is written to SSD and read back when its conversation returns), so it no longer needs the pool. (Why the pool still adds on top of it: takeaway 1.) It is now the first roadmap step in every scenario (×3.7–6.7), and the pool comes second or third (×1.23–1.52 at its step, ×1.71–1.83 leave-one-out).
* **Pool copies are sequential by default, and lanes per stage are derived** (1 per stage without batching). With batching the study keeps 4 lanes, and the grid tries 2/4/6.
* **A global lane table allows only sequential copies.** It now costs more: at 8 gx with roofline kernels the best config drops 161k → 46k.
* The best grid configs move by at most 3%: 45.1k / 82.6k / 104k / 161k. All four are [4,2] stages.

Earlier studies are kept on exabox under `/data/philei/m3_traffic_sim/results/`, which is not in git. The tables below are printed by `node tools/feature_table.js`.

| scenario | today | greedy full stack | best grid config | best config with ∞ cache |
|---|---|---|---|---|
| 4 galaxies, today's kernels | 6.1k | 65.4k | **74.2k** (16×[4,2], 4M arena, budget 16k) | 75.0k |
| 4 galaxies, roofline kernels | 8.7k | 122k | **133k** (8×[4,4], 2M arena, budget 8k) | 204k |
| 8 galaxies, today's kernels | 12.5k | 132k | **151k** (32×[4,2], 2M arena, budget 16k) | 152k |
| 8 galaxies, roofline kernels | 16.2k | 274k | **284k** (32×[2,4], 2M arena, budget 4k) | 315k |

The ∞-cache column re-runs each best config with an infinite cache, so it moves with the config: at 4 gx with roofline kernels the Oct 1 best config (16×[4,2]) reached 209k with an infinite cache, the first Oct 8 run's 4×[8,4] 152k, the second run's 8×[4,4] 193k, and this run's 8×[4,4] 204k (same config; decode now follows context). At 8 gx with roofline kernels the second run's 32×[4,2] at budget 8k reached 367k, and this run's 32×[2,4] at budget 4k reaches 315k.

Hourly volume and revenue at the goodput point (the simulated concurrency where each configuration reaches its goodput, p90 TTFT ≤ 10 s): input tokens of the requests completed per hour, split into new tokens the pipeline prefilled (re-prefill included) and cached prefix hits, requests completed per hour, and the revenue of those input tokens (output tokens not counted).

Revenue at $0.30/M input, $0.06/M cached.

| scenario | configuration | C | input tok/h | new tok/h | cached tok/h | hit | requests/h | revenue/h |
|---|---|---|---|---|---|---|---|---|
| 4 galaxies, today's kernels | today | 72 | 668.9M | 118.1M | 550.8M | 82.3% | 5.0k | $68.49 |
| 4 galaxies, today's kernels | greedy full stack | 744 | 6.66B | 242.5M | 6.41B | 96.4% | 52.7k | $458 |
| 4 galaxies, today's kernels | best grid config | 840 | 7.45B | 277.0M | 7.18B | 96.3% | 59.9k | $514 |
| 4 galaxies, roofline kernels | today | 80 | 821.6M | 203.0M | 618.6M | 75.3% | 6.9k | $98.02 |
| 4 galaxies, roofline kernels | greedy full stack | 1264 | 12.65B | 447.2M | 12.20B | 96.5% | 101.5k | $866 |
| 4 galaxies, roofline kernels | best grid config | 1384 | 14.07B | 489.9M | 13.58B | 96.5% | 111.2k | $962 |
| 8 galaxies, today's kernels | today | 112 | 1.13B | 195.1M | 938.5M | 82.8% | 10.2k | $115 |
| 8 galaxies, today's kernels | greedy full stack | 1560 | 13.39B | 496.1M | 12.89B | 96.3% | 108.0k | $922 |
| 8 galaxies, today's kernels | best grid config | 1768 | 15.37B | 565.5M | 14.80B | 96.3% | 123.2k | $1,058 |
| 8 galaxies, roofline kernels | today | 152 | 1.47B | 324.6M | 1.15B | 78.0% | 12.7k | $166 |
| 8 galaxies, roofline kernels | greedy full stack | 3000 | 28.97B | 1.01B | 27.97B | 96.5% | 228.2k | $1,979 |
| 8 galaxies, roofline kernels | best grid config | 3072 | 30.13B | 1.05B | 29.08B | 96.5% | 237.3k | $2,060 |

**Decode KV layouts and revenue** (`tools/decode_layout_ab.js`, `results/decode_layout_ab.{json,txt}`, Oct 9). The
decode side is a 62-stage ring with 75 × 1M slots of KV memory, decode offload tiers, and 180 tokens/s/u @100k on the
M3 curve. The prefill side is as in the decode backpressure table below: pool + host DRAM + SSD tiers, today's kernels
and round robin. For each decode layout the table shows the best prefill config by net revenue at its goodput point
(p90 TTFT ≤ 10 s, which includes any wait for a decode lane), at $12 per galaxy-hour with 16 decode galaxies. Output is
billed as decoded.

| decode KV layout | prefill gx | best prefill | goodput | requests/h | output decoded, tok/s | decoding, at tokens/s/u | ring full | waits | revenue/h, in + out | net/h |
|---|---|---|---|---|---|---|---|---|---|---|
| fixed slots: 62 stages / 75 slots | 6 | chunk 512, batch 4k | 39.5k @ C=432 | 32.2k | 9.2k | 56 at 163 | 26% | 98% wait 4.4 s for KV | $270 + $39.79 | **$45.36** |
| hybrid, global lanes: 62 stages / 62 lanes | 6 | chunk 1024, batch 4k | 43.0k @ C=464 | 34.6k | 10.0k | 61 at 165 | 86% | 84% wait 3.4 s for a lane | $291 + $43.18 | **$70.25** |
| hybrid, per-stage lanes: 62 stages / 1 lane | 6 | chunk 1024 | 18.4k @ C=680 | 13.9k | 2.7k | 374 at 7 | 99% | 17% wait 9.1 s for KV | $95.40 + $11.59 | **−$157** |
| (ref) paged: 62 stages / 75M-token pool | 6 | chunk 512, batch 16k | 50.7k @ C=744 | 39.6k | 10.3k | 218 at 47 | 98% | – | $324 + $44.29 | **$104** |
| (ref) unlimited decode | 6 | chunk 512, batch 8k | 54.0k @ C=592 | 42.4k | 12.1k | 75 at 162 | 0% | – | $378 + $52.22 | **$166** |
| fixed slots: 62 stages / 75 slots | 8 | chunk 2048 | 41.8k @ C=456 | 33.7k | 9.7k | 60 at 162 | 43% | 98% wait 5.3 s for KV | $282 + $41.81 | **$35.79** |
| hybrid, global lanes: 62 stages / 62 lanes | 8 | chunk 2048 | 43.5k @ C=464 | 35.0k | 10.1k | 61 at 165 | 94% | 93% wait 4.2 s for a lane | $294 + $43.67 | **$49.80** |
| hybrid, per-stage lanes: 62 stages / 1 lane | 8 | chunk 1024 | 18.5k @ C=680 | 13.9k | 2.7k | 377 at 7 | 100% | 16% wait 9.4 s for KV | $95.27 + $11.61 | **−$181** |
| (ref) paged: 62 stages / 75M-token pool | 8 | chunk 1024, batch 4k | 53.9k @ C=992 | 42.0k | 10.4k | 394 at 26 | 99% | 31% wait 3.4 s for KV | $332 + $44.88 | **$88.94** |
| (ref) unlimited decode | 8 | chunk 512, batch 8k | 72.9k @ C=816 | 59.2k | 16.5k | 101 at 164 | 0% | – | $510 + $71.41 | **$294** |

* **The 62-stage ring is the limit in all three scenarios.** It decodes at most about 62 × 165 = 10.2k tokens/s, so
  the layouts differ in how full they keep it and how much prefill they let through. Most of the revenue is input
  revenue (prefill); output adds 10–15%.
* **Global lanes beat fixed slots: +$25/h at 6 prefill galaxies (+55%), +$14/h at 8 (+39%).** With fixed slots a
  request holds its 1M slot from admission, so 15–19 of the 75 slots hold requests still queued or in prefill. Only
  56–60 sessions decode and the ring is full only 26–43% of the time, while 98% of requests wait 4–5 s for a slot.
  With 62 global lanes the 13.6M-token pool holds the sessions in prefill (120–124 held, 7–8M tokens of pool in use),
  so a lane is free whenever a session finishes. The ring stays full 86–94% of the time with 61 decoding: +4–9%
  output and +4–8% requests. The cost moves to the first token: 84–93% of requests wait 3.4–4.2 s for a lane, inside
  the 10 s TTFT.
* **Per-stage lanes with one lane are not viable for decode (−$157 / −$181 per hour).** Every token copies its
  session's whole context in on every stage, so a session decodes about 4× slower even before the ring is shared
  (about 25 ms per token at 120k instead of 6). The 74M-token pool admits 380 sessions; they share the ring at 7
  tokens/s/u, and output falls to 2.7k tokens/s. These points are not steady: tokens decoded are 80% of those
  completed, so the backlog is still growing and the real loss is larger.
* **Paging (reference)** keeps more sessions in the ring (218–394 decoding) and serves more requests ($104 / $89),
  but every session then decodes at 26–47 tokens/s/u. Nothing here caps time per output token, so a TPOT SLO would
  take most of that gain back. Unlimited decode ($166 / $294) shows what the decode limit costs.
* **The decode tiers change nothing here.** The decode side finds 95–96% of each request's context already cached
  (the pool 96%, shared across sub-agents; slots 95%, a stream's own previous request), but that only saves KV
  migration, which is not modelled.

**Decode backpressure with the decode ring** (`decodeSlots`, `decodeStages`; not part of the study base). Pool + host
DRAM + SSD tiers, today's kernels, decode following context at 180 tokens/s/u @100k, goodput at p90 TTFT ≤ 10 s, net
revenue per hour at MiniMax's prices, $12 per galaxy-hour and 16 decode galaxies; best of no batching (chunk 1024 /
2048) and batching (chunk 512 / 1024, budget 4k-16k):

| prefill | decode slots / ring stages | no batching | batching | batching gain |
|---|---|---|---|---|
| 6 galaxies | 62 / 64 | $11 | $9 | −$3 |
| 6 galaxies | 85 / 64 | $49 | $66 | +$17 |
| 6 galaxies | 150 / 64 | $54 | $95 | +$41 |
| 6 galaxies | 150 / 128 | $55 | $165 | +$111 |
| 8 galaxies | 62 / 64 | $0 | −$11 | −$11 |
| 8 galaxies | 85 / 64 | $57 | $55 | −$2 |
| 8 galaxies | 150 / 64 | $83 | $82 | −$1 |
| 8 galaxies | 150 / 128 | $150 | $259 | +$109 |

With today's 64-stage ring, batching pays only at 6 galaxies (prefill-bound); at 8 galaxies the ring is full 65–98%
of the time from 85 slots up and every session slows. A 128-stage ring (m = 2 batched decode) never fills here, and
batching then adds about $110/h at either size.

The tables below are older (Oct 7): flat 180 tokens/s/u, a hard cap on sessions decoding instead of the ring, 8
galaxies, chunk 512, batching with an 8k budget, paged pool, today's kernels; revenue at $0.45 / $0.09 per M. Slots
held count requests from admission to prefill until the end of their decode.

| decode | slots | goodput | C | requests/h | revenue/h | slots held, mean / max | decoding | waited for a slot |
|---|---|---|---|---|---|---|---|---|
| 180 tokens/s/u | unlimited | 30.7k | 312 | 25.2k | $372 | 75 / 135 | 40 | – |
| 180 tokens/s/u | 128 | 30.7k | 312 | 25.3k | $372 | 75 / 128 | 40 | 1%, 0.3 s |
| 180 tokens/s/u | 64 | 28.6k | 280 | 23.1k | $338 | 58 / 64 | 36 | 46%, 3.9 s |
| 180 tokens/s/u | 32 | 16.1k | 144 | 12.1k | $151 | 27 / 32 | 20 | 57%, 3.4 s |
| 100 tokens/s/u | unlimited | 28.2k | 320 | 22.8k | $342 | 97 / 157 | 64 | – |
| 100 tokens/s/u | 128 | 28.0k | 312 | 22.7k | $335 | 93 / 128 | 63 | 5%, 0.9 s |
| 100 tokens/s/u | 64 | 23.0k | 248 | 18.1k | $254 | 62 / 64 | 49 | 68%, 3.8 s |

One 64-session decode keeps up with this prefill at 180 tokens/s/u for a 7% loss, and at 100 tokens/s/u for 18%. 128 slots
is enough either way.

Margin per hour of the same runs, at $12 per galaxy-hour, 16 decode galaxies per 64 slots, and either OpenRouter's
prices ($0.30 / $0.06 / $1.20 per M input / cached / output) or 1.5× them ($0.45 / $0.09 / $1.80):

| decode | slots | galaxies (prefill + decode) | cost/h | revenue/h, OpenRouter (in + out) | margin | revenue/h, 1.5× | margin |
|---|---|---|---|---|---|---|---|
| 180 tokens/s/u | 64 | 8 + 16 | $288 | $225 + $28 = $253 | −$35 | $338 + $43 = $381 | +$93 (24%) |
| 100 tokens/s/u | 64 | 8 + 16 | $288 | $169 + $21 = $190 | −$98 | $254 + $32 = $286 | −$2 |
| 180 tokens/s/u | 128 | 8 + 32 | $480 | $248 + $31 = $279 | −$201 | $372 + $47 = $419 | −$61 |

Decode is two thirds of the cost but bills only the output tokens: with a slot held from admission, about 40% of the
64 slots hold requests still queued or in prefill rather than decoding.

Each cell below is "G #step / LOO":
* **G** is the gain at the greedy step where the feature was added (the step number is the build order);
* **LOO** is the leave-one-out loss when the feature is removed from the full stack.

Order and tier come from one reference scenario, **8 galaxies with today's kernels**: the larger of G and LOO there, highest first. P0 ≥ 1.25, P1 ≥ 1.07.

A single scenario is used on purpose. Taking the maximum over scenarios lets a feature rank high on one column, and substitutes can both get credit that only one of them earns.

Complexity is a rough judgement of how much of the stack a feature touches:
* **low**: one op or the scheduler, plus validation;
* **med**: a new op or allocator, or changes across a few ops;
* **high**: cross-cutting changes to the runtime, KV layout, several kernels and the scheduler.

It is not a time estimate and has not been checked with the code owners.

| tier | feature | 4 galaxies, today's kernels | 4 galaxies, roofline kernels | 8 galaxies, today's kernels (ranking) | 8 galaxies, roofline kernels | complexity |
|---|---|---|---|---|---|---|
| P0 | KV offload tiers (host DRAM + 16 TB SSD per galaxy) | ×4.05 #1 / ×2.31 | ×6.52 #1 / ×2.63 | ×3.53 #1 / ×2.16 | ×5.14 #1 / ×2.76 | high |
| P0 | Slot lanes + paged KV pool (1M lanes per stage, copy-in/out) | ×1.21 #2 / ×2.36 | ×1.37 #3 / ×2.02 | ×1.24 #3 / ×2.23 | ×1.30 #3 / ×2.19 | high |
| P0 | Async stage handoff (overlap D2D) | ×1.34 #4 / ×1.50 | ×1.32 #2 / ×1.30 | ×1.38 #2 / ×1.74 | ×1.89 #2 / ×2.21 | med |
| P0 | Multi-request batching (16k token budget) | ×1.34 #3 / ×1.60 | ×1.05 #4 / ×1.12 | ×1.42 #4 / ×1.64 | ×1.18 #4 / ×1.22 | high |
| P1 | Request padding to 32·SP | ×1.08 #5 / ×1.19 | ×1.01 #6 / ×1.01 | ×1.15 #5 / ×1.17 | ×1.06 #6 / ×1.03 | high |
| P1 | Variable-size lanes (contiguous arena, 4M tokens) | ×1.08 #8 / ×1.11 | ×1.00 #8 / ×1.00 | ×1.02 #6 / ×1.08 | ×1.00 #9 / ×1.00 | med |
| P2 | Fused multi-user attention | ×1.03 #7 / ×1.11 | ×1.00 #9 / ×1.00 | ×1.03 #7 / ×1.04 | ×1.00 #7 / ×1.00 | high |
| P2 | MSA SP-local indexer (no K/V/index prefix all-gather) | ×1.04 #6 / ×1.03 | ×1.01 #7 / ×1.01 | ×1.03 #8 / ×1.04 | ×1.00 #8 / ×1.00 | high |
| P2 | index_k cache not replicated over TP (store once) | ×1.00 #9 / ×1.01 | ×1.11 #5 / ×1.13 | ×1.00 #9 / ×1.00 | ×1.06 #5 / ×1.11 | med |
| P2 | Shortest-first run to completion (30 s aging) | ×0.98 #10 / ×0.98 | ×0.99 #10 / ×0.99 | ×0.99 #10 / ×0.99 | ×1.00 #10 / ×1.00 | low |

Takeaways (study numbers are from this run, the third Oct 8 study, unless marked first / second Oct 8 run, Oct 7 (host DRAM + 32 TB SSD, bf16 index_k, 3 GB reserve) or Oct 1 (a 1 TB / 64 GB/s SSD tier in place of host DRAM + SSD); the separate `tools/*.js` runs cited below (layout, batch-shape, cache/batch, SLO, tier, fill, attention and torus diagnostics) were re-run on Oct 8 against this run's best configs and the current defaults: decode speed following context (180 tokens/s/u at 100k), round robin, sequential copies, derived lanes (the study's 4 with batching on the pool), unaligned resume, bf8 index_k, 1 GB reserve and, where a tier is on, host DRAM + 16 TB SSD):
1. **On AgentX, KV capacity is the first wall; with the offload tiers it is gone with today's kernels, and tier bandwidth is the next wall with roofline kernels.** The best configs reach 99% of their own ∞-cache goodput with today's kernels and 65% (4 gx) / 90% (8 gx) with roofline kernels (second Oct 8 run: 99% and 71–80%; first Oct 8 run: 97–99% and 86–90%; Oct 1, 1 TB SSD tier: 40–75%).
   * With an infinite cache the roofline winners reach 204k at 4 gx (8×[4,4]; Oct 1's 16×[4,2] 209k) and 315k at 8 gx (32×[2,4] at budget 4k; the second run's 32×[4,2] at 8k reached 367k), against 133k and 284k with the tiers.
   * The tiers and the pool are the top features in every scenario. Behind static slots the tiers alone give ×3.5–6.5; the pool then adds ×1.21–1.37 by sharing prefixes and freeing the slots' memory.
   * **Why the pool still adds on top** (`tools/tier_diag.js`, `results/tier_diag.txt`: today's 4-gx config, round robin, chunk 2048, 36 slots). Behind slots, SSD capacity stops mattering at 8 TB/gx: 8 and 64 TB both give 24.7k at C=232, 1 TB 24.3k. Two structural losses remain:
     * **No prefix sharing across streams.** A slot, and its offloaded copy, holds one stream's KV, so a request can only reuse the prefix of its *own* previous request. Sub-agents (42% of requests) re-prefill the context they share with their parent and siblings: hit 95.1–95.6% vs 96.2–96.6% possible. That is 21–27% of all prefill work. The content-addressed pool shares those pages.
     * **Slot churn.** 36 slots of 1M tokens fit on device, so almost every request swaps a slot: the evicted stream's whole KV is written out and the new request's whole prefix is read back.
       * Reads from host DRAM: 240M tokens (plus 15M from SSD) vs 1M with pool + tiers at C=128; 756–761M (+84–86M) vs 36M (+23M) at each one's goodput point.
       * In-flight prefill is capped at the slot count.
       * The admission waits push p90 TTFT up early (11.9–12.2 s vs 3.4 s at C=256).
     * Result: pool + tiers reaches 29.9k at C=312 (1 or 8 TB/gx) and paging + tiers 31.5k at C=344, the infinite cache's goodput, vs 24.3–24.7k for slots + tiers.
     * At the same concurrency it also processes fewer tokens for the same useful work: at C=128, 19.2k vs 23.4k processed for 15.2k vs 15.1k useful.
   * Tier bandwidth matters more than capacity, once there is enough capacity. On the best configs:
     * 64 TB instead of 16 TB of SSD per galaxy changes nothing. Less does cost: 4 TB costs 0.5% at 4 gx and 11% at 8 gx with roofline kernels (under 0.5% with today's kernels), 1 TB 25–27% with today's kernels and 27–42% with roofline kernels.
     * PCIe at 181 GB/s instead of 63 (relaying through the x8 chips) moves goodput by under 0.5%.
     * Half the SSD bandwidth costs under 1% with today's kernels and 23–24% with roofline kernels (4 gx 133k → 102k, 8 gx 284k → 217k).
     * No host DRAM tier (SSD only) costs 0.1–0.3% with today's kernels and 15–18% with roofline kernels.
     * Oct 1, with the 1 TB tier, capacity was what mattered: 0.5 → 2 TB/galaxy moved 4-gx goodput 36.4k → 54.5k (today's kernels), and 64 → 16 GB/s cost at most 7%.
   * The lane table must be per stage: a global lane table (today's slot_id) drops the best configs 2.0× at 4 gx (74.2k → 36.2k, 4M arena; the second run's 2M-arena winner dropped to 20.1k) and 7.4× at 8 gx (151k → 20.3k) with today's kernels, and 2.0× at 8 gx with roofline kernels (284k → 142k). The 8-stage [4,4] winner at 4 gx with roofline kernels loses only 2.5%.
2. **Async stage handoff: ×1.30–2.21,** growing with pipeline depth and kernel speed (×1.32 / ×1.30 on the 8-stage 4-gx roofline winner, ×1.34 when added / ×1.50 leave-one-out at 4 gx today, ×1.89 / ×2.21 at 8 gx with roofline kernels). The measured blocking send is 6–23 ms per chunk per stage.
3. **Batching: ×1.05–1.64,** most at 8 gx with today's kernels (×1.42 when added, ×1.64 leave-one-out). Batches in the best configs average 1.4–4.3 requests at the goodput point (4.3 at 4 gx today, 4.2 at 8 gx today, 1.4 and 1.7 with roofline kernels).
   * On today's 4-gx config + pool (4 lanes) + offload tiers + batch 16k they average 2.38 requests; the average chunk is 11.4k of the 16k budget (`tools/slo_ab.js`; first Oct 8 run, 28 slots: 1.36 requests, 6.8k).
   * Most of the gain comes from one request taking several chunks at once (big cold prefills in fewer, larger chunks), not from mixing users.
   * **Dynamic batch size (`batchDynShape`, on by default).** A batch that is not full (e.g. a single request) runs at the tokens it holds, rounded up to whole chunks, as ops do without tracing. Off = padded to the full budget, as a traced build with one fixed shape must be; routed MoE ops still trim to the real tokens (`tools/batch_shape_ab.js`):
     * Today's 4-gx config + pool (4 lanes) + offload tiers, budget 4k / 8k / 16k / 32k: sized 36.6k / 39.5k / 40.0k / 27.3k vs static 36.6k / 39.3k / 28.3k / 12.0k. No batching is 29.9k.
       * A static 16k or 32k shape collapses below no batching (55–83% padding): batches there hold only 1.3–1.9 requests.
     * Best stacks (chunk 256 or tile padding): static 16k costs 0–2% (4 gx roofline 132.6k → 130.7k; 8 gx under 0.5%), and static 32k costs 0.6–31% (4 gx today 52.9k → 36.3k).
       * Batches there hold 1.4–4.3 requests at 16k, and cold prefills fill the budget.
     * A static shape near the typical fill (8k) is within 0.5% of dynamic sizes on the best stacks and on today's config (39.3k vs 39.5k).
   * Without the offload tiers, batching on the pool loses: 21.9k unbatched (one derived lane) vs 19.1k with 4 lanes + batch 16k and 16.4k with 8 lanes. The extra lanes come out of the pool, and the goodput point is set by cache misses (32–34% of prefilled tokens are re-prefill), not compute (`tools/cache_batch_ab.js`, `results/cache_batch_ab.txt`). Paging + batch 16k is 20.4k (unbatched 21.8k).
   * With the tiers, batching adds 34% on today's config (pool 29.9k → 40.0k; first Oct 8 run, 28 slots: 16%), and today's config reaches 99% of its own ∞-cache goodput: pool / paging + tiers + batch 16k give 40.0k / 40.2k, the infinite cache 40.5k (unbatched: 29.9k / 31.5k vs 31.5k). Paging vs pool is worth under 1% batched (5% unbatched), so further gains must come from compute and TTFT. Best budget (`tools/batch_shape_ab.js`): 16k on today's config (40.0k, 8k 1% behind); on the best stacks 16k with today's kernels; with roofline kernels 8k (at 4 gx 4k–16k are within 0.5%, 32k −2%; at 8 gx 8k is 1–2% ahead of 16k).
4. **Bounded dense gather is already done** (tt-metal #47539). Without it (the old whole-lane gather), the first Oct 8 run's best 4-gx config with today's kernels (6 fixed 1M lanes) would drop 60.3k → 31.4k. Configs with arena lanes, which all four best configs now use, lose under 1%.
   * The M3 comments that say the dense layers gather the whole cache shard (`prefill.py`, `tt_prefill_runtime.reconfigure_capacity`, README `PREFILL_MAX_SEQ_LEN`) are stale. Only the gather buffer is capacity-sized, which costs memory, not time.
5. **index_k stored once: ×1.06–1.13 with roofline kernels, ×1.00–1.01 with today's kernels.** bf8 index_k is now the default (the deployed dtype); going back to bf16 would cost 8–9% with roofline kernels and under 0.5% with today's kernels. With today's kernels the tiers already hold the working set; with roofline kernels smaller KV means less to read back from the tiers. (Oct 7, bf16 default: stored once ×1.11–1.17 and bf8 ×1.04–1.11 with roofline kernels; Oct 1, 1 TB tier: ×1.05–1.13 and ×1.05–1.08 everywhere.)
6. **Request padding to 32·SP (even split + all-to-all KV write) is worth 1–19%,** least at 4 gx with roofline kernels (×1.01); ×1.08 / ×1.19 at 4 gx today, ×1.15 / ×1.17 at 8 gx today, ×1.06 / ×1.03 at 8 gx roofline (Oct 1: 1–8%). (Before unaligned resume (#57636) joined the baseline it showed up to 20%: most of that was the chunk-rounding loss on resume.) The a2a KV write before the cache write is costed on every layer.
   * **A small chunk plus batching gets nearly all of it** (`tools/layout_ab.js`, `results/layout_ab.{json,txt}`: the stack and topology of each best grid config, offload tiers, its arena).
     * The best chunk is 128: 69.2k / 133.4k / 140.2k / 281.7k (4 gx today / 4 gx roofline / 8 gx today / 8 gx roofline), at budget 16k / 8k / 16k / 8k.
     * For the same attention it matches tile padding: per-request attention within 0.4% of it (ahead in three of four), fused attention within 0.3%. The 7–8% gap to tile padding + fused attention with today's kernels (4 gx 74.2k, 8 gx 150.8k) is the fused attention, not the layout.
     * Chunk 1024 is about 3% behind that (2.6–3.4%), and chunk 2048 is 5–7% behind 1024.
     * **At 4 gx with roofline kernels (8×[4,4]) none of this matters:** every chunk from 128 to 2048 at budget 8k is within 0.3% (133.0–133.4k), and so are fused attention and prefetch; budgets 4k–16k at chunk 128 are within 0.6%. The SSD read wall (takeaway 10) sets the goodput point there, not compute.
     * Batching itself is worth ×1.35–1.63 over the best unbatched chunk (×1.16 at 4 gx with roofline kernels).
     * Chunk padding here means each request takes whole chunks, padded to C (10% padding at C=1024).
     * Chunk 1024 is extrapolated: the model was calibrated at 2048 and 5120.
   * **Batch size and compute:** per-token MoE cost at [4,2] (today's kernels) falls 12% from 4k to 8k, 6% from 8k to 16k and 3% from 16k to 32k. The expert matmuls cross from weight-bound to compute-bound at about 8k tokens per chunk. With today's kernels goodput is 6–9% higher at 16k than at 8k, 4k is 20% worse, and 32k loses 5–24%, because the longer period pushes p90 TTFT over the SLO. With per-request attention and roofline kernels 8k is best: at 4 gx 4k and 16k are within 1% and 32k is −2%; at 8 gx 16k is −1.8%, 32k −7% and 4k −5%. (With fused attention the study's grid prefers 4k for 32×[2,4] at 8 gx with roofline kernels: 282.8k vs 280.3k at 8k, both with a 4M arena.)
   * **Smaller chunks (down to 32·SP = 128)** only cut padding once a request's chunks form one attention call. C=128 matches tile padding and is about 3% better than C=1024.
   * **One attention call and prefix gather per request (`attn: 'request'`) is essential for small chunks.** With one per chunk (`attn: 'chunk'`) instead:
     * small chunks collapse, e.g. C=128 at 16k gives 2.0k instead of 140.2k at 8 gx today;
     * the best choice becomes C=2048, which is 20–26% below the de-duplicated best (8 gx today: 104.0k vs 140.2k); only at 4 gx with roofline kernels, where the SSD binds, is it within 1% (132.5k).
     * Example: a cold 16k-token segment at 140k context pays 2.7 ms of gather per MoE layer as one call, but 41 ms as 16 × 1024 units.
   * **Prefetching the KV-prefix gathers (`prefetchKV`) is worth 0–3%** with per-request attention (most with today's kernels: 4 gx 69.2k → 70.9k, 8 gx 140.2k → 144.7k). On top of chunk padding + fused attention it gives 4 gx today +3.5% (74.4k → 77.0k, above the study's best of 74.2k), 8 gx today +2.5%, and under 0.5% with roofline kernels.
     * In a synthetic 16k batch of 8 requests at 140k context it cuts the MoE layer 69 → 48 ms.
     * On AgentX at the goodput point, batches average 1.4–4.3 requests, so the gathers are small.
     * At 8 gx with today's kernels the bottleneck is the three single-dense-layer stages (at the ∞-cache peak 100% busy vs 93% for the MoE stages, `tools/attn_diag.js`). Prefetch hides only part of their ring gather (about 3% of a dense layer, `tools/attn_diag.js`), so it barely helps them.
     * The next lever there is dense attention itself: ring-joint compute runs at 33% efficiency, or the dense layers could get more chips.
   * **Fused attention** (`attn: 'fused'`) means one attention call per batch instead of one per request: the per-call fixed costs (op launch, collective latency, the fitted dense ring-joint cost) are paid once, and wave quantization is over the whole batch. Each request still attends only to its own context and gathers its own prefix. It does not assume a packed varlen kernel; one would be one way to provide that call.
   * **The same questions with an infinite cache** (compute-bound view; `tools/layout_ab.js --inf` → `results/layout_ab_inf.{json,txt}`, `tools/budget_ext_inf.js`, `tools/fused_fixed_inf.js`). The findings above use the real cache (pool or arena + offload tiers), where with roofline kernels SSD read bandwidth limits concurrency.
     * **Best fixed setup:** chunk 128 everywhere, and chunk size now matters more, also at 4 gx with roofline kernels:
       * chunk 1024 is 2–10% behind and chunk 2048 is 7–20% behind (most at 8 gx roofline);
       * budget 16k with today's kernels (8k is −5–8%; 32k is −2% at 8 gx and −24% at 4 gx, and ≥48k collapses on TTFT);
       * with roofline kernels the largest budget wins, still rising 0.4–2%: at 4 gx 32k → 48k → 64k = 240.7 → 245.2 → 247.2k, at 8 gx (32×[2,4]) 330.6 → 334.2 → 335.5k.
       * Goodput: 69.5k / 247.2k / 140.8k / 335.5k (4 gx today / 4 gx roofline / 8 gx today / 8 gx roofline). At 4 gx with roofline kernels the ∞-cache goodput of 8×[4,4] is 1.85× the real cache's 133.4k (the first Oct 8 run's 4×[8,4]: 150.3k vs 131.2k).
     * **Tile padding adds nothing.** Tile padding + per-request attention is within 1% of chunk 128: 0.4–0.9% below it in three scenarios, 0.4% above at 8 gx roofline.
     * **Fused attention is what helps, and it works with chunk padding at C=128 too** (chunk + fused is within 0.6% of tile + fused, ahead in three of four):
       * +7.9% at 8 gx today (140.8 → 151.9k) and +13.2% at 8 gx roofline (330.6 → 374.4k), where the single-dense-layer stages are the bottleneck and a fused call pays the fitted 2.75 ms per-call dense-attention cost once per chunk instead of once per request;
       * +8.2% at 4 gx today (69.5 → 75.2k; first Oct 8 run +1.8%). The split for budget 16k, [2,2,4×14], puts two dense layers on stage 0, which becomes the bottleneck: at C=776 it is 99% busy vs 86–87% for the MoE stages with per-request attention (one replay, not in `results/`).
       * +1.4% at 4 gx roofline.
       * This relies on that per-call cost being launch/setup-like, which is not verified.
     * **Prefetching the KV-prefix gathers: 0–3.5%** on top of chunk + fused (4 gx today 75.2 → 77.8k, 4 gx roofline +1.5%, 8 gx today 151.9 → 156.4k, 8 gx roofline −0.1%), about as much as with the real cache.
     * **One prefix gather per request (`attn: 'request'` rather than `'chunk'`) matters even more.** Without it, the best setup is chunk 2048 and is 22–29% lower.
     * **Peak throughput (any TTFT), infinite cache** (`tools/layout_ab.js --inf --slo none --budgets …,65536`, `results/layout_ab_inf_peak.{json,txt}`). Same conclusions:
       * Best fixed setup: chunk 128, budget 64k (still rising 0.6–3.9% from 32k) except at 4 gx today, where 16k is best (32k −6.0%, 64k −4.5%; at 64k the split becomes [1,1,3,5,…], with five MoE layers on some stages). Peak: 71.2k / 251.4k / 152.5k / 341.8k.
       * Relative to that: chunk 1024 is 4–9% lower, chunk 2048 is 9–18% lower.
       * Tile padding alone: −1.1 to +0.5%.
       * Fused attention with chunk padding: +6.7% at 4 gx today, +1.6% at 4 gx roofline, +9.9% at 8 gx today and +12.9% at 8 gx roofline.
       * Prefetch: 0–4.5%.
       * Per chunk (`attn: 'chunk'`): 18–29% lower.
       * **Where fused attention and prefetch act** (`tools/attn_diag.js`, `results/attn_diag.txt`; the 8-gx best configs, 32×[4,2] with today's kernels and 32×[2,4] with roofline kernels):
         * **Sparse (MSA) layers:** each request gathers its own K/V + index prefix, which takes longer than its indexer + sparse attention (2.4 vs 1.3 ms per request at 140k with today's kernels, 0.4 vs 0.1 with roofline kernels on [2,4]). A fused call cannot share those gathers, so fused attention saves only 0–3% of an MSA layer (latency floors, core fill).
         * **Dense layers:** fused attention saves 11–17% at 4–10 requests per batch. That is almost entirely the fitted fixed cost of about 2.75 ms per ring-joint call, paid once per chunk instead of once per request. It is not verified that this cost is per call: time a dense layer with 1 vs several segments.
         * The gain grows with requests per batch, not with smaller chunks (chunk 128 vs 1024 changes it by ≤1.6 points).
         * **Prefetch** cuts an MSA layer by 3–5% with today's kernels (1–2% with roofline kernels) and a dense layer by 0–3%. That is 3% with today's kernels, where the dense ring gather exceeds its compute, and 0% with roofline kernels, where it doesn't.
         * **Bottleneck at the peak:** at 8 gx the three single-dense-layer stages are at 97–100% with both kernel sets (MoE stages 93% with today's kernels, 67% with roofline). That is why fused attention gains most there. Prefetch still gives +4% with today's kernels only because it hides part of the dense gather, and nothing with roofline kernels. At 4 gx (64k budget) the MoE-only stages are the bottleneck (99–100%); the stages holding the dense layers are at 77–86% (first Oct 8 run, which gave each dense layer its own stage: 32–49%).
       * With today's kernels the peak is 2–8% above goodput at 10 s (4 gx at the same 16k budget; 8 gx at 64k with 14 requests per batch). With roofline kernels it is 1.7–1.9% above.
     * **Owner placement instead of the all-to-all, with MSA's 128-row KV blocks** (`tools/placement_ab.js`, `results/placement_ab_inf_peak.{json,txt}`; peak, infinite cache, each scenario's best budget). With tile padding the KV slab is fixed at 128·SP, on the assumption that MSA needs whole 128-row KV blocks on each SP rank. `placement: 'owner'` computes each token on the SP rank that owns its KV row (as #57636 does for a chunk): requests padded to 32, no all-to-all, every rank padded to the busiest one.
       * Batched, against the even split: 0 to −4.0% with fused attention (−4.0 / 0.0 / −2.4 / −0.4%: 4 gx today / 4 gx roofline / 8 gx today / 8 gx roofline), +0.3 to −7.8% per request. A request's tokens in a batch shorter than 128·SP land on few ranks, and attention waits for the busiest one; at 8 gx roofline, packing to per-rank caps also puts 21 requests in a batch instead of 14.
       * Unbatched (budget 2048 / 5120): owner and even split within 1% of each other, 2–8% / 25–56% above chunk padding at the same size.
       * Chunk padding at its smallest MSA-valid chunk (C = 128·SP): 1.6–2.5% behind tile padding with an even split, fused attention. The chunk 128–256 results above have C/SP < 128 at SP = 4, which this assumption rules out.
       * So with the 128·SP slab, the even split plus the all-to-all is the better tile-padding option.
       * Balance-aware packing (choosing which requests share a batch to even out the ranks) would matter only for owner placement, since the even split is balanced whatever is packed. It could recover at most the all-to-all's cost, which the model puts at 0.2–0.5% of a batched layer (bytes over the links at the MoE dispatch's measured efficiency, plus 0.04 ms latency; not measured). Not worth adding.
       * Chunk padding's last chunk of a request is costed with its routed tokens spread evenly over the ranks; on hardware they are uneven (#57636), so that is slightly optimistic. Owner placement uses the busiest rank's.
7. **Topology** (best of the grid per topology; the grid runs every topology at the middle budget and only the best two at every budget; seeds move results by up to 5%):
   * [4,2] stages win both grids with today's kernels; they need KV heads sharded 2 per chip.
     * 4 gx today: 16×[4,2] 74.2k vs 66.4k for 16×[2,4].
     * 8 gx today: 32×[4,2] by 13% (150.8k vs 133.9k for 32×[2,4]).
     * 8 gx roofline: 32×[2,4] at budget 4k wins by 0.7% (284.4k vs 282.5k for 32×[4,2] at 4k), within seed noise. In the second Oct 8 run 32×[4,2] won by 1%.
     * At 4 gx with roofline kernels 8×[4,4] stages win: 133.3k vs 130.4k for 4×[8,4] and 123.0k for 16×[4,2]. In the first Oct 8 run, 4×[8,4] won with 131.3k and [4,4] had 123.0k.
   * **[4,4] torus stages:** with ring collectives they win at 4 gx with roofline kernels and lose elsewhere: 62.5k vs 74.2k at 4 gx today, 127.9k vs 151k at 8 gx today, 256.6k vs 284k at 8 gx with roofline kernels. With the old auto split they lost everywhere (55.4k, 103k, 266k): it gave each dense layer its own stage, which costs most when there are few stages.
     * Per chip, a [4,4] MoE layer is still about 20% more expensive than a [4,2] one. The TP=4 collectives cost more, and a 16-chip stage pays the same fixed per-op latency as an 8-chip one.
     * Rings alone give [4,4] +18–21% with today's kernels (4 gx 53.0k → 62.5k, 8 gx 105.6k → 127.8k), +1% at 4 gx with roofline kernels and +11% at 8 gx (16×[4,4] 235.0k → 259.8k), on each scenario's best stack (`tools/torus_ab.js`, `results/torus_ab.txt`).
   * Rings on every 4-long axis ([4,2]'s SP axis, [2,4]'s TP axis) add 16% to [4,2] at 8 gx with today's kernels (151k → 175k) and nothing at 4 gx (74.2k → 73.3k; with the old split they gave 60.3k → 72.2k, mostly by speeding up the single-dense-layer stages). With roofline kernels they move it by under 1%.
   * 8 gx: one 32-stage pipeline still beats 2×16 at goodput, but by less (both [2,4]: 134k vs 129k today, 284k vs 241k roofline; first Oct 8 run 98k and 246k). At the ∞-cache throughput peak without batching (chunk 2048, no SLO, unlimited decode) 2×16×[2,4] is ahead: 64.6k vs 55.3k (53.4k with the old split). [8,4] stages lose at 8 gx.
8. **Decode capacity, not prefill features, is the binding limit for deployment.** The study ranks prefill features with unlimited decode: every request past prefill decodes at once, at its own speed on the context curve, and decode slots are unlimited. The decode-limited sensitivity rows re-run each scenario's best config with a real decode:
   * **Today's M3 decode** (85 slots: memory for about 85 sessions of 1M tokens; a 64-stage ring decodes 64 sessions at full speed and shares it beyond that): every study winner falls to 41–46k useful tok/s, 41.2k / 46.2k / 42.7k / 45.9k against 74.2k / 133.3k / 150.8k / 284.4k (4 gx today / 4 gx roofline / 8 gx today / 8 gx roofline). With unlimited decode the KV offload tiers alone already reach more than that in three scenarios (56.6k / 44.2k / 83.0k at 4 gx roofline / 8 gx today / 8 gx roofline); at 4 gx today the tiers, pool, batching and async handoff reach 53.6k.
   * **150 slots and a 128-stage ring** (batched decode, m = 2, K/V split across the columns): 65.5k / 92.3k / 74.6k / 92.3k, still 12–68% below the best configs.
   * So the deployable goodput is set by how many sessions decode can hold and serve, and the feature ranking above assumes decode keeps up. Past the first one to four greedy steps, prefill features pay off only as decode capacity grows. (The decode backpressure table above runs one smaller 8-gx configuration, 30.7k with unlimited slots, where 64 slots cost 7% at 180 tokens/s/u.)
   * **Decode speed, with decode unlimited** (`decodeTps`, the speed at 100k context): faster decode raises prefill goodput with roofline kernels (90 → 360 tokens/s/u @100k: 116k → 148k at 4 gx, 246k → 288k at 8 gx; 180 gives 133k and 284k), because it shrinks the live KV working set per unit of load. With today's kernels it moves goodput within seed noise (4 gx 74.5k → 73.1k, 8 gx 154k → 150k). Oct 1, with the 1 TB tier: 42.7k → 46.7k at 4 gx today.
   * **The context curve itself** costs 2–3% with roofline kernels against a flat 180 tokens/s/u (flat: 137.7k at 4 gx, 290.3k at 8 gx) and is within 0.5% with today's kernels (73.8k, 150.3k). At the measured speed (98 tokens/s/u @100k) the roofline winners give 118.3k and 249.8k (−11% and −12%), and today's 74.4k and 153.9k.
9. **Pool copies:** double-buffered copies are within 1% of sequential; triple buffering costs 11% at 8 gx with today's kernels (its lanes take memory), 1% at 4 gx today (whose best config now has a 4M arena; the second run's 2M arena lost 12%), and nothing with roofline kernels. 4 fixed 1M lanes are 1–4% behind an arena.
10. **The SLO is not what limits batch fill. With today's kernels nothing much does; with roofline kernels SSD read bandwidth does** (`tools/slo_ab.js`, `results/slo_ab.txt`).
    * Requests per batch at p90 ≤ 10 s → no SLO: 2.38 → 2.98 on today's 4-gx config + pool + tiers + batch 16k, and 1.37–4.28 → 1.39–4.30 on the best stacks.
    * Dropping the SLO gains 0.3–1.1% of goodput on the best stacks: the throughput peak sits at about the same concurrency as the 10 s point. Today's config gains 4% (40.0k → 41.7k). In the first Oct 8 run (28 slots) it gained 21%, because its batches filled only past the 10 s point; with 36 slots they hold 2.38 requests (11.4k of 16k) at 10 s and 3.0 (14.5k) at the peak, at p90 16 s.
    * With an infinite cache, the best stacks hold 1.86–4.44 requests per batch at 10 s (15.8k and 16.1k of the 16k budget with today's kernels; 8.1k of 8k at 4 gx and 4,056 of 4,096 tokens at 8 gx with roofline kernels) and 1.78–4.45 with no SLO, with 97–100% of the bottleneck stage busy.
    * With today's kernels the real cache fills batches almost as well (4.28 vs 4.31 requests at 4 gx, 4.24 vs 4.44 at 8 gx) and reaches 99% of the ∞-cache goodput. With roofline kernels it does not (1.37 vs 2.70 at 4 gx, 1.73 vs 1.86 at 8 gx; 65% and 90% of the ∞-cache goodput).
    * **Where the real cache falls short** (`tools/fill_diag.js`, `results/fill_diag.txt`: the best config at its own goodput concurrency and at the ∞-cache one, one tier assumption varied at a time; `tools/host_curve.js`, `results/host_curve.txt`: the same variants as full sweeps):
      * **Roofline kernels: the SSD is the wall.** At the goodput point the SSD is 100% busy (0.73M tok/s read at 4 gx, 1.45M at 8 gx), PCIe only 33–38%, and the pipeline 77–89% busy. One step further, at the ∞-cache goodput concurrency (2320 at 4 gx, 3416 at 8 gx), reads queue: wait-to-start reaches 39–119 s, re-prefill rises from 2–3% to 7–11%, and useful throughput falls to 32.6k / 141.5k vs 204k / 314k with an infinite cache.
        * At the same goodput concurrency, half the SSD bandwidth gives 31.9k / 45.0k instead of 133k / 284k, and no host DRAM tier 44.7k / 69.6k. Re-run as sweeps, the goodput point moves to lower concurrency instead: half the SSD bandwidth −23% / −24% and no host DRAM tier −18% / −16% (the study: −23–24% and −15–18%).
        * PCIe at 181 GB/s changes nothing (PCIe is not the bottleneck). Full paging instead of pool lanes adds 0.6–1.1%.
        * The 4-gx winner, 8×[4,4], has an ∞-cache goodput of 204k at budget 8k (247k at 64k, takeaway 6), so the SSD wall costs it 35% (second Oct 8 run: 29%; the first Oct 8 run's 4×[8,4]: 13%, 131k vs 152k). At 8 gx it costs the 32×[2,4] winner 10% (284k vs 315k; the second run's 32×[4,2] at budget 8k: 20%).
      * **Today's kernels: the tiers hold the working set.** At the goodput point the SSD is 32–33% busy and PCIe 11–14%. Every variant (no host DRAM, half SSD bandwidth, PCIe 181 GB/s, paging) lands within 0.7% of the best config (paging +0.6–0.7%, the others within 0.3%), and the infinite cache is 0.7–1.1% higher.
      * At the same concurrency, batches are slightly less full with the real cache than with an infinite one (4 gx roofline 1.37 vs 1.38), unless a slower tier adds re-prefill, which makes each request longer (half SSD bandwidth: 2.07, no host DRAM tier: 2.36).
      * Oct 1, with the 1 TB / 64 GB/s tier, capacity was the whole gap: past about 480 streams at 4 gx (1136 at 8 gx) the working set no longer fit, re-prefill reached ~60% at the ∞-cache peak, and 8 TB/galaxy matched the infinite cache. With 16 TB of SSD per galaxy, capacity no longer binds (64 TB changes nothing); read bandwidth does. More SSD bandwidth per galaxy (more drives, or the Gen5 links the drives support) and smaller KV (index_k stored once) are the levers.
    * A 5 s SLO costs 17–31% on the best stacks with today's kernels (47% on today's config), and 1–2% with roofline kernels.
    * The page has an SLO slider, whose right end means no SLO. A no-SLO run bisects the throughput peak.
      * Any sweep whose throughput peaks before the SLO crossing also bisects the peak (`lib/sweep.js` step 4). Before that fix, a re-run with a looser SLO could report up to 3% *lower* goodput (4 gx roofline: 81.4k at 10 s, 78.9k at 30 s), because its bisection moved past the peak. With the fix, a looser SLO is at most 1% lower (4 gx today 75.1k at 20 s, 75.0k at 30 s; with an infinite cache 75.9k at 20 s, 75.7k at 30 s).
      * `results/study.json` (Oct 8) includes the fix.
11. **The cliff is steep.** Goodput can change by 10% between neighbouring concurrency points, so the sweep bisects the SLO crossing.

## Assumptions to revisit

* Offload tiers: capacities and bandwidths come from the Galaxy Blackhole documentation and the drive datasheet, but the host DRAM reserves (OS, runtime, KV staging) are estimates, and transfers are ideal page DMAs. The PCIe figure assumes each chip moves its own KV over its own link; measure the sustained rate.
* Pool copies are page-list gathers at 50% DRAM efficiency. Arena fragmentation is not modelled.
* Decode speed is the measured M3 shape scaled to a 180 tokens/s/u @100k target, and the ring is ideal processor
  sharing (no per-step overheads, no batched-decode slowdown, no TPOT limit). Decode KV is a memory of 1M slots in one
  of three layouts (`decodeCache`); per-stage lane copies are DRAM-bound page gathers at 50% efficiency. KV migration to
  decode is not modelled, so decode-side prefix hits and the decode offload tiers do not change throughput.
* Meshes without a profile ([4,4], [1,4], …) are extrapolated. TP=2 stages use the [4,2] single-stage profile.
* Ring-collective speed-ups on the torus are textbook link-load ratios, not measured; profile a [4,4] stage with ring CCLs to pin them down.
* Batched chunks larger than 5120 are extrapolated from the roofline scaling of each op.

## Extending (notes for the next agent)

* **New feature.** Add a knob to `DEFAULTS` in `sim_core.js`, use it in `roofTok` / `roofSeg` / `layerMs` (cost), `makePlan` (memory), or the scheduler (`formChunk`, `tryStart`). Then add a `FEATURES` entry in `study.js`, a control in `artifact/template.html` (`FIELDS`) and a complexity line in `build_artifact.js` (`SCOPE`).
* **New hardware data.** Rerun `collect_calib.py`. It segments the timing CSVs per cell with `results_16stage.jsonl`. Then run `validate.js` and check the error summary before trusting a study.
* **New corpus.** Rerun `prep_traffic.py`. hash_ids must stay prefix-chained and topologically increasing (checked on 19.6M block pairs).
* `lib/pool.js` `summarize` and the copy in `artifact/template.html` must stay identical.
