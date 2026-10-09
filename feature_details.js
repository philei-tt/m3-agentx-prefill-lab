// Per-feature explanations shown in the artifact's Roadmap table (click a feature to expand).
// what: what the feature is and where it sits today; why: mechanism + numbers; todo: rough implementation steps;
// notes: interactions and caveats. Keys match study.js FEATURES.
'use strict';
module.exports = {
  pool: {
    what: 'Replace the static 1M-token slots with a few contiguous <b>lanes</b> per stage plus a shared <b>paged pool</b>. Lanes are the KV buffers the existing attention kernels run on. The pool is the rest of KV memory, split into pages keyed by the hash of their content (64-token blocks), shared by every conversation with the same prefix and evicted least-recently-used. Before a request runs, its cached prefix is copied from pool pages into a lane; afterwards its new tokens are copied back to the pool.',
    why: 'Today a conversation owns a whole 1M slot even when its context is 50k tokens, so 4 galaxies keep only 20 conversations warm. AgentX needs hundreds of live sessions to fill the pipeline, so almost every request misses: 0.3% hit rate at 256 lanes in the simulation. A pool stores only the tokens that exist: the same 21M tokens hold about 150 median (140k) contexts instead of 20. It shares identical prefixes (sub-agents start from the main agent\'s system prompt and tools) and evicts from the tail, so a conversation keeps its head. The copies are cheap: a median hit is about 65-107 MB per chip, about 0.5 ms of DRAM time, overlapped with compute.',
    todo: [
      'Page allocator and page table per stage in device DRAM; prefix index from block hash to page on the host.',
      'A page-list gather/scatter op that copies many pages into or out of a lane in one launch (per-page host copies would be dispatch-bound).',
      'A per-stage lane table: each stage allocates a lane only while it works on the request (today slot_id is global across stages). This is required once batching is on: a global lane table drops the best 4-galaxy config from 45k to about 14k useful tok/s (8 galaxies: 102k to 28k).',
      'Pool copies double-buffered: copy-in of the next batch and copy-out of the previous one overlap compute. The peak lane memory is about 2x the computing lanes, briefly; sequential copies (1x) cost under 1% on the best configs.',
      'Scheduler: LRU eviction, pinning of in-flight prefixes, copy-in of the next request while the current chunk computes, copy-out after the last chunk.',
      'KV migration to decode reads from lane or pool pages instead of a fixed slot.',
    ],
    notes: 'Prerequisite for the KV offload tiers. The lane count limits how many requests can share a batched chunk.',
  },
  host: {
    what: 'Two KV tiers behind the device pool or the static slots: host DRAM, then NVMe SSDs in the galaxy hosts. With the pool, pages the device evicts move to host DRAM over PCIe, host DRAM evicts to SSD, and SSD drops; with static slots, a reclaimed slot moves the same way and is read back when its conversation returns (one copy per conversation, no prefix sharing). When a request hits an offloaded prefix, those pages are read back into its lane while the request waits in the queue. Defaults per galaxy: about 370 GB of the 576 GB host DRAM (8 galaxies), 16 TB of SSD (planned), PCIe 63 GB/s each way, SSD 31.5 GB/s read and 27.2 GB/s write.',
    why: 'M3 KV is about 98 KB per token on device today (bf8 index_k replicated over TP). 4 galaxies hold about 30M tokens on device, but the live working set of AgentX near the throughput limit is several times larger, so the device pool alone still evicts conversations that come back minutes later. Host DRAM adds about 18M tokens per pipeline at 4 galaxies (40M at 8; tier copies store index_k once, about 73 KB per token), and 16 TB of SSD per galaxy adds over a billion. A hit costs a read: a 150k-token prefix is about 11 GB, about 45 ms over 4 × 63 GB/s of PCIe from host DRAM, or about 100 ms from SSD at 4 × 31.5 GB/s, done while the request is queued. In the Oct 8 study 64 TB per galaxy changes nothing on the best configs and 4 TB costs 0.5-11% with roofline kernels (under 0.5% with today\'s kernels); bandwidth matters with roofline kernels (half the SSD bandwidth costs 23-24%, no host DRAM tier 15-18%) and barely with today\'s kernels (under 1%).',
    todo: [
      'SSD page store with its own LRU and an index from block hash to file offset.',
      'Asynchronous device→SSD write-back on eviction and SSD→device read before admission (GPUDirect-style or a pinned host bounce buffer).',
      'Scheduler: start the read when the request arrives; a fetching request must not block the pipeline.',
      'Host DRAM page store (pinned, registered for DMA) in front of the SSD store; demote host LRU pages to SSD.',
      'Measure sustained PCIe and drive bandwidth per galaxy host: the model assumes each chip moves its own KV over its own link (x1 Gen4 for 28 of 32 chips, 63 GB/s each way per galaxy) and the drives run at Gen4 x4.',
      'Watch write endurance: evictions are written continuously under load.',
    ],
    notes: 'Works behind the pool or static slots. With 16 TB of SSD, read bandwidth matters more than capacity (the Oct 1 study, with a 1 TB SSD tier at 64 GB/s, found the opposite). Check how the AgentX rules treat host-DRAM and SSD-backed KV (they cap host DRAM per system).',
  },
  idxdedup: {
    what: 'The index-key cache used by the MSA indexer (one 128-wide key per token per layer, bf8) is stored on all 4 TP columns. That is 544 of the 1632 bytes per token per layer: a third of all KV memory. Store it once per SP row and gather it inside the indexer. The dense layers also allocate a zero-filled index cache, which can be dropped.',
    why: 'KV per token per layer drops from 1632 to 1224 bytes (−25%), so the same device memory holds a third more tokens (more slots, or a bigger pool). KV capacity is what limits goodput on this traffic; the gain is in the Roadmap table. SSD copies already store index_k once.',
    todo: [
      'Allocate index_k sharded over TP (by token blocks) or on one TP column, instead of replicated.',
      'In the indexer, all-gather index keys over TP before scoring, or score per TP shard and merge top-k. The extra traffic is small next to the K/V gather.',
      'Update the KV migration table and the decode-side layout.',
      'PCC.',
    ],
    notes: 'Its value grows with more lanes (more concurrent contexts).',
  },
  async: {
    what: 'Today each stage receives a chunk, computes it, then sends the activation to the next stage (about 63 MB for a 5120-token chunk), and the send blocks the stage. Fitted from runs A/B/C: about 23 ms per chunk at 5120 (8 ms at 2048) of blocking send, plus 14-19 ms of hop latency. With async handoff the send overlaps with computing the next chunk.',
    why: 'The blocking send adds 10-35% to the pipeline period. Its share grows with more stages (fewer layers per stage) and with faster kernels: ×1.34-1.50 at 4 galaxies with today\'s kernels, ×1.89-2.21 at 8 galaxies with roofline kernels. It also cuts TTFT, since 32 hops × 15 ms is about 0.5 s.',
    todo: [
      'Double-buffered D2D send/receive: post the send and start the next chunk immediately; the receiver pre-posts buffers.',
      'Make sure the transfer uses enough links (the activation is spread over the stage\'s chips).',
      'Re-measure with PREFILL_SYNC_PER_CHUNK=0: the per-chunk sync used for timing in the benchmark may be part of the measured blocking.',
    ],
    notes: 'Independent of the other features. Its value grows as the kernels get faster.',
  },
  batch: {
    what: 'Put several requests\' new tokens into one chunk, up to a token budget (8-16k works best). Projections, norms and the whole MoE run on the concatenated tokens; attention runs per request (sequential).',
    why: 'Most AgentX requests are small (median 1600 new tokens), so most chunks are small. An MoE layer has a large per-chunk cost that does not depend on tokens: expert weights (16 experts × 32 MB per chip per layer at [2,4]) and about 10 collectives per layer with ~40 µs latency each. More tokens per chunk amortise it. At chunk 2048 each expert sees only about 64 tokens (T×4/128), far below the 260-460 tokens per expert where the matmuls become compute-bound. Measured: ×1.05-1.42 when added, ×1.12-1.64 when removed from the full stack, at slightly higher TTFT. With sequential attention it does not matter which requests share a chunk.',
    todo: [
      'Chunk metadata per segment: lane/slot, start position, length.',
      'An attention loop over segments; each segment\'s KV is read and written in its own lane.',
      'MoE dispatch/combine buffers and activations sized for the budget instead of the chunk.',
      'Traces per chunk size (or a few buckets, e.g. 4k/8k/16k), so a partly filled batch is not padded to the budget. With one static 16k shape today\'s config (about 1.9 requests per batch) is 55% padding and falls below no batching; on the best stacks, which fill batches better, it costs 0-2%.',
      'One attention call per request per chunk, so a request\'s several chunks share one KV-prefix gather.',
      'A packing scheduler: fill the budget, split long requests across chunks, keep segments of one request in order.',
      'Needs request padding to 32·SP (or a small chunk) for token packing; with chunk padding only whole chunks can be packed.',
    ],
    notes: 'Each request in a chunk needs a lane on the stage, so batching wants more (or variable-size) lanes. Fused multi-user attention adds up to 11% on top (most at 4 galaxies with today\'s kernels).',
  },
  reqPad: {
    what: 'Today each request\'s tokens in a batch are padded to a multiple of the chunk C, which is also the block-cyclic KV slab: C/SP rows per SP rank. The host rotates a chunk\'s tokens so that each rank computes the rows it owns (tt-metal #57636), so the KV write is local and a chunk may start at any 32-token boundary. Request padding to 32·SP pads them only to whole 32-row tiles on every rank: they are split evenly over SP, and each layer\'s new K/V rows go to the ranks that own them with a small all-to-all.',
    why: 'It removes padding: 15% of processed tokens at chunk 2048, and about half at 5120, on the AgentX mix. It enables token-packed batching. The all-to-all is modelled at about 0.1 ms per layer, 0.2-0.5% of a batched layer (not measured).',
    todo: [
      'Host token rotation and MoE padding config for batches in which a request\'s tokens are not a whole number of chunks (#57636 covers a full chunk and a ragged last chunk).',
      'An all-to-all KV write op (the KV slab is 128·SP, since MSA needs 128-row KV blocks per rank).',
      'MoE buffers sized for the maximum budget; compile/trace buckets for variable token counts.',
    ],
    notes: 'Most of its value is realised together with batching. A small chunk plus batching gets nearly all of it, as long as a request\'s chunks in one batch form one attention call (its prefix is gathered once). With the same attention, chunk 128-256 plus batching matches 32·SP padding in every scenario (within 0.3% with fused attention; within 0.4% with per-request attention), and chunk 1024 is about 3% behind (tools/layout_ab.js, best configs of the third Oct 8 study; at 4 galaxies with roofline kernels, where SSD reads bind, every chunk from 128 to 2048 is within 0.3%). Chunks below 2048 are extrapolated (calibrated at 2048 and 5120). With the 128·SP KV slab, computing each token on the rank that owns its KV row instead of the all-to-all (placement: owner) is 0-4% behind with fused attention, and chunk padding at C = 128·SP is 1.6-2.5% behind (tools/placement_ab.js).',
  },
  arena: {
    what: 'Instead of fixed 1M lanes, give each request a lane of its actual length from one contiguous arena per stage (2-8M tokens).',
    why: 'Frees the memory that fixed lanes reserve (4 × 1M tokens per stage) for the pool, and lets many small requests be in flight at once for batching.',
    todo: [
      'Kernels take a base offset + length instead of a slot index.',
      'An allocator with fragmentation handling (compaction, or size classes).',
    ],
    notes: 'Its value depends on how many lanes batching needs; see the "4 fixed 1M lanes" sensitivity row.',
  },
  msa: {
    what: 'Today every MSA layer all-gathers the whole cached K/V and index keys over SP for each request (ag_kv / ag_index_k). Instead, each SP rank scores its own index keys, a small top-k merge picks the 16 blocks per query, and only those blocks are fetched.',
    why: 'The gather grows with context: at [2,4] about 0.007 ms per 1k cached tokens per layer, so about 1 ms per layer at 140k, against an 18 ms layer today. The gain is ×1.00-1.04 in the study at SP=2 (when added and leave-one-out); it would matter more with larger SP or much faster kernels.',
    todo: [
      'A distributed top-k merge.',
      'A block-fetch op for the selected K/V blocks.',
      'Changes to the indexer_score_msa / sparse_sdpa_msa inputs.',
    ],
    notes: 'Not worth it at [2,4] stages.',
  },
  fused: {
    what: 'One attention kernel over all requests in a batched chunk, instead of one attention call per request.',
    why: 'It could save launches and fill cores better for short segments. The model includes both effects (per-op latency and a 110-core wave-quantisation penalty for short segments). It finds ≤11% (×1.00-1.11 leave-one-out), because batched segments average about 1.6k tokens, which already fill the cores.',
    todo: ['New multi-user MSA and ring-joint kernels.'],
    notes: 'Not recommended: sequential per-request attention is as good. That confirms the "batch the MoE, attention per request" plan.',
  },
  srpt: {
    what: 'Replace round robin (the base, as tt-d-gen) with run to completion, the queue ordered by new tokens (shortest first) with 30 s aging so long requests are not starved.',
    why: 'Short requests stop waiting behind 50k-token prefills, so p90 TTFT drops. Round robin (the base) already does most of that, so on top of it shortest-first is ×0.98-1.00.',
    todo: ['Scheduler-only change.'],
    notes: 'Cheap; its gain grows near saturation.',
  },
};
