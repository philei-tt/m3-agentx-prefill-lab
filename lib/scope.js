// Rough implementation complexity (low / med / high) and a one-line summary of what each feature takes.
// Used by build_artifact.js (Roadmap table) and tools/feature_table.js (README table). Keys match study.js FEATURES.
'use strict';
module.exports = {
  pool: ['high', 'Paged KV pool + per-stage lane table, page-list gather/scatter op for copy-in/out, LRU, scheduler overlap of copies.'],
  arena: ['med', 'Lanes allocated contiguously at request size from one arena (kernels take offset+length), fragmentation handling.'],
  host: ['high', 'KV offload tiers: async PCIe DMA of KV pages (pool) or whole slots (static slots) to host DRAM, NVMe SSD behind it; write-back on eviction, prefetch while queued.'],
  idxdedup: ['med', 'Store index_k once per SP row instead of on all 4 TP columns; broadcast/all-gather it in the indexer.'],
  segPad: ['high', 'Pad segments to 32·SP instead of the chunk: even split over SP with an all-to-all KV write (or a small chunk), host rotation and MoE padding for any segment length, MoE buffers sized to the budget.'],
  batch: ['high', 'Several requests per chunk: per-segment metadata (slot, start, len), sequential per-request attention, MoE on the concatenated tokens, packing scheduler.'],
  fused: ['high', 'Multi-user attention kernels. The model finds no gain over sequential per-request attention.'],
  async: ['med', 'Non-blocking stage-to-stage D2D (double-buffered send overlapped with the next chunk).'],
  msa: ['high', 'SP-local index scoring + top-k merge, fetch only the selected K/V blocks. Small gain at [2,4].'],
  srpt: ['low', 'Scheduler-only: shortest-new-first run to completion with 30 s aging, instead of round robin.'],
};
