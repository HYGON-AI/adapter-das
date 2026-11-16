# Kimi K3.1 prefill delivery


## Kimi K3.1 prefill optimization delivery (2026-11-16)

Purpose: preserve the measured16/32-GPU optimization with fixed SGLang
`kimi-k3.1-day-0@ce0d5d18abfc3c15d8d526453c4f00b24f45e4c6`, without
editing upstream source. Source hashes: `docs/kimi31-prefill-source-manifest.json`.
Adapter owns GetKAndS classmethod/page-table compatibility, reduced-head FP32
MQA contraction, opt-in group64 quantization, zeroing nonlocal EP slots,
source-only JIT root detection and the optional layernorm-class guard.
Plugin owns early BLAS handle warmup,96GiB Torch allocator budget before
weight loading and the retained diagnostics hooks. Load adapter before plugin.

Enable `HCU_KIMI31_DAY0=1`, `HCU_MQA_RAGGED_REDUCE_HEADS=1` and
`HCU_SLIMQUANT_FUSED_GROUP_QUANT=1` for the measured combination.
Adapter performance flags default off and retain the existing paths.
Plugin budget defaults to96GiB; set `KIMI_DEVICE_BUDGET_GIB=144` on the
historical16-GPU144GiB profile. Allocator limits exclude native allocations.

Historical full-matrix results (not a new benchmark of these local commits):
16GPU TP8/PP2/CP8/EP8, C1=8.184006/13.919462/25.626389s;
32GPU TP16/PP2/CP1/EP1, C1=6.471078/10.955740/20.003728s for16K/32K/64K
inputs, output1, cache off. Each final matrix passed15 cases/93 requests;
six sampled out16 requests and the64K out16 check passed. These were measured
from the pinned source snapshots, not from the local working trees with
unrelated KDA changes. Graphs were disabled; no new graph or dataset claim.
32GPU used physical143.98GiB cards with96GiB software budget, static fraction
0.58 and sampled peak91.603GiB; this is not real96GiB hardware emulation.
The old16GPU profile used static fraction0.90 and different networking.

Use a complete matching gfx938/CU64 native rocBLAS directory through
`ROCBLAS_TENSILE_LIBPATH`;16GPU uses BLAS TN6016x8192x7168,32GPU uses
TN5248x8192x7168, BF16 inputs/FP32 output. Disable selection by unsetting
that variable; never overwrite system libraries. The complete device-specific
library and raw logs remain external artifacts; their manifests/selected
kernels are recorded in the adapter configs/kimi31-prefill directory.
Set `KIMI_WORK_LOG` to the existing1115 work log to find
`parallel-tuning/gemm-native-config/library_gpu6`,
`scale32/gemm5248/gemm-native-config/library_gpu6` and `scale32/REPORT.md`.
CP shares TP ranks: world size=TP*PP*DP. Fixed upstream rejects CP TP>8;
CP8/PP4 was rejected for exact continuation mismatch; EP16/PP2 regressed.

This delivery does not restart or redirect the running service. Local CPU
checks and any failures are recorded in the delivery receipt; historical GPU
results are not substituted for fresh commit-level verification.

Packaging follows the adapter tier naming convention: the measured quantizer
module is renamed to `slimquant_group_quant_triton.py` and its sole import is
adjusted. Kernel bytes remain identical; the source manifest retains both
snapshot and delivery hashes. This naming change has no fresh GPU benchmark.

Local delivery validation: clean-index L0 passed; changed Python ASTs and
source SHA256 manifests verified; staged whitespace check passed. Detailed
results and limitations are in `docs/kimi31-prefill-validation.json`.
