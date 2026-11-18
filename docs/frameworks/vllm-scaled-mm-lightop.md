# vLLM LightOp scaled-MM integration

Recorded before implementation, 2026-11-08.

## Scope and placement

Restore the existing token/tensor-activation and channel/tensor-weight scaled-MM contract on HCU in Adapter. Preserve native quantization, loading, fused partition scale expansion, asymmetric INT8 correction, kernel selection, disabled-kernel checks and platform identity. No model/plugin/core source edits.

Public `register_linear_kernel` remains responsible for the official FP8 fallback. It appends providers, so a new INT8 provider cannot supersede the existing always-supported Triton provider. There is no public execution-backend registration seam. Therefore bind only the `triton_scaled_mm` global in the official scaled_mm.triton module, shared by the existing FP8 provider and INT8 custom op; preserve method/function objects and cached aliases. Keep all LightOp math/scale/layout code in a framework-free kernel module. An opaque custom op with fake output protects FP8 Dynamo/Graph capture.

## Failure modes (before code)

- A/B orientation or noncontiguous/padded weight view changes the dot product; use A[M,K], B[K,N], vendor B.t()[N,K] only when dense.
- FP8 E4M3FN versus FNUZ/E5M2 or INT8 reinterpretation produces silent wrong values; exact operand dtype and real architecture gates.
- FP8 hardware is absent on gfx936; only gfx938 may use vendor FP8, gfx936 retains official Triton.
- Scalar/per-token and scalar/per-channel scales broadcast along the wrong axis; canonicalize validated FP32 scale forms to contiguous [M,1]/[N,1] without host reads.
- Bad dimensions/device/bias/scale dtype or block scales must not reach vendor; retain original Triton behavior for unsupported input contracts.
- Unsupported K/N alignment or zero batch/dimensions may cause launch errors; static unsupported contracts use original fallback.
- Vendor failure status, bad output dtype/device/shape must raise, not retry after a possibly asynchronous fault.
- Asymmetric INT8 zero-point compensation could be lost or doubled; native apply_weights remains intact.
- Quantization/load/transposition could be skipped; exercise actual native init, post-load and forward in the replay.
- Graph could freeze inputs/scales/bias, have host sync, or Dynamo trace vendor internals; opaque custom op/fake and same-address changed-input replay.
- Public registry append cannot reprioritize existing INT8; verify actual calls reach the vendor via the native provider, not only direct helper calls.
- Missing wheel or signature drift; auto uses stock when absent, explicit LightOp fails. Explicit ABI admission is at startup; automatic ABI admission is at the first eligible real GEMM.
- Duplicate install/wrong module ownership or drift; validate native call globals/signatures before rebind; repeated install is a no-op only for the same binding.
- Fast disable/config selects wrong backend; startup `HCU_DISABLE_OPS=vllm_scaled_mm` or `ops.vllm_scaled_mm: triton` preserves the original function.
- Cross-host import side effects; bridge installs only in vLLM HCU runtime, kernel module imports no host framework.

## Predeclared verification

No post-implementation unit tests. A standalone replay script is written before production code. Run in existing nmz30/gfx938 and available gfx936 containers with installed vLLM and Adapter development source, bounded one GPU, no service/container/package changes. Cover both FP8/INT8: 200 seeded cases per dtype, scalar/token/channel scales, BF16/FP16, bias, M=0/1/ragged, aligned and fallback layouts. Compare FP32 decoded dot products with cosine >=0.9999, relative L2 <=0.01 and normalized max error <=0.01 (same existing FP8 probe thresholds). Graph first replays a baseline then changes values/scales/bias in place for three waves. Native entry/weight lifecycle and disabled startup routes are recorded separately. Produce JSONL, summary, source hashes and repeatable command. Component proof is separate from multi-card DSV41 HumanEval and performance qualification.

## Implementation and measured results

Runtime target: `vllm.model_executor.kernels.linear.scaled_mm.triton.triton_scaled_mm`;
registry op: `vllm_scaled_mm`, mechanism: `rebind`, impl: `lightop`. This is an execution
bridge, not a model patch. The existing `vllm_fp8_linear` public registration is retained.
Math/scale normalization lives in `hcu_adapter/kernels/scaled_mm_lightop.py`; installation,
ABI and call-site validation live in `hcu_adapter/bridges/vllm_scaled_mm.py`.

For the supported 0.32 line, INT8 `apply_weights` reads this shared global
directly. The 0.33 line reads it from `_int8_scaled_mm_func`. Inspect the
actual callable and its module ownership before binding; a missing 0.33-only
helper is not a 0.32 startup error. Unknown host call sites retain the official
implementation in `auto`; explicit `lightop` refuses unsupported ownership.
CPU binding regressions cover both call sites and preserve the original method
objects. They do not transfer the historical numerical results below to a new
host version or installation.

| Node / architecture | INT8 | E4M3FN FP8 | Total contract checks |
| --- | --- | --- | --- |
| nmz30 / gfx938 | LightOp, 200 seeds | LightOp, 200 seeds | 410/410 |
| bw46 / gfx936 | LightOp, 200 seeds | Official Triton, 200 seeds | 410/410 |

Each precision includes BF16/FP16 output, bias, scalar/token/channel scales and
12 Graph shapes, with baseline replay then three same-address changed-input waves.
Four nonempty native lifecycle checks use actual selector, quantization, post-load
and forward. Four checks preserve upstream rejection for empty inputs and malformed
row-oriented weight scales; two exercise numerical fallback for unaligned or row-major
weights. These rejection checks are not empty-input numerical support.

For gfx938 eager results: INT8 minimum cosine 0.999997437, maximum relative L2
0.002285651; FP8 minimum cosine 0.999997616, maximum relative L2 0.002207025.
The original thresholds above were not relaxed. The prewritten probe was corrected
to enter the required native VllmConfig context and expect native shape refusals;
initial failure records remain in `.artifacts/scaled-mm-20261108-gfx938-03`.

Real `vllm.plugins.load_general_plugins()` installs the bridge automatically, and
repeat installation returns False. A fresh process with
`HCU_DISABLE_OPS=vllm_scaled_mm` leaves the native function object unchanged.
L0 structure/governance gate passed (31 pre-existing bare-environ warnings).

Final replay artifacts in this checkout:

- `.artifacts/scaled-mm-20261108-gfx938-04/`: protocol/source hashes, 410 case records,
  summary, runtime-hook and kill-switch records, copied replay script and checksums.
- `.artifacts/scaled-mm-20261108-gfx936-01/`: protocol/source hashes, 410 case records,
  summary, copied replay script and checksums.

Re-run in the existing node/container, using a fresh output directory and an idle GPU:

```bash
# On nmz30 (or bw46), inside vllm_032_1030_laibao:
source /opt/dtk/env.sh
unset ROCR_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=3 CUDA_VISIBLE_DEVICES=3
export HCU_ADAPTER_HOST=none VLLM_PLUGINS=hcu_adapter VLLM_ROCM_USE_AITER=0
export PYTHONDONTWRITEBYTECODE=1
# run from the adapter-das checkout root
export PYTHONPATH="$PWD"
timeout 600 python3 -B tests/tools/_probe_vllm_scaled_mm_lightop.py \
  --output .artifacts/scaled-mm-replay-new
```

Normal vLLM runtime requires no new environment variable. `ops.vllm_scaled_mm`
accepts `auto` (default: vendor if importable and contract matches), `lightop`
(require vendor; unsupported contracts still use upstream), or `triton` (startup
rollback). The existing HCU_DISABLE_OPS kill switch wins. Policy changes detected
on reinstallation require a fresh process.

In the 0.33 lazy-loading follow-up, `auto` publishes the same opaque dispatch
without importing LightOp. Its first metadata-eligible real invocation resolves
the provider; fake tracing, an unused GEMM and unsupported contracts do not.
Optional missing imports/entries are remembered and delegate to the original
Triton call. Explicit `lightop` still resolves and validates during startup.
This does not change supported dtypes, architecture admission, math, scales,
native Linear methods or their existing aliases. The existing shared
`ProcessLocalLifecycle` serializes first resolution, resets after a completed
fork and refuses a child inheriting incomplete resolution; an ABI/load failure
refuses later bridge calls, including fallback and explicit tuning arguments.
The original kernel's runtime computation errors retain their existing behavior.

This avoids importing the complete LightOp native library surface merely to
register an unused automatic GEMM for a BF16 model. Other selected providers
may legitimately require LightOp too, so deferred registration alone does not
establish a serving-memory improvement or a clean shutdown. The targeted
contracts in `tests/adapter/test_vllm_scaled_mm_bridge.py` cover unused and
unsupported paths, missing optional providers, explicit admission, sticky ABI
failure, concurrent resolution and a real fork while another thread resolves.
A fresh installed-wheel gfx936 replay passes the unchanged 410-case protocol:
200 eligible INT8 cases execute vendor channelwise GEMM, while FP8 retains the
existing official Triton gate on gfx936. The trace records 96 actual Graph
launches, including changed-input replay. Observed worst relative L2 is
0.002285651 and normalized maximum error is 0.003435369, with minimum cosine
0.999997437; the original 0.01/0.01/0.9999 thresholds are unchanged. Startup has
no LightOp before the first eligible GEMM and loads it during execution when
paired with the Plugin's lazy RMS candidate. This combination also passes all
30 existing Plugin RMS device cases and 104 RMS Graph launches. The six
selected package RECORDs and source/wheel/install payloads were verified.

The local Adapter scratch wheel reuses #140's build suffix solely to preserve
the unchanged native kernel dependency; it is identified by SHA256
`53d931ed5fd6a64d1eecb09383db172d4e74f355a1fb60fda1fc9330fd2bb721`,
not as a new release. All 254 Python payloads retain their original set and only
the scaled-MM bridge differs from the parent artifact. The same sealed Qwen3-1.7B checkpoint also completed an unchanged 32-task
HumanEval before/after comparison with BF16 weights, FLASH_ATTN, E4M3 KV,
TP1 and default Graph/prefix. Both score raw 1/32 and normalized
28/32; all 32/32 messages/finish reasons/usages and pass sets agree.
Each records 3840 prefix-hit tokens and real native E4M3 writer/prefill/decode
kernels. The reference retains LightOp; the combined lazy Adapter/Plugin
candidate has none in the worker before/after generation. The reference already
contains atexit unfreeze, so this pair measures the new import policy.

The strict output-format probes still fail in both arms. Original 15-second
EngineCore forced exit is reference=True, candidate=True; nanobind leak
reports are reference=False, candidate=False. Both API return codes are zero
and device handles are released. Prior Gemma2 lifecycle failures are not closed
by this Qwen comparison. Performance, gfx836 and full upgrade acceptance remain
separate gates. The paired Plugin validation is documented in its
`docs/validation/vllm033-rms-lazy-provider.md`.

## Real service validation

The unchanged Adapter and model-plugin sources were also run on nmz10 (gfx938) with
the DSV41 channel-FP8 checkpoint, TP1/DP8/EP8, DSpark5 and PIECEWISE CUDA graphs;
AITER stayed disabled. HumanEval completed 164/164 with accuracy 0.9634 (158
correct) and exit code 0; the acc_d.sh trap released all eight GPUs afterwards.

Actual operator execution was then audited in a second process with the same
production source hashes and serving parameters. Eight real chat requests
completed, and import-time observers recorded successful returns of the unchanged
vendor function on all eight worker PIDs. Receipts show
`TritonFP8ScaledMMLinearKernel.apply_scaled_mm -> routed -> dispatch ->
scaled_mm_lightop.scaled_mm -> lightop.hipblaslt_w8a8_channelwise_gemm`, for
example M=5, K=5120, N=1792 with FP8 operands and BF16 output. Receipts are
deduplicated per shape/phase/capture state/PID: they are not call totals and do
not count CUDA Graph replay launches. No fallback receipt was observed. The
earlier HumanEval process kept no per-operator receipts, so its hit is inferred
from identical source hashes and serving configuration, not from that process's
own log.

PIECEWISE capture and decoder replay completed, but empty-graph warnings and an
eager draft metadata rebuild were logged, so draft FULL graph capture remains
unproven. A separate nmz30 attempt stopped producing output after weight loading
and the node later became unreachable over SSH; that is an environment
observation and is not attributed to this backend.

## Known limits

Only aligned K/N (multiples of 16), dense A and transposed-dense B, FP32 scales and
BF16/FP16 output enter LightOp. Other contracts delegate to the original implementation,
including its validation errors. FP8 is never sent to vendor on gfx936. Explicit Triton
kernel-tuning arguments preserve native execution. No new behavior is claimed for
FNUZ/E5M2/block scales, unsupported architectures, multi-card serving or other hosts.
Asymmetric INT8 correction remains native but was not exercised by this symmetric
native-lifecycle replay. Graph replay here is the Linear component, not a complete
DSpark/model graph qualification: the DSV41 FP8 run used PIECEWISE graphs while
draft FULL capture is unproven. HumanEval ran once on FP8 without a baseline A/B,
so no performance gain is claimed and no INT8 model accuracy run was made. No Core
or model-plugin source was changed and no package was installed. Changes are
committed locally and not pushed.
