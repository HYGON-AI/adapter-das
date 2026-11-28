# vLLM 0.33 unquantized AITER compatibility candidate

This Adapter candidate uses the public
`Platform.get_unquantized_aiter_moe_compatibility()` interface proposed in
OpenDAS/vllm. A Core wheel without that interface continues its existing MoE
selection; installing this Adapter alone does not admit HCU AITER experts.

Select `--moe-backend aiter` for explicit admission. Automatic selection still
obeys the official `VLLM_ROCM_USE_AITER` and `VLLM_ROCM_USE_AITER_MOE` policy.
This does not enable global AITER, alter CDNA identity, or admit quantized AITER
experts. The official selector, method, loading, modular kernel, routing,
prepare/finalize and TP reduction remain Core owners.

Platform identity follows the shared `SUPPORTED_HCU_ARCHES` set (currently
gfx928/gfx92a/gfx936/gfx938). There is no separate MoE architecture allowlist.
Admission additionally requires the installed vendor's
`aiter.fused_moe_router.feasible_layouts(FusedMoeProblem(...))` to include `raw`
for the actual architecture, dtype, activation and local expert geometry.
The Adapter retains BF16/FP16, gated SiLU/GELU, ordinary TP partitions with
supported dimensions, no extra padding, expert bias, LoRA, EP/EPLB/DP/PCP,
router-weight-on-input or fused shared experts. Unsupported explicit selection
fails before conversion; automatic selection retains official fallback. Separate
ordinary shared experts remain with the official runner. Preprocessed runtime
weights are refused because their format provenance is not yet qualified.

Weights remain canonical raw across postprocessing. Official shuffle(16,16) and
vendor asm_shuffle are distinct permutations; relabeling one as the other is
incorrect and can also eliminate a valid Triton fallback. Vendor neutral
SwiGLU `None` is passed only after rejecting non-neutral semantics. Native
execution errors propagate. The process-local publication guard refuses retry
after partial registration, including BaseException and incomplete fork states.

For this provider, startup materializes `VLLM_MOE_SKIP_PADDING=0` before workers
spawn, keeping valid routes for Graph padding. A previously cached conflicting
value causes a clear error. No graph mode or model scheduler is replaced.

CPU contracts: `tests/adapter/test_vllm_moe_aiter.py`. The initial isolated
Core+Adapter stack passed 19 installed CPU contracts and a single-card component
probe: 14 configurations, 56 expert reference states, 14 whole-runner forwards,
42 changed-input Graph replays and 5 manual parameter-update/postprocess checks.
Original cosine/global-relative/absolute gates were retained. All 14 observed
vendor plans were Triton, including E256/H2048/N512/top8. This is not ASM/CK,
distributed TP, public checkpoint reload, model HumanEval32 or performance
acceptance. The candidate remains under integration validation.

## gfx938 admission and validation

Architecture admission and startup padding policy share the platform's HCU
identity set. Adding an architecture only to `supports_config` leaves the default
`VLLM_MOE_SKIP_PADDING=1` in place, so explicit AITER still fails or produces
invalid padded routes. Do not change global AITER or platform-generation
predicates. The previous gfx936/gfx938-only revision (`deb72e3`) and its validation are
recorded below; they are not hardware validation of the other platforms.

The worker queries vendor layout feasibility only after semantic checks, before
weight conversion. It passes `M=1` because the vendor query explicitly normalizes
away token count. Results are cached by PID, architecture, dtype, activation,
local expert count, hidden/intermediate dimensions and top-k. Neither this query
nor startup padding publishes the Adapter execution operator or executes a
vendor kernel. AITER import may register its own operators. Startup padding does
not import AITER. The vendor still chooses the actual per-token execution plan.
No synthetic device identity, weight preparation or on-the-fly repacking is used.

A missing query API, a vendor `NotImplementedError`, or absence of a raw route
refuses admission with a reason: explicit `--moe-backend aiter` fails, while
official automatic selection may try other backends. Unexpected query failures
and native execution errors propagate. The query interface is required; package
presence, FP8 hardware flags and tuned tables alone are not proof of raw MoE
support. Refresh package/configuration in a fresh interpreter. Capability
admission does not certify native numerical or performance support on a device.

The capability-query revision passes 120 CPU contracts (with a substitute only
at the device-dependent vendor query boundary), 160 fresh-process CPU/bridge
regressions, and eight real gfx938 numerical/changed-input Graph cases using the
requested vendor wheel. Its installed baseline reproduces four CPU failures for
gfx928/gfx92a admission and startup padding. The other platform identities have
CPU contract coverage; there is no gfx928/gfx92a native or model result here.

The gfx938 validation uses Torch 2.13, Core 0.33 with the public compatibility
interface, and `aiter==0.1.8+dtk2610.torch2130.2611271727.g273794` from
`https://pypi.sourcefind.cn/nightly/dtk/`. This is the inspected vendor build,
not an exact-version requirement imposed by this bridge. Install into a private
environment rather than changing a shared serving process's dependencies.

The installed baseline rejects the real gfx938 device with the old architecture
message. After the change, 35 CPU contracts pass. The real provider also passes
eight native numerical/Graph cases: BF16/FP16, SiLU/GELU, E8/H256/N128/top2 and
the Qwen3.6-35B-A3B geometry E256/H2048/N512/top8, with three query rows each.
The reference uses independent CPU FP32 selected-expert math with intermediate
dtype storage. Two changed-input replays per case vary hidden states, expert
IDs and routing weights; output canaries and raw-weight identity are checked.

Run the native regression in its own vLLM-host interpreter:

```bash
HCU_ADAPTER_HOST=vllm VLLM_PLUGINS=none \
  python -m pytest -q tests/adapter/cases/vllm_unquantized_aiter_device.py
```

The test belongs to the vLLM host lane, not common/SGLang collection. CPU boundary
coverage uses a vendor-query substitute for all four platform identities; it
does not execute another architecture on the local GPU. Real native regressions
use the actual device and installed vendor query, with no capability substitute.
Component
results do not qualify EP/DP/PCP, quantized experts, other architectures, model
accuracy or performance. Preserve checkpoint/model and shutdown results
separately from these contracts.

On 2026-11-28, the gfx938 single-node TP1/DP1 BF16 text-only run of
`/llm-models/qwen3.6/Qwen3.6-35B-A3B` selected `AiterUnquantizedExperts` and
completed official FULL_AND_PIECEWISE capture (sizes 1/2/4/8), chunked prefill,
13/13 public output probes and 3/3 prefix probes (4224 cache-hit tokens).
Configuration: max context 4096, max sequences 4, batched tokens 1024, explicit
1 GiB KV cache, existing Adapter default attention (`TRITON_ATTN`) and
`--kv-cache-dtype auto`. The installed
Core was `c440c089fe9621c5b03e1618cf86a13e95de8bf1`; unchanged Plugin payload
was `d2dcf41`. Adapter was a development Python wheel on main `92e0ca1` plus
this architecture fix; the existing native kernel payload was `51bd28`.
These are isolated integration sources, not a newly released matched wheel set.

Original HumanEval tasks 0–31 scored **raw 0/32**, **code-normalized 32/32**,
with no length-limited responses. Do not report this as raw HumanEval passing
or use it to certify general model accuracy. Normal SIGINT exit took 13.64s,
within the original 15s gate, with exit code 0, no cleanup signals and no owned
survivors. No performance, distributed or multimodal acceptance was performed.
The reproducible validation scope and vendor version above must accompany these
results; replacing the provider build requires repeating its native/model gates.

The capability-query revision was also rerun on the same Qwen3.6-35B-A3B
checkpoint and deployment settings above, on physical gfx938 GPU6 in a new
private environment. It again selected AiterUnquantizedExperts and completed
FULL_AND_PIECEWISE capture, output probes 13/13 and prefix probes 3/3 (4224 hit
tokens). HumanEval0–31 remained raw0/32, code-normalized32/32 with no truncation.
Normal SIGINT exit took 12.83s within the unchanged 15s gate, exit0, with no
cleanup signals or owned survivors. This is gfx938 revalidation only; the other
platform identities still need their own real-device runs.
