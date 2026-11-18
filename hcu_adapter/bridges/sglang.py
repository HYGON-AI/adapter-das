# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Legacy sglang compatibility chain behind an explicit install boundary.

Bridge rule (see skills/hcu-adapter-dev): when rebinding a host operator that
has a tier in sglang's kernel catalog, import the PACKAGE face
(``sglang.kernels.ops.<domain>``) so the fused-op dispatch picks the backend.
Importing the module body (``sglang.kernels.ops.<domain>.<module>``) bypasses
that dispatch and pins one implementation; it is allowed only with an
inline ``JIT 唯一实现`` note (or a pointer to the alternative tier).
"""

_JIT_BODY = "sglang.kernels.ops.activation.activation"


def _bind_deepgemm_legacy_silu() -> bool:
    """Rebind the DeepGEMM runner's `_legacy_silu_and_mul` through the package face.

    Upstream binds ``_legacy_silu_and_mul`` at import time:

        if not (_is_npu or _is_hip) and _is_cuda:   # JIT module body
        elif _is_musa: ...
        else: _legacy_silu_and_mul = None            # HIP pre-seam

    The adapter's platform seam answers CUDA for sglang, so ``is_hip()`` is
    False by the time the runner is imported and upstream binds the JIT module
    body — that is the 33s ninja build (``sgl_kernel_jit_activation_*``)
    observed in the L1 kernels segment. Only a boot order that imports the
    runner before the seam leaves ``None``.

    The bridge therefore replaces **exactly the JIT module-body function** with
    the PACKAGE face (``sglang.kernels.ops.activation``). The fused-op dispatch
    then runs on the real platform: its JIT candidate is CUDA-only and the
    fused-op capability gate keeps upstream's HIP truth (see the capability
    bridge below), so the call resolves to AOT = our shadow ``sgl_kernel`` op
    and continues through HCU_SILU_MUL_IMPL (auto = lightop > aiter > triton;
    explicit lightop|aiter|triton|torch) like every other ``silu_and_mul``
    call. Any other host binding is left untouched.
    If upstream reorders the priority the package face follows automatically;
    the bridge never pins an implementation.

    Returns True when the runner carries a usable binding after the call,
    False when the host package is absent.
    """
    import logging

    log = logging.getLogger("hcu.adapter")
    try:
        import sglang.srt.layers.moe.moe_runner.deep_gemm as _dgr

        bound = getattr(_dgr, "_legacy_silu_and_mul", None)
        if bound is not None and getattr(bound, "__module__", "") != _JIT_BODY:
            return True
        from sglang.kernels.ops.activation import silu_and_mul as _hcu_silu
    except ImportError:
        return False
    except Exception:
        log.warning("HCU adapter: silu_and_mul binding failed", exc_info=True)
        return False

    _dgr._legacy_silu_and_mul = _hcu_silu
    log.info(
        "HCU DeepGEMM bridge: _legacy_silu_and_mul bound to "
        "sglang silu_and_mul (replaced %s)", "JIT body" if bound else "None"
    )
    return True


def hcu_diffusion_platform_plugin():
    from hcu_adapter.sglang_diffusion import platform_plugin

    return platform_plugin()


def __getattr__(name):
    if name == 'HcuDiffusionPlatform':
        from hcu_adapter.sglang_diffusion import get_platform_class

        return get_platform_class()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def install() -> None:
    """Install the existing sglang bridge sequence without changing its order."""
    """HCU compatibility bridges for the upstream sglang stack.

    Loaded automatically (as `sitecustomize`) whenever PYTHONPATH includes this
    directory, including spawn'd sglang scheduler children. sglang and vllm files
    stay untouched; what we bridge here:

    1. vllm._custom_ops.rms_norm / fused_add_rms_norm -> sgl_kernel triton kernels.
       The HIP vllm build's `torch.ops._C` extension does not carry these ops, so
       sglang's layernorm forward_hip would crash at call time (the import-time
       `_has_vllm_rms_norm` check passes). We replace the python wrappers with
       bridges to our kernel — same signatures, fp32-accumulated math.
    2. `aiter` stub registration happens via `import sgl_kernel` (see
       sgl_kernel._aiter_hcu_stub) for every AMD-only import on the HIP path.
    3. DSA model overrides (`sglang.srt.arg_groups.model_overrides.deepseek_v2`,
       which also owns Glm5Next) force `page_size=1` on legacy HIP (no aiter
       preshuffle paged-MQA). Models whose indexer pools keys (index_kpool > 1,
       e.g. GLM-5.3-Flash kpool=4) then die on `page_size % index_kpool != 0`.
       The memory pool only needs a multiple of index_kpool, so we wrap the
       override registry to keep a CLI-provided page_size when upstream forces 1.
    """

    import logging
    import os
    import sys

    import torch

    if getattr(torch.version, "hip", None) is not None:
        # is_cuda semantics hard-depends on vllm only for `_custom_ops` symbol
        # stubs (bound elsewhere). The vllm_hcu platform plugin, however,
        # hangs the process on this stack (apply_platform_patches under our
        # import order — observed live on validation-host-26 dtk26041: "Platform plugin hcu
        # is activated" then silence until watchdog). sglang resolves its own
        # platforms via the seam below; the plugin is dead weight here.
        # Gate value must be a NON-empty allowlist (vllm envs treats an unset
        # var as None = load-all, and "" would be racy against vllm envs being
        # imported before sitecustomize); "none" is a name no real plugin has,
        # so the filter keeps nothing regardless of import order. Operator can
        # still opt in by exporting VLLM_PLUGINS explicitly (setdefault).
        os.environ.setdefault("VLLM_PLUGINS", "none")
        # Common bootstrap already logged the registry and installed the
        # topology/tvm-ffi guards. Keep the resolver handle used below.
        try:
            from hcu_adapter import hcu_env as _he
        except Exception:
            _he = None
        # Optional code-object repair must precede DeepGEMM/SGLang aliases.
        try:
            from hcu_adapter import deepgemm_cov5

            deepgemm_cov5.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: requested DeepGEMM COV5 repair not installed",
                exc_info=True,
            )
        # deep_gemm name availability (DSV4 family): sglang.srt.models.deepseek_v4
        # (and _dspark) import deep_gemm at module level, and the FIRST registry
        # scan can happen as early as the sections below importing sglang.srt.
        # Without the alias the scanner logs "Ignore import error ... No module
        # named 'deep_gemm'" and caches the modules missing (the registry
        # backfill section further down heals the cache, but the ignore line
        # stays). Alias the DTK `deepgemm` wheel into sys.modules BEFORE any
        # sglang.srt import so the first-pass scan imports them cleanly.
        try:
            import deep_gemm  # noqa: F401
        except ImportError:
            try:
                from hcu_adapter import deep_gemm_dtk as _dgd_early

                _dgd_early.install()
            except Exception:
                logging.getLogger("hcu.adapter").warning(
                    "HCU adapter: early deep_gemm alias not installed", exc_info=True
                )
        # platform semantics seam (governance §0.5 is_cuda route) MUST be the
        # first sglang-touching step: common.py freezes FP8_E4M3_MAX from
        # is_hip() at import time and every module's `from ... import is_hip`
        # binds constants then too. Everything below imports sglang.
        try:
            from hcu_adapter import platform_seam

            platform_seam.install()
            from hcu_adapter.bridges._sglang import device_visibility

            device_visibility.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: platform semantics seam not installed", exc_info=True
            )
        # Bind the parent launch context after platform selection, before
        # downstream imports can cache its scheduler launch entry.
        try:
            from hcu_adapter import sglang_spawn_device

            sglang_spawn_device.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: scheduler device bridge not installed", exc_info=True
            )
        # TP init-broadcast payload fix (torch2.11 dlpack readonly BufferError);
        # runtime binding retains imported aliases without source writes;
        # needs sglang.srt.utils.common importable -- after the seam above.
        try:
            from hcu_adapter import dist_bc

            dist_bc.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: dist broadcast bridge not installed", exc_info=True
            )
        # cold-JIT watchdog floor (box16 cold-boot fatality, 2026-10-25): cold
        # caches + stock 300s watchdog = SIGQUIT mid compile storm. Wraps the
        # watchdog factory BEFORE scheduler.py from-imports it; explicit
        # --watchdog-timeout always wins (watchdog_cold.py).
        try:
            from hcu_adapter import watchdog_cold

            watchdog_cold.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: cold-JIT watchdog floor not installed", exc_info=True
            )
        # KV-pool stride tensor fix (torch2.11 numpy-scalar dtype infer); text
        # patch per governance §3 tier-3, allowlist entry "kv_pool_strides".
        try:
            from hcu_adapter import kv_fix

            kv_fix.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: kv stride text patch not installed", exc_info=True
            )
        # Compile adapted csrc/include assets from an immutable adapter cache.
        # Community sources remain untouched; cached function aliases are retained.
        try:
            from hcu_adapter import jit_sources

            jit_sources.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: immutable JIT compiler inputs not installed", exc_info=True
            )
        try:
            from hcu_adapter.bridges._sglang import jit_eh_norm

            jit_eh_norm.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: JIT EH norm launch bounds not installed", exc_info=True
            )
        # DSA top-k v2 fusion (dsv41f decode): stride seam on the upstream JIT
        # topk_v2 kernel + the v41 full-topk fused decode arm. Replaces the
        # torch.topk fallbacks behind 57 mbtopk kernels/step (2026-10-22 trace).
        try:
            from hcu_adapter import dsa_topk_v2

            dsa_topk_v2.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: dsa topk v2 fusion not installed", exc_info=True
            )
        # tvm-ffi fp8 dtype aliasing for the DSV4 JIT MoE activation kernels
        # (silu_and_mul_{masked,contig}_post_quant): the kernels declare the
        # fp8 output as uint8; CUDA torch fp8 tensors present as uint8 through
        # DLPack, DTK ones present as float8_e4m3fn and the type match rejects
        # them. View fp8 args as uint8 before module.run (byte-identical; the
        # kernel writes raw fp8 patterns into the same storage).
        try:
            from hcu_adapter import jit_fp8_ffi

            jit_fp8_ffi.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: jit fp8 ffi alias not installed", exc_info=True
            )
        # JIT build progress reporter: one stderr/journal line per real ninja
        # compile (compiling/built/FAILED) so a cold 35-module JIT storm is
        # tellable from a wedge in gate logs; the gate heartbeat tails the
        # journal (ci/l1_gpu_gate.sh run_seg). Observability only -- a missed
        # install costs a log line, never numerics.
        try:
            from hcu_adapter import jit_build_log

            jit_build_log.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: jit build progress reporter not installed",
                exc_info=True,
            )
        # DeepSeek-V4.1-Flash checkpoints carry architectures
        # ["DeepseekV41ForCausalLM"], a name no sglang tree registers (checked
        # v0.5.22 and upstream main); the V4 class serves V41 configs via its
        # model_type == "deepseek_v41" gates. Insert the alias directly into
        # ModelRegistry.models -- deliberately NOT via get_supported_archs(),
        # which would populate the registry cache before the DeepGEMM DTK
        # bridge (section 9) makes `deep_gemm` importable; deepseek_v4 would
        # then be cached-missing and even the V4 arch itself stops resolving.
        # A dict insert survives the later `import_model_classes(...) | self.models`.
        try:
            import sglang.srt.models.registry as _dreg

            if "DeepseekV41ForCausalLM" not in _dreg.ModelRegistry.models:
                try:
                    import deep_gemm  # noqa: F401
                except ImportError:
                    # deepseek_v4 imports deep_gemm at module level; the DTK
                    # bridge (section 9) has not run yet at this point.
                    from hcu_adapter import deep_gemm_dtk as _dgdb

                    _dgdb.install()
                import sglang.srt.models.deepseek_v4 as _dsv4

                _dreg.ModelRegistry.models.setdefault(
                    "DeepseekV4ForCausalLM", _dsv4.DeepseekV4ForCausalLM
                )
                _dreg.ModelRegistry.models["DeepseekV41ForCausalLM"] = (
                    _dsv4.DeepseekV4ForCausalLM
                )
                logging.getLogger("hcu.adapter").warning(
                    "HCU adapter: DeepseekV41ForCausalLM -> DeepseekV4ForCausalLM "
                    "registry alias installed (checkpoint arch ahead of upstream)"
                )
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: deepseek_v41 registry alias not installed",
                exc_info=True,
            )
        # CUDA-only attention backends (fa4 = CUTLASS cute DSL SM90a;
        # flashinfer = is_flashinfer_available() False yet unguarded names):
        # fail at the backend-registry seam (constructor-time ImportError) so
        # upstream attention-kit tests skip "backend not available" instead of
        # dying mid-construction/mid-forward.
        try:
            from hcu_adapter import attn_backend_gate

            attn_backend_gate.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: attention-backend registry gate not installed",
                exc_info=True,
            )
        # Breakable prefill CUDA graph (sglang main): its runtime probes
        # stream-capture status through cuda-python, which needs a CUDA
        # driver HCU does not have (error 35 mid capture, bw47 graph/mla
        # 2026-11-05). Driver-layer HIP truth: route breakable prefill to
        # the eager runner; decode FULL graphs stay on.
        try:
            from hcu_adapter import prefill_bcg_gate

            prefill_bcg_gate.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: prefill BCG gate not installed",
                exc_info=True,
            )
        # DeepSeek-V3 fused QKV-A GEMM min-latency path (sglang main): its
        # eligibility predicate sees the seam's CUDA answer plus sm 93 and
        # routes decode through a TensorRT-LLM Hopper kernel whose body is
        # __CUDA_ARCH__ >= 900 only and which asks for ~192 KB dynamic LDS
        # against a 64 KB/block cap -- graph capture dies in
        # set_max_dynamic_smem (invalid argument). Eligibility -> False on
        # HIP, each layer keeps its own linear.
        try:
            from hcu_adapter import fused_a_gemm_gate

            fused_a_gemm_gate.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: fused A-GEMM eligibility gate not installed",
                exc_info=True,
            )
        # Debug tooling (default OFF): in-graph side-buffer probe for the
        # graph-vs-emu corruption hunts — see hcu_adapter/probe_side.py. Only
        # loads when HCU_SIDE_PROBE is set (replay|eager); never in production.
        if os.environ.get("HCU_SIDE_PROBE"):
            try:
                from hcu_adapter import probe_side

                probe_side.install()
            except Exception:
                logging.getLogger("hcu.adapter").warning(
                    "HCU adapter: side-buffer probe not installed", exc_info=True
                )
        # Debug tooling (default OFF): request-boundary state snapshot probe for
        # the R-01/R-06 cross-request leakage hunt — see
        # hcu_adapter/state_snapshot.py. Only loads when HCU_STATE_SNAPSHOT is set.
        if os.environ.get("HCU_STATE_SNAPSHOT"):
            try:
                from hcu_adapter import state_snapshot

                state_snapshot.install()
            except Exception:
                logging.getLogger("hcu.adapter").warning(
                    "HCU adapter: state snapshot probe not installed", exc_info=True
                )
        # Model-aware env defaults (R-08 B-tier, default on via HCU_MODEL_ENV):
        # DSA models get their two fusion escape hatches, fp16 GDN bottoms get
        # the conv-dtype fix — derived from model_config, setdefault only.
        try:
            from hcu_adapter import model_env

            model_env.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: model-env defaults not installed", exc_info=True
            )
        # fused-MoE triton tile guard: HCU 64KB LDS vs the ~72KB NV default
        # config — clamp oversized tiles at the durable resolution seam.
        try:
            from hcu_adapter import moe_config_guard

            moe_config_guard.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: moe config smem guard not installed", exc_info=True
            )
        # MLA-absorb fp8 bmm: upstream gemm.bmm_fp8 spec resolves to
        # fp8_utils:bmm_fp8 which only exists behind flashinfer import. Supply a
        # torch-native dequant-bmm (scales applied in the epilogue role).
        try:
            from hcu_adapter import bmm_fp8_bridge

            bmm_fp8_bridge.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: bmm_fp8 bridge not installed", exc_info=True
            )
        # R-09(2/3): aiter per-token fp8 quant vendor tier -- rowwise wrapper
        # NameError fix (HIP has no sgl_per_token_quant_fp8), group plain arm
        # 15.1x, scaled_fp8_quant natives via upstream's own aiter calls, and
        # the whole-row fused silu+quant arm (HCU_QUANT_IMPL, auto default).
        try:
            from hcu_adapter import quant_aiter

            quant_aiter.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: aiter fp8-quant tier not installed", exc_info=True
            )
        # R-09(1): aiter fused AR+RMSNorm -- upstream layernorm carries the
        # whole consumer chain dormant behind _use_aiter; re-point the
        # flashinfer entry its else-branch already calls at the aiter
        # GraphSafe fused kernels (HCU_AR_NORM_IMPL, auto default; full-TP-
        # group-only, per-call decline to stock).
        try:
            from hcu_adapter import ar_norm_aiter

            ar_norm_aiter.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: aiter fused AR+RMSNorm tier not installed",
                exc_info=True
            )
        # aiter JIT cache observability: the vendor wheel ships every op
        # prebuilt (fresh import ~seconds, zero compiles); surface the wheel
        # version and any runtime-built artifacts so they get harvested or the
        # wheel version pinned (see the module docstring for the harvest rule).
        try:
            from hcu_adapter import aiter_jit_cache

            aiter_jit_cache.report()
        except Exception:
            logging.getLogger("hcu.adapter").debug(
                "HCU adapter: aiter jit cache report skipped", exc_info=True
            )
        # triton extra.cuda libdevice -> hip/OCML redirect (is_cuda semantics
        # picks the cuda libdevice package; hcu backend rejects __nv_* LLVM).
        try:
            from hcu_adapter import triton_libdev

            triton_libdev.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: triton libdevice redirect not installed", exc_info=True
            )
        # strip triton-emitted griddepcontrol.* asm (capability>=9 enables PDL
        # upstream; gfx936 assembler rejects the instructions, and removing
        # them == PDL-off semantics).
        try:
            from hcu_adapter import triton_pdl

            triton_pdl.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "hcu adapter: triton pdl stripper not installed", exc_info=True
            )
        # KDA prefill + causal_conv1d router (2026-10-16): vendored sglang-das
        # hcu triton kernels (hcu_adapter/fla_hcu/) — parity-gated
        # replacements for the gfx938-broken stock kernels (chunk_kda wrong Aqk;
        # conv1d update noise / fn dead initial-state arm). Kernel-level wins:
        # prefill 473-1009x, conv fn 16.4x, update 4.3x, cos >= 0.99999 vs the
        # emus — but the SERVING gate turned red (Berlin anchor flips
        # deterministically under =fork across clean boots while every standalone
        # gate stays green, incl. serving shapes and strided state views), so emu
        # keeps the defaults and fork is opt-in only until a serving-side dump
        # compare finds the divergence. HCU_KDA_PREFILL_IMPL / HCU_CONV1D_IMPL =
        # emu (default) | fork | stock.
        try:
            from hcu_adapter import fla_hcu_router

            fla_hcu_router.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: fla-hcu router not installed", exc_info=True
            )
        # fa3 attention vendor routing (DTK flash_attn wheel; emu fallback).
        # Needs sgl_kernel.flash_attn importable (below pulls it in).
        try:
            from hcu_adapter import fa_vendor

            fa_vendor.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: fa vendor router not installed", exc_info=True
            )
        # GDN / gated-delta-rule linear attention aiter routing (opt-in via
        # HCU_LINEAR_ATTN_GDN=aiter; default keeps upstream triton fla).
        # Requires sglang importable (imports the fla modules it rebinds).
        try:
            from hcu_adapter import gdn_aiter

            gdn_aiter.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: gdn aiter router not installed", exc_info=True
            )
        try:
            from hcu_adapter import gdn_cutedsl_fallback

            gdn_cutedsl_fallback.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: gdn CuTeDSL fallback not installed", exc_info=True
            )
        try:
            # Unconditional rebind: torch's lazy op namespace can pass an
            # import-time probe and still miss the native symbol at call time.
            from hcu_adapter.bridges import vllm as _vllm_bridge

            _vllm_bridge.install_rms_norm_bridge()
        except Exception:  # pragma: no cover - vllm absent: sglang falls back natively
            pass

        # v0.5.22's layernorm face otherwise compiles CUDA-source JIT kernels.
        try:
            from hcu_adapter import sglang_layernorm as _sgl_layernorm

            _sgl_layernorm.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang layernorm route not installed", exc_info=True
            )

        try:
            import sgl_kernel as _sk

            if not hasattr(_sk, "apply_token_bitmask_inplace_cuda"):
                # sglang's xgrammar shim takes the HIP branch and imports this from
                # sgl_kernel; bridge it to the in-tree triton op (CUDA branch).
                # 替代档: 本文件 remap 到 triton（sgl_kernel 无 HIP bitmask 符号）
                from sglang.kernels.ops.grammar.bitmask_ops import (
                    apply_token_bitmask_inplace_triton as _bitmask_triton,
                )

                _sk.apply_token_bitmask_inplace_cuda = _bitmask_triton
        except Exception:
            pass

        # fp8 quant path bridges (HCU_FP8_TRACE probe, native per-token quant bug, group-quant dtype) -> hcu_adapter/bridges/sglang_fp8_quant.py
        try:
            from hcu_adapter.bridges import sglang_fp8_quant

            sglang_fp8_quant.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_fp8_quant bridge not installed", exc_info=True
            )

        # kernel-selector CUDA capability check HIP acceptance (AOT sgl_kernel) -> hcu_adapter/bridges/sglang_kernel_selector.py
        try:
            from hcu_adapter.bridges import sglang_kernel_selector

            sglang_kernel_selector.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_kernel_selector bridge not installed", exc_info=True
            )

        # CT + explicit w8a8_int8 consumers GEMM import backfill -> hcu_adapter/bridges/sglang_w8a8_int8.py
        try:
            from hcu_adapter.bridges import sglang_w8a8_int8

            sglang_w8a8_int8.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_w8a8_int8 bridge not installed", exc_info=True
            )

        # FlashMLA construct-only stubs + sgl-flash-mla wheel face skip contract -> hcu_adapter/bridges/sglang_flash_mla.py
        try:
            from hcu_adapter.bridges import sglang_flash_mla

            sglang_flash_mla.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_flash_mla bridge not installed", exc_info=True
            )

        # fa3 attention backend torch-emulation route (SGLANG_USE_SGL_FA3_KERNEL) -> hcu_adapter/bridges/sglang_fa3_backend.py
        try:
            from hcu_adapter.bridges import sglang_fa3_backend

            sglang_fa3_backend.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_fa3_backend bridge not installed", exc_info=True
            )

        # moe_wna16 quant config non-linear layer contract restore -> hcu_adapter/bridges/sglang_moe_wna16.py
        try:
            from hcu_adapter.bridges import sglang_moe_wna16

            sglang_moe_wna16.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_moe_wna16 bridge not installed", exc_info=True
            )

        # HCU_DUMP_INT8 serve-time W8A8Int8 apply audit (torch recompute + cos/maxerr log) -> hcu_adapter/bridges/sglang_int8_audit.py
        try:
            from hcu_adapter.bridges import sglang_int8_audit

            sglang_int8_audit.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_int8_audit bridge not installed", exc_info=True
            )

        # family-override modules _register_for interception (incl. deepseek_v2 DSA branch) -> hcu_adapter/bridges/sglang_family_overrides.py
        try:
            from hcu_adapter.bridges import sglang_family_overrides

            sglang_family_overrides.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_family_overrides bridge not installed", exc_info=True
            )

        # DSA aiter preshuffle paged-MQA gate force-on (page/table/tile decisions only) -> hcu_adapter/bridges/sglang_aiter_preshuffle.py
        try:
            from hcu_adapter.bridges import sglang_aiter_preshuffle

            sglang_aiter_preshuffle.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_aiter_preshuffle bridge not installed", exc_info=True
            )

        # DSV4/GLM swiglu-clamp MoE activation triton route (fused_experts_impl assert relax) -> hcu_adapter/bridges/sglang_swiglu_clamp.py
        try:
            from hcu_adapter.bridges import sglang_swiglu_clamp

            sglang_swiglu_clamp.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_swiglu_clamp bridge not installed", exc_info=True
            )

        # DSV4 quant/rope family bridges (silu-fused-quant, MHC stats, sparse rope/norm, q/k norm+rope) -> hcu_adapter/bridges/sglang_dsv4_quant.py
        try:
            from hcu_adapter.bridges import sglang_dsv4_quant

            sglang_dsv4_quant.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_dsv4_quant bridge not installed", exc_info=True
            )

        # HCU_LAYER_AUDIT per-module output magnitude audit (triage tooling) -> hcu_adapter/bridges/sglang_layer_audit.py
        try:
            from hcu_adapter.bridges import sglang_layer_audit

            sglang_layer_audit.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_layer_audit bridge not installed", exc_info=True
            )

        # slimquant_w4a8 checkpoint auto-rule (HCU_SLIMQUANT gate) -> hcu_adapter/bridges/sglang_slimquant_guard.py
        try:
            from hcu_adapter.bridges import sglang_slimquant_guard

            sglang_slimquant_guard.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_slimquant_guard bridge not installed", exc_info=True
            )

        # DSA indexer top-k sgl-kernel->torch bridge -> hcu_adapter/bridges/sglang_indexer_topk.py
        try:
            from hcu_adapter.bridges import sglang_indexer_topk

            sglang_indexer_topk.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_indexer_topk bridge not installed", exc_info=True
            )

        # HCU store-trace + DSA topk torch-arm probe (debug capture) -> hcu_adapter/bridges/sglang_store_trace.py
        try:
            from hcu_adapter.bridges import sglang_store_trace

            sglang_store_trace.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_store_trace bridge not installed", exc_info=True
            )

        # DSA indexer rotate_activation hadamard + prefill/decode backend validator -> hcu_adapter/bridges/sglang_dsa_guards.py
        try:
            from hcu_adapter.bridges import sglang_dsa_guards

            sglang_dsa_guards.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_dsa_guards bridge not installed", exc_info=True
            )

        # aiter all-reduce gate (custom AR retired; auto-rule + kill-switch precedence) -> hcu_adapter/bridges/sglang_aiter_ar.py
        try:
            from hcu_adapter.bridges import sglang_aiter_ar

            sglang_aiter_ar.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_aiter_ar bridge not installed", exc_info=True
            )

        # gfx936 MQA-logits fp8 emu (paged/ragged torch fallback auto-rule) -> hcu_adapter/bridges/sglang_mqa_emu.py
        try:
            from hcu_adapter.bridges import sglang_mqa_emu

            sglang_mqa_emu.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_mqa_emu bridge not installed", exc_info=True
            )

        # HCU_DUMP_LAYERS per-layer debug capture (2026-10-16) -> hcu_adapter/bridges/sglang_layer_dump.py
        try:
            from hcu_adapter.bridges import sglang_layer_dump

            sglang_layer_dump.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_layer_dump bridge not installed", exc_info=True
            )

        # 8. deep_ep API bridge -> hcu_adapter/bridges/sglang_deep_ep.py
        try:
            from hcu_adapter.bridges import sglang_deep_ep

            sglang_deep_ep.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_deep_ep bridge not installed", exc_info=True
            )


        # 9. DeepGEMM DTK bridge (EP, 2026-10-11) -> hcu_adapter/bridges/sglang_deep_gemm.py
        try:
            from hcu_adapter.bridges import sglang_deep_gemm

            sglang_deep_gemm.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_deep_gemm bridge not installed", exc_info=True
            )


        # 10. DTK aiter fused-MoE bridge (2026-10-11) -> hcu_adapter/bridges/sglang_moe.py
        try:
            from hcu_adapter.bridges import sglang_moe

            sglang_moe.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_moe bridge not installed", exc_info=True
            )

        # 11. Prefill context-parallel bridge (2026-10-12) -> hcu_adapter/bridges/sglang_cp.py
        try:
            from hcu_adapter.bridges import sglang_cp

            sglang_cp.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_cp bridge not installed", exc_info=True
            )

        # 12. GLM-5.3-Flash layer-boundary precision probe (2026-10-16) -> hcu_adapter/bridges/sglang_glm_probe.py
        try:
            from hcu_adapter.bridges import sglang_glm_probe

            sglang_glm_probe.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_glm_probe bridge not installed", exc_info=True
            )

        # 13. GLM-5.3-Flash MHC hyper-connection emu (2026-10-16) -> hcu_adapter/bridges/sglang_mhc.py
        try:
            from hcu_adapter.bridges import sglang_mhc

            sglang_mhc.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_mhc bridge not installed", exc_info=True
            )

        # 14. Spec-decode KV-loc safety (2026-10-17) -> hcu_adapter/bridges/sglang_spec_kv.py
        try:
            from hcu_adapter.bridges import sglang_spec_kv

            sglang_spec_kv.install(_he)
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: sglang_spec_kv bridge not installed", exc_info=True
            )

        try:
            from hcu_adapter import spec_h2d_pin

            spec_h2d_pin.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: speculative pinned metadata not installed", exc_info=True,
            )

        # Native routing needs stable expert slots before its weight sum.
        # Keep the compiled entry and cached aliases; never rewrite source.
        try:
            from hcu_adapter import moe_topk_order

            moe_topk_order.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: native grouped TopK order bridge not installed",
                exc_info=True,
            )

        # Install the model-method binding only after operator/vendor bridges
        # are ready. This imports the community class and retains existing
        # aliases of forward; no early model scan or disk patch is needed.
        try:
            from hcu_adapter import moe_dualstream_gate

            moe_dualstream_gate.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: dual-stream MoE capture gate not installed",
                exc_info=True,
            )

        try:
            from hcu_adapter import qsa_pool

            qsa_pool.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: QSA pool not installed", exc_info=True
            )

        try:
            from hcu_adapter.bridges._sglang import grouped_topk, sampling

            grouped_topk.install()
            sampling.install()
        except Exception:
            logging.getLogger("hcu.adapter").warning(
                "HCU adapter: grouped_topk/sampling bridge not installed", exc_info=True
            )
