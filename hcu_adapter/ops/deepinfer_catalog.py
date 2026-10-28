"""Initial operator catalog; candidate backends require explicit installation."""

PLANNED_TILELANG_FACTORIES = ()


BACKEND_CATALOG = {
    "runtime.device_visibility": {
        "ordinal_mask": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.hcu_env.align_device_visibility",
            "limitations": "Before GPU initialization, align HIP with upstream numeric physical CUDA assignment without expanding an explicit HIP mask; ROCR/UUID mappings unsupported; native DP topology qualification pending",
        },
    },
    "worker.driver_exposure": {
        "rccl_engram": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.worker_runtime._prepare_collective_engram_driver",
            "limitations": "RCCL Engram has no NVIDIA peer-memory driver requirement; defer HIP initialization to Worker; guard exact active provider classes; original CP8 offline example completed (attempt61), formal quality still pending",
        },
    },
    "platform.device_inventory": {
        "drm": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.platform_guard.install_inventory",
            "limitations": "DRM cardinality through GPU discovery, cache invalidation and spawn startup; no torch/GPU context; NUMA mapping remains unavailable, no SM100 identity",
        },
    },
    "runtime.upstream_defaults": {
        "compatible": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.hcu_env.apply_upstream_defaults",
            "limitations": "Default GPU Engram storage and non-batch-invariant kernels; eager-only installer defaults main/draft eager; explicit environment choices preserved",
        },
    },
    "runtime.configuration": {
        "file": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.configuration.load",
            "limitations": "Strict JSON/YAML installer options; two configurable backend groups and two fixed providers; explicit function arguments win; per-operator hot switching unsupported",
        },
    },
    "runtime.dump": {
        "json": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.runtime_dump.write",
            "limitations": "Atomic per-PID report of actual bindings/catalog/env/version/module paths/source hashes and installer selection sources; never probes GPU",
        },
    },
    "deep_gemm.cublaslt_gemm_nt": {
        "native": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.gemm.cublaslt_gemm_nt",
            "limitations": "BF16 contiguous aligned-K uses qualified LightOp provider, FP32/strided fallback uses FP32 torch matmul; C/D alias and empty-K semantics; FP16/FP8 operands unsupported; vision model pending",
        },
    },
    "tile_kernels.quant.swiglu_forward": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.quant_emu.swiglu_forward",
            "limitations": "Standalone BF16/FP32 V4.1 ViT activation; same FP32/clamp/weight math as fused cast provider; clamp counting unsupported; model qualification pending",
        },
    },
    "tile_kernels.utils.dtype_to_str": {
        "native": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.runtime_config.dtype_to_str",
            "limitations": "Pinned host map: FP32/BF16/E4M3/int8 packed-E2M1 only",
        },
    },
    "worker.cpu_affinity": {
        "inherited_mask": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.worker_runtime.install_cpu_affinity",
            "limitations": "Worker and SO builder share inherited allowed-CPU partitions; avoids NVIDIA procfs GPU census; NUMA optimization pending",
        },
    },
    "worker.bootstrap.context": {
        "inherited": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.bootstrap.publish_worker_context",
            "limitations": "Internal provider context propagated after explicit complete installation; spawned CP qualification pending",
        },
    },
    "worker.bootstrap.import_hook": {
        "lazy": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.bootstrap.register_worker_import_hook",
            "limitations": "No torch import at Python startup; install providers before the engine is unpickled in CP children; qualification pending",
        },
    },
    **{
        "deep_ep." + name: {
            "rccl": {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.bucket." + name,
                "limitations": "RCCL session storage with pinned 2MiB/64B alignments; byte-preserving CP all-gather; numerical/graph/CP model qualification pending; no symmetric-memory registration",
            },
        }
        for name in ("BucketBuffer", "BucketSession", "get_num_allocation_alignment", "get_num_rdma_alignment")
    },
    "flash_attn.cute.interface._flash_attn_fwd": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.vision_attention_emu.flash_attn_fwd",
            "limitations": "Pinned ViT noncausal varlen Q/K/V call; bounded FP32 query tiles; numerical/graph/model qualification pending",
        },
    },
    "worker.runtime": {
        "hip_eager": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.worker_runtime.install",
            "limitations": "Eager by default; explicit allow_cuda_graph=True passed DP8/EP8/ENP8 CP1 NEXTN0 model smoke (attempt44, 2 rounds); formal accuracy/long-context pending; PDL/VMM segments/CuMem RCCL disabled, batch-invariant rejected",
        },
    },
    "platform.bf16_tflops_limit": {
        "guard": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.platform_guard._bf16_tflops_limit",
            "limitations": "MFU denominator only; BW1101 peak not verified, defaults to uncalibrated (MFU 0) unless HCU_BF16_TFLOPS is set",
        },
    },
    "vendors.import_surface": {
        "stub": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.vendor_imports.install",
            "limitations": "Missing fused MLA, HC GEMM and vision imports only; calls fail explicitly",
        },
    },
    "deep_select.topk": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.deepselect_emu.topk",
            "limitations": "Pinned GPU top-k, native stable sort; no e2e/throughput qualification",
        },
    },
    "deep_ep.EngramBuffer": {
        "rccl_gpu": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.deepep_compat.EngramBuffer",
            "limitations": "GPU tables, eager, byte-exact collective lookup; no CPU RDMA/graph/e2e qualification",
        },
        "rccl_gpu_graph": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.deepep_compat.GraphEngramBuffer",
            "limitations": "Explicit graph candidate for GPU tables and fixed query capacity; 8-rank BF16/FP8 changed-query replay and model smoke (attempt44) passed; formal accuracy/long-context pending, no CPU RDMA",
        },
    },
    "deep_ep.EPBuffer": {
        "dtk_v1_expanded": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.deepep_compat.EPBuffer",
            "limitations": "Single-host eager expanded dispatch; no graph, cached dispatch, EPLB or e2e qualification",
        },
        "dtk_v1_expanded_graph": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.deepep_compat.GraphEPBuffer",
            "limitations": "Explicit single-host graph candidate; fixed receive bound and do_cpu_sync=False during capture; largest decode warmup must create the separate fixed-capacity Buffer before capture; 8-rank mixed graph/prefill and model smoke (attempt44) passed; formal accuracy/long-context pending; no cached dispatch/EPLB",
        },
    },
    **{
        "deep_ep." + name: {
            "lsa1": {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.deepep_compat." + name,
                "limitations": "Reports LSA domain 1 on DTK2610 (no RCCL LSA team, R-16 VMM cap); routes MoE off MegaMoE and EPLB off intranode symm-mem shuffle",
            },
        }
        for name in ("get_physical_domain_size", "get_logical_domain_size")
    },
    "tilelang_ops.gather_tl": {
        backend: {
            "status": "candidate",
            "implementation": ("hcu_adapter.kernels.elementwise_triton.gather" if backend == "triton"
                               else "hcu_adapter.kernels.elementwise._torch_gather"),
            "limitations": "Pinned int8 factory contract; no model/e2e qualification",
        }
        for backend in ("triton", "torch")
    },
    "tilelang_ops.norm_tl": {
        backend: {
            "status": "candidate",
            "implementation": ("hcu_adapter.kernels.elementwise_triton.norm" if backend == "triton"
                               else "hcu_adapter.kernels.elementwise._torch_norm"),
            "limitations": "BF16/FP32, FP8 per32/per128; no model/e2e qualification",
        }
        for backend in ("triton", "torch")
    },
    "tilelang_ops.expand_swa_index_tl": {
        backend: {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.index_ops." + ("expand_swa_index" if backend == "triton"
                                                      else "_torch_expand_swa_index"),
            "limitations": "Exact int32 contract incl. wraparound/clipping; no model/e2e qualification",
        }
        for backend in ("triton", "torch")
    },
    "tilelang_ops.fused_norm_rope_hadamard_cast_store_kvcache_tl": {
        backend: {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.fused_transform.fused_transform",
            "limitations": "Norm via norm_tl provider, torch RoPE/cast mirroring the pinned CUDA rounding; E4M3/E2M1 with UE8M0/E4M3/FP32 scales; no Hadamard (no V4.1 caller); no model/performance qualification",
        }
        for backend in ("triton", "torch")
    },
    "tile_kernels.transform.apply_rotary": {
        "triton": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.rope_triton.apply_rotary",
            "limitations": "In-place FP32 fma rotation, NeoX/interleaved/conjugate; 4D input needs mergeable batch/seq strides; no model/performance qualification",
        },
    },
    "tile_kernels.quant.per_token_cast_and_cast_back": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.quant_emu.per_token_cast_and_cast_back",
            "limitations": "In-place E4M3/E2M1 round trip with FP32 or rounded scales; no E4M3 scales or model qualification",
        },
    },
    "tilelang_ops.load_paged_kv_cache_tl": {
        backend: {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.fused_transform.load_paged",
            "limitations": "Torch gather/scatter of compact segments; cast layouts dequantize as bf16(float(q)*sf); no model/performance qualification",
        }
        for backend in ("triton", "torch")
    },
    "tilelang_ops.store_kv_cache_tl": {
        backend: {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.fused_transform.store_paged",
            "limitations": "Torch quantize+scatter mirroring the pinned CUDA store rounding (division-based UE8M0); no model/performance qualification",
        }
        for backend in ("triton", "torch")
    },
    "flash_mla.fused_norm_rope_attn_rope_cast.decode": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.mla_attention_emu.fused_decode",
            "limitations": "Torch sparse MLA decode on V4.1 FP8 (528B) / FP4 (288B) paged caches; FP32 O-RoPE through FP8 cast follows pinned CUDA finish_tile, not Python ref's intermediate BF16; pinned config only; candidate precision correction",
        },
    },
    "flash_mla.fused_norm_rope_attn_rope_cast.prefill": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.mla_attention_emu.fused_prefill",
            "limitations": "Torch sparse MLA prefill mirroring vendor ref (FP32 softmax, chunked gather); pinned V4.1 config only; correctness tier, no performance qualification",
        },
    },
    "tile_kernels.quant.swiglu_forward_and_per_token_cast": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.quant_emu.swiglu_forward_and_per_token_cast",
            "limitations": "FP32 SwiGLU (+clamp, topk weight) then per-token cast; no clamp counting; computes PSUM gap rows (vendor leaves them unspecified)",
        },
    },
    **{
        "tilelang_ops.compress_and_update_state_" + phase + "_tl": {
            backend: {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.compress_emu.compress_" + phase,
                "limitations": "Torch gather/softmax/scatter on device metadata (graph-safe); no model/performance qualification",
            }
            for backend in ("triton", "torch")
        }
        for phase in ("prefill", "decode")
    },
    **{
        "tilelang_ops." + name: {
            backend: {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.index_ops." + impl,
                "limitations": "Torch index op mirroring the upstream *_ref (device-only); no performance qualification",
            }
            for backend in ("triton", "torch")
        }
        for name, impl in (('load_kv_index_tl', 'load_kv_index'), ('kv_index_cat_tl', 'kv_index_cat'), ('convert_kv_index_tl', 'convert_kv_index'), ('candidate_block_max_tl', 'candidate_block_max'), ('get_candidate_compact_lens_tl', 'candidate_compact_lens'), ('restore_candidate_indices_inplace_tl', 'restore_candidate_indices'))
    },
    **{
        "tilelang_ops." + name: {
            backend: {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.sampling_emu." + impl,
                "limitations": "Mirrors upstream fused_sample_ref/random_sampling_ref (torch sort/cumsum, SplitMix64); property-equivalent to the TileLang kernel, not bitwise; no performance qualification",
            }
            for backend in ("triton", "torch")
        }
        for name, impl in (("fused_sample_tl", "fused_sample"), ("random_sampling_tl", "random_sampling"))
    },
    **{
        "tilelang_ops." + name: {
            backend: {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels." + ("zos_metadata_triton." if backend == "triton" else "zos_metadata.") + impl,
                "limitations": "Fused device-only Triton metadata; 50 exact integer/changed-input Graph cases and V4.1 CP8EP8/DP8EP8 DSpark regression passed (2026-10-28); pinned layouts only" if backend == "triton" else "Device-only torch reference for decode/ZOS metadata",
            }
            for backend in ("triton", "torch")
        }
        for name, impl in (('prepare_v4_decode_metadata_tl', 'prepare_v4_decode_metadata'), ('copy_block_tables_tl', 'copy_block_tables'), ('gather_input_ids_zos_tl', 'gather_input_ids'), ('gather_engram_tokens_zos_tl', 'gather_engram_tokens'), ('compute_seqlens_k_zos_tl', 'compute_seqlens_k'))
    },
    **{
        "tilelang_ops." + name: {
            backend: {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.verification." + impl,
                "limitations": "Device-only torch mirror of the upstream sampler *_ref (scatter instead of mask indexing); no performance qualification",
            }
            for backend in ("triton", "torch")
        }
        for name, impl in (('prepare_verification_metadata_tl', 'prepare_verification_metadata'), ('expand_sampling_params_tl', 'expand_sampling_params'), ('map_lookup_tl', 'map_lookup'), ('remap_draft_topk_tl', 'remap_draft_topk'), ('get_accept_result_tl', 'get_accept_result'), ('finalize_verification_tl', 'finalize_verification'), ('sum_smaller_probs_tl', 'sum_smaller_probs'))
    },
    "tile_kernels.import_surface": {
        "stub": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.tile_kernels.install",
            "limitations": "Pinned production imports; only separately registered numerical providers execute",
        },
    },
    "tile_kernels.quant.batched_transpose_weight_sf": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.weight_scale_emu.batched_transpose_weight_sf",
            "limitations": "GPU UE8M0 int32, block32/block128, grouped layouts and reused outs; no model qualification",
        },
    },
    "tile_kernels.quant.per_token_cast": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.quant_emu.per_token_cast",
            "limitations": "Deterministic BF16/FP32 to E4M3 or packed E2M1, row/column SF and packed UE8M0; no prequantized input, E4M3 scales, stochastic cast or model qualification",
        },
    },
    **{
        "tile_kernels." + suffix + "." + symbol: {
            "torch": {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.engram_init_emu." + symbol,
                "limitations": "Pinned initialization contract; no model/e2e or throughput qualification",
            },
        }
        for suffix, symbol in (("engram", "fused_weight"), ("torch.engram", "make_offsets"))
    },
    **{
        "tile_kernels.engram." + symbol: {
            ("triton" if symbol == "engram_gate_fwd" else "torch"): {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.engram_runtime_emu." + symbol,
                "limitations": (
                    "SGLang 7b91060 fused inference gate with pre-fused FP32 weights; saved backward intermediates retain torch reference; model qualification recorded separately"
                    if symbol == "engram_gate_fwd" else
                    "Hash requires nonnegative INT64-safe unmasked inputs; model qualification recorded separately"
                ),
            },
        }
        for symbol in ("engram_gate_fwd", "engram_hash")
    },
    "deepinfer._C": {
        "hip_staged": {
            "status": "candidate",
            "implementation": "hcu_adapter.bridges._deepinfer.native_extension.install",
            "limitations": "Pinned source build; external GCC13 runtime path; no e2e qualification",
        },
    },
    "deep_gemm.bf16_gemm_nt": {
        "lightop": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.gemm_lightop.bf16_gemm_nt",
            "limitations": "Contiguous BF16 inputs, K divisible by 32; no e2e qualification",
        },
    },
    "deep_gemm.transform_sf_into_required_layout": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.weight_scale_emu.transform_sf_into_required_layout",
            "limitations": "SM100 packed UE8M0 ABI; no PSUM gaps or SM90 float-scale mode; no model qualification",
        },
    },
    **{
        "deep_gemm." + name: {
            "host": {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.runtime_config." + impl,
                "limitations": "Host-side vendor knob mirrored from pinned DeepGEMM; non-SM100 MK alignment 128; num_sms = HIP CU count",
            },
        }
        for name, impl in (
            ("set_num_sms", "deep_gemm_set_num_sms"),
            ("get_num_sms", "deep_gemm_get_num_sms"),
            ("set_mk_alignment_for_contiguous_layout", "set_mk_alignment_for_contiguous_layout"),
            ("get_mk_alignment_for_contiguous_layout", "get_mk_alignment_for_contiguous_layout"),
            ("get_theoretical_mk_alignment_for_contiguous_layout",
             "get_theoretical_mk_alignment_for_contiguous_layout"),
            ("get_tma_aligned_size", "get_tma_aligned_size"),
        )
    },
    **{
        "tile_kernels." + name: {
            "host": {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.runtime_config." + impl,
                "limitations": "Host-side vendor knob mirrored from pinned TileKernels 2.1.0 config.py; num_sms = HIP CU count",
            },
        }
        for name, impl in (
            ("set_num_sms", "tile_kernels_set_num_sms"),
            ("get_num_sms", "tile_kernels_get_num_sms"),
            ("set_token_alignment", "set_token_alignment"),
            ("get_token_alignment", "get_token_alignment"),
        )
    },
    "deep_gemm.fp8_fp4_gemm_nt": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.gemm.fp8_fp4_gemm_nt",
            "limitations": "FP8 E4M3 operands only; exact BF16/FP32 dequant + torch GEMM (FP32 accumulate); no alpha/epilogue; correctness tier, not performance",
        },
    },
    "deep_gemm.fp8_einsum": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.gemm.fp8_einsum",
            "limitations": "'bhr,hdr->bhd' only; exact dequant + torch bmm; FP8-pair output via per-32 UE8M0 cast of the BF16 result; correctness tier",
        },
    },
    "deep_gemm.mega_mhc": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.mhc.mega_mhc",
            "limitations": "Composed torch mHC post+mixes+pre+RMSNorm(+per-32 FP8, gemm/routed SF); no Mega MoE shared-SF layout; correctness tier",
        },
    },
    "deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.gemm.m_grouped_fp8_fp4_gemm_nt_contiguous",
            "limitations": "Route A correctness tier: exact BF16 dequant of FP8 acts x MXFP4/FP8 experts + dense per-group torch GEMM (graph-safe, O(G) work); DTK m_grouped_wfp4a16 rejected (calc_diff~1 on gfx938, see config.md)",
        },
        "triton": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.grouped_gemm_triton.m_grouped_fp8_fp4_gemm_nt_contiguous",
            "limitations": "Route A fast tier: per-BLOCK_M group lookup, FP8/E2M1 x UE8M0 dequantized to BF16 in registers, FP32 accumulate (same products as the torch tier); per-32 packed UE8M0 recipes only, others use the torch tier; tiles from a BW1101 sweep (tests/tools/_probe_grouped_gemm_bench.py)",
        },
    },
    "deep_gemm.fp8_fp4_mqa_logits": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.mla_attention_emu.fp8_fp4_mqa_logits",
            "limitations": "Dense prefill indexer logits (compressed layout, BF16, tail zeroed); FP8/FP4 with per-row MX scales; KV-chunked torch; correctness tier",
        },
    },
    **{
        "deep_gemm." + name: {
            "torch": {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.mla_attention_emu." + name,
                "limitations": "Sparse (candidate-block) indexer logits per vendor test contract; metadata is an HCU-private dict; correctness tier",
            },
        }
        for name in ("fp8_fp4_sparse_mqa_logits", "get_sparse_mqa_logits_metadata", "fp8_fp4_paged_mqa_logits",
                     "get_paged_mqa_logits_metadata", "fp8_fp4_paged_sparse_mqa_logits",
                     "get_paged_sparse_mqa_logits_metadata")
    },
    "deep_gemm.bf16_mega_gate": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.moe_gate_emu.bf16_mega_gate",
            "limitations": "FP32 gate GEMM + graph-safe sqrtsoftplus stable top-k (tile_kernels reference); no force_random; correctness tier",
        },
    },
    "tile_kernels.moe.moe_topk_gate": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.moe_gate_emu.moe_topk_gate",
            "limitations": "Graph-safe rewrite of tile_kernels torch reference; no force_random or training scores output",
        },
    },
    "deep_gemm.tf32_hc_prenorm_gemm": {
        "torch": {
            "status": "candidate",
            "implementation": "hcu_adapter.kernels.mhc.tf32_hc_prenorm_gemm",
            "limitations": "BF16 x FP32 prenorm GEMM, contiguous K-major contract, K tiled in 64-element blocks; no model or performance qualification",
        },
    },
    **{
        "tile_kernels." + suffix + "." + symbol: {
            "torch": {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.mhc." + symbol,
                "limitations": "Pinned mHC forward contract; eager candidate, no model or performance qualification",
            },
        }
        for suffix, symbols in (
            ("mhc.expand_kernel", ("expand_to_mhc_fwd",)),
            ("mhc.pre_apply_mix_kernel", ("mhc_pre_apply_mix_fwd",)),
            ("mhc.head_compute_mix_kernel", ("mhc_head_compute_mix_fwd",)),
            ("mhc.pre_split_mixes_kernel", ("mhc_pre_split_mixes_fwd",)),
            ("mhc.sinkhorn_kernel", ("mhc_sinkhorn_fwd",)),
            ("mhc.post_kernel", ("mhc_post_fwd",)),
            ("mhc.norm_fn_kernel", ("mhc_fn_normw_merge_fwd", "mhc_reduce_partials_and_rmsnorm_fwd")),
            ("mhc.pre_big_fuse_kernel", ("mhc_pre_big_fuse_fwd",)),
        )
        for symbol in symbols
    },
    **{
        "flash_mla.fused_norm_rope_attn_rope_cast." + symbol: {
            "torch": {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels.mla_layout_emu." + symbol,
                "limitations": "Byte-preserving initialization permutation only; no fused attention or model qualification",
            },
        }
        for symbol in ("permute_q_b_proj", "permute_wv_proj")
    },
    **{
        "tilelang_ops." + name: {
            backend: {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels." + impl,
                "limitations": "Device-only MTP/state/constraint contract; affected upstream refs and changed-input graph passed; no full feature qualification",
            }
            for backend in ("triton", "torch")
        }
        for name, impl in (('confidence_head_fused_tl', 'mtp.confidence_head'), ('dflash_input_store_tl', 'mtp.dflash_input_store'), ('copy_dflash_decode_input_tl', 'mtp.copy_decode_input'), ('store_mtp_info_tl', 'mtp.store_proposals'), ('compute_seqlens_q_by_confidence_rates_tl', 'mtp.schedule_by_confidence'), ('get_fused_dflash_get_input_tl', 'mtp.dflash_get_input'), ('select_state_slice_tl', 'state_and_constraints.select_state_slice'), ('load_state_to_cache_tl', 'state_and_constraints.load_state_to_cache'), ('apply_token_bitmask_by_step_inplace_tl', 'state_and_constraints.mask_logits'), ('detect_outlier_and_abort_tl', 'state_and_constraints.detect_outlier'))
    },
    **{
        "tilelang_ops." + name: {
            backend: {
                "status": "candidate",
                "implementation": "hcu_adapter.kernels." + impl,
                "limitations": "Device-only vision/CP metadata contract; upstream refs and changed-input graph passed; full vision/CP model qualification pending",
            }
            for backend in ("triton", "torch")
        }
        for name, impl in (('batch_downsample_tl', 'vision.batch_downsample'), ('prepare_cp_mirror_tl', 'context_parallel.prepare_mirror'), ('replace_input_embeds_tl', 'vision.replace_input_embeds'), ('rgb_pad_to_patches_tl', 'vision.rgb_pad_to_patches'), ('yoco_cp_boundary_fused_tl', 'context_parallel.yoco_boundary'))
    },
    **{
        "tilelang_ops." + name: {
            "stub": {
                "status": "planned",
                "implementation": "hcu_adapter.bridges._deepinfer.tilelang_backend.factories._unsupported",
                "limitations": "Importable factory only; no numerical provider, raises on call",
            },
        }
        for name in PLANNED_TILELANG_FACTORIES
    },
}

# Explicit LightOp candidate, selected only by HCU_LIGHTOP_OPS=1. Compatibility
# quantization formats remain visible in the provider's native/fallback counters.
for _op, _symbol in (
    ('tilelang_ops.norm_tl', 'norm'),
    ('tile_kernels.quant.per_token_cast', 'per_token_cast'),
    ('tile_kernels.quant.swiglu_forward', 'swiglu_forward'),
    ('tile_kernels.quant.swiglu_forward_and_per_token_cast', 'swiglu_forward_and_per_token_cast'),
    ('tile_kernels.moe.moe_topk_gate', 'moe_topk_gate'),
    ('tile_kernels.transform.apply_rotary', 'apply_rotary'),
    ('deep_select.topk', 'topk'),
    ('deep_gemm.bf16_mega_gate', 'bf16_mega_gate'),
    ('deep_gemm.mega_mhc', 'mega_mhc'),
    ('deep_gemm.fp8_fp4_mqa_logits', 'fp8_fp4_mqa_logits'),
):
    BACKEND_CATALOG[_op]['lightop'] = {
        'status': 'candidate',
        'implementation': 'hcu_adapter.kernels.lightop_ops.' + _symbol,
        'limitations': 'Explicit LightOp b2f8c9d; preserve residual/clamp/FP32 intermediate and SF ABI; quant native only rounded block32/128 FP8, other formats counted as compatibility; mega-MoE shared SF unsupported; finite and model qualification recorded separately',
    }

BACKEND_CATALOG['tilelang_ops.fused_norm_rope_hadamard_cast_store_kvcache_tl']['lightop'] = {
    'status': 'candidate',
    'implementation': 'hcu_adapter.kernels.fused_transform.fused_transform_lightop',
    'limitations': 'Native norm/RoPE and block32/128 FP8; original BF16 stage boundaries, FP4 quantization and V4.1 paged scatter retained; finite and model qualification recorded separately',
}

for _symbol in ('prefill', 'decode'):
    BACKEND_CATALOG['flash_mla.fused_norm_rope_attn_rope_cast.' + _symbol]['native'] = {
        'status': 'candidate',
        'implementation': 'hcu_adapter.kernels.native_flashmla.fused_' + _symbol,
        'limitations': 'Explicit code-object-v4 1287a01 plus optional FP32 API, gfx938 H64/H128 D512; LightOp Q/O-RoPE and FP8; decode gathers V4.1 packed cache to BF16, not native paged decode; qualification recorded separately',
    }

for _symbol in ('fp8_fp4_mqa_logits', 'fp8_fp4_sparse_mqa_logits',
                'fp8_fp4_paged_mqa_logits', 'fp8_fp4_paged_sparse_mqa_logits'):
    BACKEND_CATALOG['deep_gemm.' + _symbol]['lightop'] = {
        'status': 'candidate',
        'implementation': 'hcu_adapter.kernels.lightop_mqa.' + _symbol,
        'limitations': 'D128 H8/16/32 MXFP4 uses packed LightOp sparse/paged math; unpaged sparse K is byte-repaged for unaligned request starts, not dequantized per query; paged dense requires page64; BF16 logits default/zero tail retained; explicit output_dtype supports the independent plugin FP32 policy; other layouts use counted compatibility bridge; qualification recorded separately',
    }

for _op in ('tile_kernels.moe.moe_topk_gate', 'deep_gemm.bf16_mega_gate'):
    BACKEND_CATALOG[_op]['lightop']['limitations'] += (
        '; 256/384 expert routing uses LightOp; 128/top3 draft uses SGLang-aligned Triton log1p/NaN-first/lowest-id ties; force_random metadata retains existing HCU normal routing; forced-balance benchmark remains guarded')
