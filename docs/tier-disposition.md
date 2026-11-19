# tier 处置表（emu 退出运行时 · 零降级）— 活文档

> 依据 governance §3b-8/9（2026-11-19）。本表是 emu 清零与降级点拆除的
> 司法台账：每完成一项把处置改为完成态并留 commit 锚点。
> 分类：**A** = 有兄弟档（fork/triton/vendor/cpp），只需拆运行时降级（W2）；
> **B** = emu 是唯一实现 → 补 triton 或寻源 vendor（W3 核心）；
> **C** = 纯开发/诊断参考 → 移出运行时路由，保留 dev 工具。
> 状态：`待处置` → `进行中` → `完成(档位/commit)`。

| # | emu 实现 | 兄弟档 | 分类 | 处置 | 状态 |
|---|---|---|---|---|---|
| 1 | `hcu_adapter/bridges/sglang_mqa_emu.py` | — | B(补triton/寻源vendor) | 改判：桥本体是 triton 面装载器（HCU_MQA_EMU_IMPL=triton 默认），运行时 fallback 已拆 | **完成**（W2-2） |
| 2 | `hcu_adapter/features/dsa/dsa_mqa_logits_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 3 | `hcu_adapter/features/dsa/kpool_write_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 4 | `hcu_adapter/features/dsa/mqa_logits_emu.py` | `hcu_adapter/features/dsa/mqa_logits_triton.py`, `hcu_adapter/kernels/mqa_logits_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 5 | `hcu_adapter/features/gdn/conv1d_fn_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 6 | `hcu_adapter/features/gdn/conv1d_update_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 7 | `hcu_adapter/features/gdn/kda_prefill_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 8 | `hcu_adapter/features/hadamard/hadamard_emu.py` | `hcu_adapter/features/hadamard/hadamard.py`, `hcu_adapter/features/hadamard/hadamard_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 9 | `hcu_adapter/kernels/compress_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 10 | `hcu_adapter/kernels/deepselect_emu.py` | `hcu_adapter/bridges/_deepinfer/deepselect.py`, `hcu_adapter/kernels/deepselect_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 11 | `hcu_adapter/kernels/engram_init_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 12 | `hcu_adapter/kernels/engram_runtime_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 13 | `hcu_adapter/kernels/gather_emu.py` | `hcu_adapter/kernels/gather.py`, `hcu_adapter/kernels/gather_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 14 | `hcu_adapter/kernels/gemm_emu.py` | `hcu_adapter/kernels/gemm.py`, `hcu_adapter/kernels/gemm_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 15 | `hcu_adapter/kernels/grouped_gemm_emu.py` | `hcu_adapter/kernels/grouped_gemm_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 16 | `hcu_adapter/kernels/index_ops_emu.py` | `hcu_adapter/kernels/index_ops.py`, `hcu_adapter/kernels/index_ops_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 17 | `hcu_adapter/kernels/kv_cache_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 18 | `hcu_adapter/kernels/mhc_emu.py` | `hcu_adapter/bridges/vllm_dsv41/mhc.py`, `hcu_adapter/kernels/mhc_aiter.py`, `hcu_adapter/kernels/mhc_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 19 | `hcu_adapter/kernels/mla_attention_emu.py` | `hcu_adapter/kernels/mla_attention_dtk.py` | A(拆降级) | 待处置 | 待处置 |
| 20 | `hcu_adapter/kernels/mla_layout_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 21 | `hcu_adapter/kernels/moe_gate_emu.py` | `hcu_adapter/kernels/moe_gate_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 22 | `hcu_adapter/kernels/mtp_emu.py` | `hcu_adapter/kernels/mtp_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 23 | `hcu_adapter/kernels/norm_emu.py` | `hcu_adapter/kernels/norm.py`, `hcu_adapter/kernels/norm_lightop.py`, `hcu_adapter/kernels/norm_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 24 | `hcu_adapter/kernels/quant_emu.py` | `hcu_adapter/features/aiter/quant_aiter.py`, `hcu_adapter/kernels/quant_lightop.py` | A(拆降级) | 待处置 | 待处置 |
| 25 | `hcu_adapter/kernels/request_indices_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 26 | `hcu_adapter/kernels/rope_emu.py` | `hcu_adapter/kernels/rope_lightop.py`, `hcu_adapter/kernels/rope_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 27 | `hcu_adapter/kernels/sampling_emu.py` | `hcu_adapter/bridges/_sglang/sampling.py` | A(拆降级) | 待处置 | 待处置 |
| 28 | `hcu_adapter/kernels/sglang_mhc_prenorm_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 29 | `hcu_adapter/kernels/state_and_constraints_emu.py` | `hcu_adapter/kernels/state_and_constraints_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 30 | `hcu_adapter/kernels/verification_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 31 | `hcu_adapter/kernels/vision_attention_emu.py` | `hcu_adapter/bridges/_deepinfer/vision_attention.py` | A(拆降级) | 待处置 | 待处置 |
| 32 | `hcu_adapter/kernels/vision_emu.py` | `hcu_adapter/kernels/vision_triton.py` | A(拆降级) | 待处置 | 待处置 |
| 33 | `hcu_adapter/kernels/weight_scale_emu.py` | — | B(补triton/寻源vendor) | 待处置 | 待处置 |
| 34 | `hcu_adapter/kernels/zos_metadata_emu.py` | `hcu_adapter/kernels/zos_metadata_triton.py` | A(拆降级) | 待处置 | 待处置 |

## 运行时降级点拆除台账（W2，governance §3b-8 违规清单）

| 降级点 | 位置 | 处置 | 状态 |
|---|---|---|---|
| shadow 三层梯子 try-descent | `sgl_kernel/elementwise.py` 等 | 解析期定档，缺实现=启动报错 | **部分完成**（W2-1：elementwise silu 族严格化+torch 退役；layernorm/rotary/gemm_fp8 等剩余梯子待后续批逐文件严格化） |
| aiter one-strike（失败置 None 落 triton） | `sgl_kernel/elementwise.py` | 移除，vendor 错误直曝 | 待处置 |
| aiter_moe 无调优配置→stock triton (log-once) | fused_moe 桥 | 硬报错或显式配置 | **完成**（显式 aiter 拒绝/无配置=RuntimeError；auto=确定性选择保留+log-once；包裹层 except→orig 全量吞错拆除；W2-3） |
| mqa triton 臂 torch-emu fallback | `bridges/sglang_mqa_emu.py` | 拆（emu 退运行时） | **完成**（per-call emu 回落拆除，严格分发；W2-2） |
| HCU_FA3_EMU=1 默认 | env 注册表 | 默认翻转+报错 | **完成**（default=0，docs 同步；W2-2） |
| _auto_mqa_emu auto 落 emu | `infra/hcu_env.py` | auto 不落 emu | **完成**（return None；W2-2） |
| R-12 桥隔离静默走上游 | bridges try-isolation | 配置档失败=首用报错 | **部分完成**（W2-5：失败升 ERROR + 启动降级汇总行，gate 可断言；per-op 首用报错待 enforcement 引擎批） |
| 安装链碎行降噪 | bridges/features 12 处成功行 | warning→info（P7 默认静默） | **完成**（W2-5） |
| KDA/conv1d emu 兜底 | gdn 族路由 | 无可用档=报错（gfx938 接受断供） | **完成**（=emu 路由即 RuntimeError，运行时值 fork\|stock；W2-5） |
| P7 启动 tier 表 | `bridges/sglang.py` install 尾部 + `registry.log_summary` | 完成（桥装完后集中打表，一行一算子） | **完成**（W2-4） |
| wizard/validator 拒退役档 | `hcu_config._validate` + wizard | status=retired 选择即报错（3b-9） | **完成**（W2-4） |
