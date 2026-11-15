# 影子库矩阵（shadow-library matrix）

## 边界定义

以**社区代码的 import 面**为边界：sglang / vllm 源码里 `import` 到的算子
库名，本仓库都要让 `import` 成功。不锚定任何库的内部实现（内部实现随上游
版本漂移，import 面稳定得多）。

DeepInfer 契约实现按算子族和档位放入 `hcu_adapter/kernels/`，文件位置及入口职责见[算子文件导航](../../../docs/frameworks/deepinfer-kernels.md)。
拆分实现时同步 bridge/catalog 与 plugin 的导入和能力检查；删除迁移后的旧算子转发路径。

当前覆盖的 import 面（发现新的就加进矩阵）：

| 社区 import 的包 | 我们的形态 | 位置 | 状态 |
|---|---|---|---|
| `sgl_kernel` | 完整同名包（triton/torch） | `sgl_kernel/` | 生产 |
| `vllm._custom_ops` | SGLang 启动链使用的 RMS 属性级兼容绑定；不是 vLLM 宿主的默认重绑定 | `hcu_adapter/bridges/vllm.py::install_rms_norm_bridge` | SGLang 兼容路径 |
| `aiter` | 真包优先 + 缺口填空 + 高危守卫 | `hcu_adapter/aiter/` | 生产 |
| `flashinfer` | 填空桥 + 响亮 stub | `hcu_adapter/flashinfer/` | 生产 |
| `flash_attn` / `flash_attn_interface` | 响亮 stub；装 HCU flash_attn wheel 后真包自动胜出（零改动切 vendor 档） | `hcu_adapter/flash_attn/` | 生产 |
| `lightop` | vendor 直调档（不出现在 import 面，按 ops 路由调用；见下盘点） | 系统包直调 | silu/MQA 已入 catalog |

**RMS 宿主 ABI 分开核对**：上述 helper 的生产调用点在
`hcu_adapter/bridges/sglang.py`；vLLM runtime plugin 不调用它。helper 的
`fused_add_rms_norm(out, input, residual_out, residual, weight, epsilon)` 是
六参数、独立输出目标的兼容签名；vLLM 0.32 官方
`fused_add_rms_norm(input, residual, weight, epsilon)` 是四参数原位接口，
修改 input/residual 并返回 `None`。不能因同名把前者装进 vLLM 启动链。
该属性绑定也不是 `vllm._custom_ops` / `_C` 的完整影子实现；缺失的原生符号
须逐项验证。vendor 的响亮 stub 可让 import / `hasattr` / `callable` 成功，
这些检查不能证明所需 provider 的 ABI 或计算可用。

**flash_attn wheel 兼容性现状**（2026-10-10 实测）：共享盘
`/public/share/dtk_fa_whl/*/flash_attn-2.6.1+das.opt1.dtk2604-cp310`
只匹配 torch2.5 栈；dtk26041/torch2.11 上 `c10::hip` 符号不匹配，需
vendor 出新轮。轮子就位后 `tests/vendor/test_flash_attn.py` 的探针会自动
切到真包跑全场景矩阵（sglang-das 侵入仓同款回退链
`kernels → sgl_kernel.flash_attn → flash_attn 包`）。

## 三档实现策略

每个算子出口按优先级路由，出口签名逐字对齐社区版（含 `out=`、
destination-passing、占位参数）：

1. **triton（生产主力）**：自研移植，fp32 累加、graph-safe、无主机同步。
2. **vendor（转发档）**：DTK wheel 里真包已有且经过我们的压测+守卫的
   （如 aiter 部分 kernel、lightop）。接入条件 = 过场景压测 + 注册快关。
   arch 探测失败自动降级到 triton/torch 并打日志。
3. **torch（参考档）**：语义金标 + 功能兜底（慢但永远正确），也是
   契约测试的参考实现来源。

**档位状态机与演进（catalog 管制）**：每算子
出口的可用实现在 BACKEND_CATALOG 挂状态——
`planned → candidate（离线三段式绿+对拍+bench）→ validated（serving A/B
过）→ production（默认档，连续稳定后）`，定罪不可用标 `broken`（选配即
拒，bug 签名与复现探针在册；哨兵意外转绿=轮子修好，重跑门禁即晋级）。
演进**单向**：**emu（torch 调试图稳/金标准）→ triton → vendor**；
emu/triton 换下不删、转 ref 角色只留离线测试对照。校准方向不变：emu 校准
triton，emu/triton 暴露 vendor bug。晋级证据门与登记要求见
`docs/governance.md`。

路由已落地（env + yaml 双层）：`HCU_SILU_MUL_IMPL` / `HCU_RMS_NORM_IMPL` /
`HCU_FUSED_MOE_IMPL` 等已在册（hcu_env REGISTRY），正式机制是
`hcu_adapter.yaml` 的 ops 键（见 docs/config.md 与 BACKEND_CATALOG），
`auto` = 探测+压测结论决定；新算子接入时同步登记 catalog。

## lightop 接入盘点（DTK 0.6.0，nmz26 dtk26041 实测 export 面与 aiter 交集）

gfx938 有 fp8（gfx936 只有 int8）。**与 aiter 同用法的交集算子**（同一语义
两边都有实现 → 同一份契约测试跑双 backend，正是 73-297 要的"默认打开、
排精度时可切"）：

| 语义 | lightop 出口 | aiter 出口 | 接入优先级 |
|---|---|---|---|
| silu_and_mul(+fp8 quant 融合) | `activation.existing.fuse_silu_and_mul` / `fuse_silu_mul_fp8_quant` 系列（含 _ep） | `silu_and_mul` / fused_clamp_act_mul 融合 | P1（契约最简单，跑通双 backend 模式） |
| MQA logits（DSA） | `attention.mqa.paged_mqa_logits` / `mqa_logits` / `_fp4` 变体 | `aiter.ops.triton(.attention).pa_mqa_logits` | P1（我们有 mqa_logits_emu torch 金标可直接对拍） |
| MoE group gemm | `moe.groupgemm_permute / fuse_groupgemm_permute_quant / gemm_ops.quantized.gemm_w8a8_*`（marlin w4a8/w4a16 配置齐） | aiter 项目前 moe 出口少 | P2 |
| RMSnorm+RoPE 融合 | `fuse_rmsnorm_rope.rms_rotary_embedding_fuse(_with_kv_store)` | 无对应（aiter 分开） | P2（性能档） |
| KV cache 反量化 | `kvcache.dequantize_k_cache_paged` / `gather_and_maybe_dequant_cache` | `cp_gather_indexer_k_quant_cache`（我们已桥） | P3 |

接入顺序：P1 两个先做（silu 融合→mqa），每个出口注册 `HCU_<OP>_IMPL`
env 档位（auto=vendor 过压测则用/triton/torch），四个维度测试各跑一遍。
 أصل轻盘点记录：`lightop._lmslim_native.quantize` 顶层无导出（空）、
marlin repack 只在 lightop 有（aiter 无交）。

## 新增一个算子垫片的流程

1. **定契约**：找到社区调用点，抄下完整签名/布局/返回语义（读上游源码，
   不要猜；rope 教训：同一个函数名在 cache 布局上还有隐含约定）。
2. **写参考**：`torch` 档参考实现（fp32），这就是测试金标。
3. **写生产档**：triton 或 vendor 转发；遵守黄金法则（fp32 累加、无
   主机同步、import 不失败）。
4. **写契约测试**：按 `test-suite.md` 四维度；vendor 高危面同时写场景
   压测 + 注册快关（`kill-switches.md`）。
5. **更新矩阵**：本文件表格 + README 覆盖范围表。

### SGLang dense cross-attention 与 varlen 的边界

`flash_attention_v3.flash_attn_varlen_func` 的官方 FA2 回退还支持四维
`[batch, sequence, heads, dim]` Q/K/V 与两项 `cu_seqlens=None`。
`fa_vendor` 的 Triton/vendor ops 入口须先分流这种 dense 调用，保留 batch
并按官方参数和返回契约委托 `flash_attn_func`；不能先展平再送进 varlen。
宿主桥强制放行 FA3，直接调用保存的原 v3 函数仍会进入 shadow varlen，
因此这里复用的是官方 FA2 回退调用面。LSE 保留 FA2 原返回 tuple；`out`
等额外参数沿用官方 FA2 回退的处理，不另行扩展语义。

varlen Triton 准入至少要求两项序列边界存在；缺失/单边缺失边界的三维
调用须在 ops 入口明确拒绝，不能仅给诊断日志的 `.numel()` 加判空。
离线路由回归见 `tests/adapter/test_fa_dense_dispatch.py`；真实公共 dispatcher
数值及变输入 Graph 回归见 `tests/kernels/test_flash_attn_dense_dispatch.py`，
需分别用新进程验证 `HCU_FA_IMPL=auto/emu/triton/vendor`。
`release/sglang_v0.5.22` 的 auto 默认走 emu，必须在该入口同时保留
dense 回退并刷新公共 dispatcher 缓存别名；ragged 继续调用原 emu。组件结果不替代 Wan
模型生成与质量验证，也不代表所有 vendor attention 场景均已验证。

### FlashInfer cached RoPE 的社区 Triton 接入

`flashinfer.rope.apply_rope_with_cos_sin_cache_inplace` 由
`sgl_kernel/rope_triton.py` 提供明确的 Tensor/int/bool 签名，
使社区 `register_custom_op_from_extern` 可以推导 schema；不能保留
Missing 占位对象或仅为它添加虚假签名。

实现从社区 v0.5.22 `rotary_triton.py` 的旋转对计算和候选配置迁出，
不依赖或改写社区源码。kernel 直接按 positions 读取 FP32 半分布局 cache，
支持 GPT-J/NeoX，并在读取完整旋转对后原位写回；部分旋转保留尾部。
不分配 gather、布局转换或旋转结果张量。autotune 使用 `restore_value`
恢复每次试跑的输入；缓存命中的正常调用不备份、不拷贝输入。
首次 capture 前须预热相应 shape。候选登记为 flashinfer_rope/triton。

接口与来源：FlashInfer 官方 apply_rope_with_cos_sin_cache_inplace 文档；
目标社区 v0.5.22。契约测试 `tests/kernels/test_flashinfer_rope.py` 覆盖
真实影子导入、schema、FP64 参考、GQA、随机 positions、两种布局、
部分旋转、行间 padding、strided positions/cache、空输入、首次调优仅旋转
一次、缓存命中无临时 GPU 张量、变输入 Graph 和真实 diffusion helper。
算子通过不替代模型推理验收；当前进度与结果以模型适配日志为准。


### Diffusion norm-scale-shift 的 HCU CUDA dispatch

社区 `sglang::fused_norm_scale_shift` 的 CUDA kernel 由
`hcu_adapter/diffusion_norm.py` 通过公开 `torch.library.Library(..., "IMPL", "CUDA")`
接入，允许早于 schema 注册，不提前导入 diffusion runtime。保留已有 custom-op
对象、schema、fake 和调用者别名，`LayerNormScaleShift.forward_cuda` 保持原样。
数值实现位于 `sgl_kernel/norm_scale_shift_triton.py`，不依赖 SGLang 内部模块。

Triton 从社区 v0.5.22 LayerNorm modulation 移植，gfx936、BF16、无 affine、
D4096/5120、FP32 batch 调制采用 256 逻辑线程、wave64、2 rows/CTA。
保留 normalized 的 BF16 中间舍入，固定配置来自独立 Triton/HIP 对照实验。
其他 dtype、RMSNorm、affine、scalar/per-token/per-frame 广播使用 Torch 回退；
没有 Triton 时也可导入。该回退保留社区 layer 模式仅在 weight/bias 都存在时
应用 affine、rms 模式忽略 bias 的现有语义。残差版本使用下面的独立实现。

数值未证明全形状逐元素一致，catalog 保持 candidate；定向误差/Graph 检查不等于
模型质量或 exact 契约验收。新增测试覆盖广播、只读输入、已缓存别名、原生模型层、
导入期无设备初始化、fake、变输入 Graph 和 Torch 回退。真实模型进度见项目记录。

`sglang::fused_scale_residual_norm_scale_shift` 由同一 install 注册，数学位于
`sgl_kernel/residual_norm_scale_shift_triton.py`，基于社区逐行 Triton 路线
扩展广播与 RMSNorm，保留 CUDA 融合流程的中间 dtype 存储点。返回两个新
张量，不做输入原地更新；schema/fake/缓存别名和 forward_cuda 保留。
当前 gfx936 实测，catalog 为 candidate，无法在本机执行原 CuTeDSL 作 NVIDIA
逐位参考。范围与验证见 [diffusion norm 接入](../../../docs/sglang-diffusion-norm.md)，
后续算子接入按 [CUDA 语义下的算子兼容](cuda-semantics-operators.md) 分类处理。
