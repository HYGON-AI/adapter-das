# 第三方与外部来源清单（THIRD_PARTY_NOTICES）

> 本文件登记仓库内所有非本仓原创的源码来源（HYGON 开源合规规则
> `hygon-open-source-v1.7` §4 要求）。新增移植/借源代码时同步登记本表；
> 纯 HYGON 原创文件只需文件头 HYGON Copyright + SPDX，不进本表。

## 1. 外部开源项目

### flash-linear-attention（MIT）

- 项目：flash-linear-attention（fla-org），Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
- 仓库：<https://github.com/fla-org/flash-linear-attention>（MIT）
- 引入版本：经 sglang-das fork 固定 commit `364fabf`（见 §2）整体引入；
  文件内 `Adapted from` 链接为上游 main 快照，以其所在 fork commit 为固定锚点
- 本地路径：`hcu_adapter/fla_hcu/{chunk_delta_h,chunk_intra,chunk_intra_token_parallel,cumsum,index,kda,l2norm,op,utils}.py`
- HYGON 修改：包内 import 重写到本适配层包；携带 fork 的 gfx938 修正
  （清单见 `hcu_adapter/fla_hcu/__init__.py`）。原文件内版权与许可声明原样保留。

### vLLM（Apache-2.0）

- 项目：vLLM，Copyright contributors to the vLLM project
- 仓库：<https://github.com/vllm-project/vllm>（Apache-2.0）
- 引入版本：`kda.py` 锚定 vLLM commit `0384aa7150c4c9778efca041ffd1beb3ad2bd694`；
  `conv1d_triton.py` 经 sglang-das fork（`364fabf`）引入
- 本地路径：`hcu_adapter/fla_hcu/{kda,conv1d_triton}.py`
- HYGON 修改：随 fla_hcu 包的 import 重写与 gfx938 修正；原 SPDX/版权声明原样保留。

### causal-conv1d（BSD-3-Clause）

- 项目：causal-conv1d，Copyright (c) 2024, Tri Dao
- 仓库：<https://github.com/Dao-AILab/causal-conv1d>（BSD-3-Clause）
- 引入版本：经 vLLM `causal_conv1d.py` / sglang-das fork（`364fabf`）引入
- 本地路径：`hcu_adapter/fla_hcu/conv1d_triton.py`
- HYGON 修改：同上（import 重写、fork 的 HCU 修复）。

### SGLang diffusion Triton 算子（Apache-2.0）

- 仓库：<https://github.com/sgl-project/sglang>（Apache-2.0）
- 固定来源：commit `baa9befa381511cd4d56f83980951e5a45fffc6d`。
- 来源与本地路径：
  - `python/sglang/kernels/ops/diffusion/rope/rotary_triton.py` →
    `sgl_kernel/rope_triton.py`：保留 cached RoPE 计算，改为原地更新 Q/K，
    添加 restore-safe autotune、输入契约及无额外写回的接口。
  - `python/sglang/kernels/kda_kernels/layernorm_modulate_triton.py` →
    `sgl_kernel/norm_scale_shift_triton.py`：移植 Welford 归约与调制，
    将 NVIDIA 指令替换为 HIP 可执行原语，保留 dtype 舍入点并增加广播回退。
  - `python/sglang/kernels/ops/diffusion/norm/scale_residual_norm_scale_shift_triton.py` →
    `sgl_kernel/residual_norm_scale_shift_triton.py`：扩展 LayerNorm/RMSNorm、
    独立广播布局及残差/归一化中间 dtype 舍入。
- 原许可声明保留；这些移植不声明 NVIDIA CUDA 输出逐位一致，当前状态为 candidate。

## 2. HYGON 内部仓库移植（同集团血缘，非外部第三方）

> 这些来源属 HYGON 团队自有仓库，无外部许可证义务；为满足来源可追溯
> 要求登记固定 commit。

### sglang-das fork（Apache-2.0，sglang 衍生）

- 仓库一：<https://github.com/maxiaobetter/sglang-das>，分支
  `main_glm53_flash_feature_cleanup_tx`，commit `364fabf`
- 仓库二：<https://github.com/HYGON-AI/sglang-das>，commit
  `7b91060ae3ef5e112048988da6f4f673cf4b6a89`
- 本地路径与修改：
  - `hcu_adapter/fla_hcu/`（除 `__init__.py`/`_shim.py` 外全部文件）——
    内核逐字移植，仅包内 import 重写；`_shim.py` 垫片与 `__init__.py`
    为本仓原创（已加 HYGON 头）。
  - `hcu_adapter/fla_hcu/fused_sigmoid_gating_recurrent.py`——KDA decode
    recurrent kernel，同上逐字移植。
  - `hcu_adapter/kernels/engram_gate_triton.py`——算术/归约顺序/launch
    geometry 与源一致，接口扩展（见该文件 docstring 与文件头 Modified by）。
  - `hcu_adapter/aiter_moe.py`——按 fork `fused_experts_impl_aiter` 配方
    重实现的桥接层（非逐字拷贝）。

### sglang-hygon-plugin（HYGON 内部 Qwen-Image 2.1 插件）

- 固定来源：`6985a7c5852c90554ca5bed158f8a35f82eadef4`。
- `sglang_hygon/kernels/{complex_rope,qknorm_complex_rope,qknorm_complex_rope_pack}.cuh`
  原样移至 `sgl_kernel/diffusion_rope_csrc/`。
- `sglang_hygon/ops.py` 中的 HIP JIT 包装迁至
  `sgl_kernel/diffusion_rope_cpp.py`，加入上游接口的布局选择和 Torch 回退。

### lmslim（HYGON 内部量化算子库，DTK wheel 分发，非公开开源项目）

- 来源：内部 wheel `lmslim 0.4.0+das.opt1.dtk2604.torch290` 系列
- 本地路径：`hcu_adapter/slimquant.py` 中 `_w4a8_moe_kernel`——自
  `fused_moe_kernel_int4_w4a8` 移植；该文件其余部分（checkpoint 解析、
  safetensors shim）为 HYGON 原创。

## 3. 登记规则速记

- 外部开源（MIT/BSD/Apache 等）：保留原声明 + 本表登记，不加 HYGON 头。
- HYGON 内部跨仓移植：本表登记固定 commit；有实质修改的文件加
  HYGON Copyright + `Modified by`（H2 样式，如 `engram_gate_triton.py`）。
- 纯原创：文件头两行（Copyright + SPDX-Apache-2.0），脚本
  `tools/compliance/add_hygon_headers.py` 可批量补齐并幂等跳过已有头。

### SGLang Mamba state scatter（Apache-2.0）

- 来源：sgl-project/sglang，commit `baa9befa381511cd4d56f83980951e5a45fffc6d`。
- 原路径：`python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py`。
- 本地路径：`hcu_adapter/kernels/mamba_state_scatter_triton.py`。
- 修改：仅将 request / element-block 的 CTA 映射展平到 grid x；保留 source-row、
  step/destination/source bounds、64-bit offsets、masked read/write 与原函数参数 ABI。
