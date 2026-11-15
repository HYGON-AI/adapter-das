# SGLang diffusion norm 的 CUDA dispatch 适配

## 范围与接入

适配社区已有的 `sglang::fused_norm_scale_shift` 与
`sglang::fused_scale_residual_norm_scale_shift`。SGLang bridge 调用
`hcu_adapter.diffusion_norm.install`，仅 HIP 环境注册 CUDA dispatch 实现，
保留 `torch.library.Library` 生命周期。模型 forward_cuda、custom op 对象、
schema、fake handler 及缓存别名均保留；不修改社区源码。

数学分别在 `sgl_kernel/norm_scale_shift_triton.py` 与
`sgl_kernel/residual_norm_scale_shift_triton.py`。后者基于社区
`scale_residual_norm_scale_shift_triton.py` 的逐行残差/归一化算法，扩展
LayerNorm/RMSNorm、独立广播参数、batch/token/frame 布局和缺省参数。
HIP contiguous 输入走 Triton；CPU、无 Triton、非连续输入走 Torch。
hidden 维度沿接口要求为 256 的倍数且不超过 8192；实际 GPU 验证为 gfx936。

## 残差算子的契约与精度边界

返回 `(modulated, residual_out)`，两者均为新张量，不修改 residual/x/gate。
残差乘加按 FP32 计算，按 x.dtype 保存并输入 norm；norm 使用 FP32 归约，
归一化/仿射后按 x.dtype 舍入，再执行 modulation。参数为 None、scalar、
vector、batch、token、frame 时按各自布局读取，scale/shift 可不同布局。
RMSNorm 不使用 bias；LayerNorm 保留社区仅在 weight/bias 均提供时启用仿射的行为。

参考为对照社区 CuTeDSL 中间存储点构造的独立 Torch 分阶段实现；无法在
本机运行 NVIDIA CuTeDSL，所以这不是与 NVIDIA CUDA 输出逐位对拍。
与社区 XPU Triton 的差异包括 residual 输入 norm 前的 dtype 舍入及 norm
后舍入。状态为 candidate，不声明 bitexact、全模型精度或性能最优。

## 验证与使用 skill

`tests/kernels/test_residual_norm_scale_shift.py` 覆盖数值、输出顺序、输入
不变性、广播、边界、变更输入的 Graph/eager 一致性。
`tests/adapter/test_diffusion_norm.py` 保留真实 custom op、缓存别名、fake、
未修改的 layer forward，并拦截 CuTeDSL compile 确认替代生效。
本轮算子 44 项通过，含旧 norm 接入回归的另一组 61 项通过；测试容差和
形状见测试，不扩大为原 exact gate 的豁免。

后续同类适配使用 [hcu-adapter-dev](../skills/hcu-adapter-dev/SKILL.md)，
按 [CUDA 语义下的算子兼容](../skills/hcu-adapter-dev/references/cuda-semantics-operators.md)
分类、选择实现及 seam。输入需包含真实调用链、包/源码版本、硬件及错误；
输出为契约实现、注册及消费者验证、模型实测边界。具体模型日志保存在项目记录。

2026-11-14 单卡 BW1000、社区 v0.5.22 + adapter/plugin、CPU offload、
720p/17 帧/1 步实测 job `03aa788a-d40e-4cfc-8d9a-7cff06de0bfe` completed，
通过去噪/解码并保存 MP4，无 native fallback。只证明本配置执行链通过，
不代表正式视频质量、多卡回归或性能验收。该条为当时单卡执行记录。


## 2026-11-15：组合验证与提交范围

本轮在社区 v0.5.22 `baa9befa381511cd4d56f83980951e5a45fffc6d` 上，
组合 adapter 与 plugin 的 release/sglang_v0.5.22 功能分支运行。
BW1000/gfx936 四卡 FSDP + Ulysses4，DiT 常驻、text encoder/VAE offload，
SLA topk=0.25，1280×720、81 帧、16fps、40 步、CFG4/6。
前一批10条与官方提示词队列前37条完整视频均生成成功；37条的七项 VBench
custom_input 诊断评分无漏样本。用户复核视频认为语义基本正确。
部分首帧仍有彩噪/块状伪影，未定位根因；不将语义确认或视频均分描述为
全模型逐位精度通过，也不将这批子集评分描述为官方全量成绩。

同源码四卡 torch.compile 已完成2步、4步预热与40步完整请求。
40步 API 推理927.98秒，原 eager 样本943.72秒；两组物理GPU不同且n=1，
不据此确定性能收益或改变默认编译配置。
移植来源及修改范围见 [第三方清单](../THIRD_PARTY_NOTICES.md)。
推送前仅整理来源声明与验证记录，运行时算法与已测快照保持一致。


推送前复验：本次四个 RoPE/norm/residual/custom-op 测试文件在真实 HCU 上
130 passed；`tests/test_ops_cpu.py` 38 passed；L0 GREEN，Skill 元信息通过。
正式 adapter bootstrap 环境下的全量 kernels/adapter/vendor 回归仍出现失败，
不能将定向通过表述为全仓门禁 GREEN。HOSTnone 的扩展诊断另有接口收集问题，
不作为正式 bootstrap 验收；ops triage 还受缺少其默认 GLM 检查点及其他
旧 ABI/AR 用例影响，未在本次 Wan 改动中处理。原始命令与输出留在项目记录。
