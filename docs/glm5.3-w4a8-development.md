# GLM-5.3 W4A8 适配记录

本分支配套 HYGON-AI/sglang-das v0.5.22.plugin（15c7204b7a3b），模型为 compressed-tensors。

## 2026-11-04 第一批

- 默认 CT 路由改为 native，保留激活量化；旧 w4a16 仅显式启用，不能用于本任务。未知配置拒绝。
- 修复 all-reduce 接线引用已删除的 _tar_installed，避免启动 NameError。custom AR 已除役，拒绝 AITER 时应直接保留 RCCL 回退。
- 新增 examples/glm5.3-w4a8/launch.sh，保留 CT 格式；当前为 TP8/EP8 eager 探路配方，尚未验收。
- 当前底座已内置 HCUCompressedTensorsW4A8Int8DynamicMoE；组合接线及两个 GEMM 的 A8 执行仍待运行核实。

第一批的本地 L0/Python AST、bash -n、git diff --check 已检查；没有新增单测。当时 GPU 验收、原生完整 profile、精度回归均未完成；后续进展见下文。

## 2026-11-04 原生配方与初始化检查

- 以用户提供的部署文档 revision 82 为配方来源，保留 `SGLANG_OPT_USE_TOPK_V2=0`，移除额外指定的 `--dsa-topk-backend torch`。记录实际 resolve 后的 backend，不通过强制覆盖替代文档配方。
- 多节点共用 NFS Triton 缓存时发生 `_per_token_quant_int8.amdgcn` 文件缺失；编译缓存默认改用容器本地路径，并允许环境变量显式覆盖。
- 在组合容器通过真实 CT 解析与 scheme 选择检查：抽查第 3/77 层和 MTP 专家得到 `HCUCompressedTensorsW4A8Int8DynamicMoE`，普通 Linear 得到 `CompressedTensorsW8A8Int8`，原始配置未被修改。该诊断使用无权重模块，仅证明分发，不证明实际 kernel 或模型精度。
- 指定 SGLang 底座的 11417 个跟踪文件逐项哈希未改变。本轮初始化检查时尚未生成原生完整 P/D profile，后续抓取结果见下节。

## 2026-11-04 原生 P/D profile 完成

原生镜像的 P=PP2/TP8/CP8/EP8、D=两节点 TP16/DP16/EP16、EAGLE 5/1/6，完成 4095 输入 token、64 输出 token 的完整请求抓取。32 份 trace 逐份 gzip/JSON 解析通过，取回本地后的压缩文件 SHA256 逐项一致，清单见 [原生 profile 记录](glm5.3-w4a8-native-profile.json)。

已命中 prefill HIPC contiguous W4A8、decode ASM masked W4A8、FlashMLA sparse attention、LightOp INT8 Paged MQA；这不代表组合环境的同输入数值已经对齐。ROCtracer 报告 duplicate flow 警告，kernel duration 总和不用于端到端性能验收。本次只覆盖一条约 4K 请求，其他长度、缓存及并发分支仍需补齐。

原生定向短请求在温度 0 下有复读/截断；文档的温度 1.0、top_p 0.95 下三条短请求正常结束且答案正确。这些是诊断请求，不是总体精度成绩。

初始 packed INT4 Triton 探路配方显式关闭 `SGLANG_ROCM_USE_AITER_MOE`，防止底座的通用 AITER 分支按 W4A16 标志解释 CT 特有的 per-channel W4A8 组合。实际 kernel 验证仍待完成。

## 2026-11-04 CUDA 特性探针导出与 index-K 配方

`platform_seam` 已将 NVIDIA CUDA 版本探针在 `utils.common` 中设为 False，但 `sglang.srt.utils` 的提前导出仍指向旧函数。KDA 模块被 plugin 注册链导入时，旧函数在 HIP Torch 的 `torch.version.cuda=None` 上执行 `split`，中断整个插件注册，导致后续 KV pool 钩子缺失。修复将这组探针与现有平台函数一起同步到包级导出；没有伪造 Torch CUDA 版本，也没有改变 KDA 算法。

实际容器检查：通过 `common` 和包级导入的 Ampere/Hopper/Blackwell/SM90/SM100 探针为同一函数且均返回 False。配合 plugin 的原生能力复用修复，完整注册检查通过，后续 KV pool 钩子已登记；证据为任务产物 `platform-probe-v2.log`、`plugin-bootstrap-v3.log`。启动配方补齐原生文档的 `SGLANG_DSA_HCU_INT8_INDEX_K_CACHE=1`。组合 v5 已完成八卡 CT 权重加载，约 47 GB/卡，但 warmup 尚失败于上述缓存接线，修复后的完整生成仍待验证。
