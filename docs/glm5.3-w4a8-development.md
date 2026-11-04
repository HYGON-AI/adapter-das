# GLM-5.3 W4A8 适配记录

本分支配套 HYGON-AI/sglang-das v0.5.22.plugin（15c7204b7a3b），模型为 compressed-tensors。

## 2026-11-04 第一批

- 默认 CT 路由改为 native，保留激活量化；旧 w4a16 仅显式启用，不能用于本任务。未知配置拒绝。
- 修复 all-reduce 接线引用已删除的 _tar_installed，避免启动 NameError。custom AR 已除役，拒绝 AITER 时应直接保留 RCCL 回退。
- 新增 examples/glm5.3-w4a8/launch.sh，保留 CT 格式；当前为 TP8/EP8 eager 探路配方，尚未验收。
- 当前底座已内置 HCUCompressedTensorsW4A8Int8DynamicMoE；组合接线及两个 GEMM 的 A8 执行仍待运行核实。

本地 L0/Python AST、bash -n、git diff --check 已检查；没有新增单测。GPU 验收、原生完整 profile、精度回归均未完成。

## 2026-11-04 原生配方与初始化检查

- 以用户提供的部署文档 revision 82 为配方来源，保留 `SGLANG_OPT_USE_TOPK_V2=0`，移除额外指定的 `--dsa-topk-backend torch`。记录实际 resolve 后的 backend，不通过强制覆盖替代文档配方。
- 多节点共用 NFS Triton 缓存时发生 `_per_token_quant_int8.amdgcn` 文件缺失；编译缓存默认改用容器本地路径，并允许环境变量显式覆盖。
- 在组合容器通过真实 CT 解析与 scheme 选择检查：抽查第 3/77 层和 MTP 专家得到 `HCUCompressedTensorsW4A8Int8DynamicMoE`，普通 Linear 得到 `CompressedTensorsW8A8Int8`，原始配置未被修改。该诊断使用无权重模块，仅证明分发，不证明实际 kernel 或模型精度。
- 指定 SGLang 底座的 11417 个跟踪文件逐项哈希未改变。原生完整 P/D profile 尚未生成，GPU 验收继续进行。
