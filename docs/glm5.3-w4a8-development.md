# GLM-5.3 W4A8 适配记录

本分支配套 HYGON-AI/sglang-das v0.5.22.plugin（15c7204b7a3b），模型为 compressed-tensors。

## 2026-11-04 第一批

- 默认 CT 路由改为 native，保留激活量化；旧 w4a16 仅显式启用，不能用于本任务。未知配置拒绝。
- 修复 all-reduce 接线引用已删除的 _tar_installed，避免启动 NameError。custom AR 已除役，拒绝 AITER 时应直接保留 RCCL 回退。
- 新增 examples/glm5.3-w4a8/launch.sh，保留 CT 格式；当前为 TP8/EP8 eager 探路配方，尚未验收。
- 当前底座已内置 HCUCompressedTensorsW4A8Int8DynamicMoE；组合接线及两个 GEMM 的 A8 执行仍待运行核实。

本地 L0/Python AST、bash -n、git diff --check 已检查；没有新增单测。GPU 验收、原生完整 profile、精度回归均未完成。
