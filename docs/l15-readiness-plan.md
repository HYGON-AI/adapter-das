# HCU SGLang 启动就绪门禁执行计划

目标：社区 SGLang 源码不变，通过 adapter 的 Triton 功能基线与模型执行链回归，
为明确的模型及配置提供启动依据。不以性能作为本轮验收条件。

1. **严格执行与结果判定**：校验真实单卡 HIP、源码来源、版本绑定；
   每个必测 nodeid 必须实际通过，skip/xfail/缺失报告不能算通过。保留逐用例结果。
2. **完整计算链**：补 MoE 专家计算、量化 Linear、状态更新及真实变输入 Graph replay；
   证明调用 Triton 实现，并与独立数值参照对拍。
3. **模型最小闭环**：小模型真实执行链与本地真实权重启动/生成分账；
   单卡证据不为 TP/EP 背书，随机权重不为 checkpoint 加载背书。
4. **上游变更对账**：比较绑定符号、签名、补丁与 nodeid 漂移；
   固定基线与待适配版本分开记录，串接回归与模型冒烟。

每步验证后单独提交；测试失败保留原始证据，不放宽断言制造绿色。
GPU 测试仅在 nmz28 的 `sglang_liph_adapter` 容器运行。
本轮代码目录为 `/public/home/lipenghui/lph-adapter/adapter-das`；
社区源码为同级 `sglang-0.5.22`。

## 严格组件门

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  bash ci/l15_readiness_gate.sh /home/lipenghui/lph-adapter/sglang-0.5.22
```

该入口强制实际执行五家族矩阵，不继承旧门禁的 observe/off 或失败总数预算。
它验证源码指纹、导入来源和真实 HIP 分配，输出 GPU/软件版本、两个仓库的
commit/dirty 状态和逐 nodeid 阶段结果。报告缺失、所有 skip、xfail、xpass、
setup/teardown 异常及超时均不通过；运行前删除旧逐 case 报告，预检失败覆盖旧汇总。

`coverage=cpu-contract` 的 sparse emu 项仅证明 CPU 数学契约，不计 GPU 计算覆盖。
组件门通过不等于真实模型已通过；后续模型闭环需另有启动/生成证据。

## 第一步验证（2026-11-03）

- nmz28 / `sglang_liph_adapter` / 单卡 GPU 0：Dense prefill/decode、
  fragmented layout、graph metadata 三个严格 case 全通过，保留逐 subtest
  结果，passing parent 不能掩盖子用例 skip。
- 同容器隐藏 GPU 后，矩阵与清单回归 **67 passed**；L0 **GREEN（31 warn）**。
- 原始报告：`results/l15-readiness/phase1-dense.json`（节点本地生成，不入版本库）。
- 这是严格执行机制和 Dense 的证据；其余家族与模型启动尚待后续步骤验证。
