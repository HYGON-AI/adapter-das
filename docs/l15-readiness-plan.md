# HCU SGLang 启动就绪门禁执行计划

目标：社区 SGLang 源码不变，通过 adapter 的 Triton 功能基线与模型执行链回归，
为明确的模型及配置提供启动依据。不以性能作为本轮验收条件。

## 当前迭代目标（2026-11-03）

用户要求按以下顺序持续推进，每个独立功能点验证后及时提交：

1. **社区源码完整性**：直接使用社区源码，不改写其 Python、C++/CUDA/HIP
   文件，也不在启动时向社区源码目录打磁盘补丁。适配实现留在 adapter，
   优先使用正式扩展接口，其次运行时接口适配。记录当前仍会写回上游的路径，
   逐项迁移；不能仅因用户没有手工改文件就宣称“源码零修改”。
2. **多模型功能一致性**：社区 SGLang + adapter 在 HCU 上运行多个真实
   checkpoint，覆盖 Dense、MoE、MLA、状态空间/混合架构及适用量化配置。
   保留上游功能与数值语义；硬件专属例外必须有具体原因和证据，不因缺少
   NVIDIA 库就直接排除可用 Triton/Torch 实现恢复的功能。
3. **功能与精度验收**：性能不作为当前门槛。验收包含真实加载、启动、
   prefill/decode、生成、数值对拍或语义金标，以及受测功能对应的状态、
   变输入 graph 和请求隔离回归。dummy 模型和历史战报不能替代当前组合
   的真实权重验证；启动成功、有限 logprob 本身不能证明模型精度。
4. **每日追新最后启动**：保留社区升级对账工具；其余优化和验证完成前，
   不配置或启用每日巡检。最终再接入每日社区 main 变更跟踪、适配修复、
   最新源码 + adapter 的模型回归与结果记录。

模型配方优先参考用户指定的
[inference-cookbook-das](https://github.com/HYGON-AI/inference-cookbook-das)，
逐模型记录 cookbook commit/脚本、checkpoint、卡型、依赖/框架/adapter
版本、实际参数、生成与精度结果。GPU 测试只在用户授权的远端空闲卡运行；
禁止在本节点执行全量 adapter 测试。

2026-11-03 用户新增 nmz7 调试授权：nmz7 与 nmz28 都使用
`sglang_liph_adapter` 容器，代码目录相同。nmz7 经资源检查确认 GPU
4–7 空闲、0–3 在用；首次回归选择物理 GPU 4，只暴露这一张卡。
每次任务启动前重新检查空闲状态，结果单独记录节点和设备，不沿用
nmz28 的环境结论；不清理或占用其他用户任务。

当前前置差距：`jit_*`、`dsv4_*`、`spec_h2d_pin`、
等存在源码磁盘写回，须以实际调用点逐项审计，
上述目标尚未完成。对象广播 `dist_bc` 已迁移为运行时只读 numpy buffer
兼容，保留已导入函数别名，不再写回 `common.py`。nmz28 隐藏 GPU 的
回归为 **6 passed**：包括以 Git 原始社区广播函数执行两 rank spawn/Gloo，
验证双向广播、空/中文/bytes/大对象和测试源文件字节不变。
这是广播路径的迁移证据，尚不代表整个 adapter 启动不再修改源码。
回归日志为 `results/l15-readiness/pristine-dist-bc.log`；本地 CPU 契约
4 passed（需远端社区树的 Gloo 项在本地不运行），L0 GREEN（31 warn）。
真实 checkpoint eager/graph 复验在启动前因 GPU 0 仅约 4% 空闲而被阻止，
未运行，也未占用其他卡；该集成复验仍待空闲窗口补齐。

新增 nmz7 授权后，在空闲物理 GPU 4 补齐真实 Qwen2-0.5B checkpoint
eager/graph 回归，**2/2 PASS**；报告为
`results/l15-readiness/iteration1-nmz7-checkpoint.json`。此次使用旧完整性
判据，社区树仍包含其他历史磁盘补丁，所以只作为广播迁移后的启动回归，
不作为新“原样源码”目标通过的证据。

### 源码完整性门禁

`HCU_ADAPTER_HOST=none python3 -S -m tools.upstream_source --up-tree <社区树>`
逐文件将实际字节与 HEAD Git blob 比较，不依赖索引、mtime、
assume-unchanged 或 skip-worktree。被忽略的新增 Python/JIT 源文件也
计为失败；子模块未递归验证则明确拒绝。报告保存所有差异路径。
该工具证明检查时的源码状态，不能单靠前后快照证明从未发生又被撤回的写入。

严格组件和模型入口默认在 GPU 预检前拒绝已有改动；组件运行结束、每个
模型 worker 结束后再核对一次，要求 commit 未改变且源码仍完整。
模型即使生成成功，只要改写社区源码也判失败并停止后续模型；预检失败
覆盖旧绿色报告。历史 27 case / 12 model-mode 的通过记录继续保留，
但不转授为新完整性规则下的通过结果。

nmz7 审计当前共享社区树：**10,594 tracked files，30 changed files，
0 added sources**。报告为 `results/l15-readiness/iteration2-upstream-source.json`。
这说明原样源码目标尚未达成；不通过 reset/删除现有工作树改动制造绿色，
后续在保留现场的前提下迁移适配并从干净社区树验证。
门禁与广播、编排工具组合远端 CPU 回归 **68 passed**；补充旧报告覆盖、
已暂存新增源码等拒绝测试后，最终源码完整性专项远端 **12 passed**。
日志为 `iteration2-nmz7-source-tests.log`、`iteration2-source-extra.log`；
真实模型 CLI 对当前脏树在任何 GPU worker 启动前拒绝，并保存 30 文件差异
（`iteration2-source-rejection.json`）；L0 GREEN（31 warn）。

### DeepSeek 双流 graph 保护迁移

`moe_dualstream_gate` 不再写回 `deepseek_v2.py`。仅为原始 `forward`
提供私有的分支选择 predicate，保留原函数对象、签名和已有别名。其他函数
仍可见真实 capture 状态，不修改 `alt_stream` 等共享实例属性；原社区
`forward_normal` 自己接收 `skip_shared_experts`、返回类型和输入元数据。
DeepEP、mega-MoE、已有专门 graph 路径、空 batch、eager/breakable
分支不变。仅在真实 `torch.version.hip` 下安装，安装顺序移到算子桥之后，
避免为提前改文件而提前导入模型。对应磁盘补丁白名单已移除。

nmz7 隐藏 GPU 的控制流/广播/安装结构回归 **24 passed**，包含直接执行
社区 HEAD 原始 `DeepseekV2MoE.forward`（非手抄协议）以及早先绑定的方法
别名、公开 install、参数透传和全局 capture 状态不变。证据为
`results/l15-readiness/iteration3-moe-runtime.log`。这些是控制流证据，
不冒称完整 DeepSeek checkpoint、数值或 graph 端到端已重新通过。

## 原四步实施与验证记录

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

## 第二步验证（2026-11-03）

- 提交 `5cb6c11`：五家族严格矩阵 **27/27 PASS**（25 GPU case + 2 CPU 契约）。
- 社区 MoE routing 接入后，计算链与 GDN 桥另跑 **13 passed**。
- 实际发现并修复 GDN bridge 多余转置；非方形状态、envelope stride、
  禁写、ring 和三轮 graph replay 均通过独立 FP32 对拍。
- 报告：`results/l15-readiness/phase2-all.json`；
  最终链路日志：`results/l15-readiness/phase2-final-chains.log`。

## 模型最小闭环入口

```bash
HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 HCU_ADAPTER_HOST=none \
  python3 -m tools.l15_model_smoke \
    --up-tree /path/to/sglang \
    --checkpoint /path/to/local-checkpoint
```

默认顺序执行 Dense、MoE、MLA、Mamba2、GDN hybrid 的两层 dummy 模型，
以及显式本地 safetensors checkpoint。每个模型分别启动 eager 和 graph，
经真实 Engine/scheduler 执行两批变输入，每批两个请求，每请求生成 8 token；
检查词表范围和有限 token logprob。graph 必须从 scheduler 读回真实 decode
graph 分配量，日志另保留 batch 1/2 的捕获过程；当前上游启动路径没有填写
role-switch 使用的 capture-bs 列表，不能以空列表误判未捕获。
结果保留 resolved server info、原始输出及每次启动日志。
该验证检查执行闭环，不把 dummy 的随机输出当语义金标，也不冒称 HTTP、
多卡 TP/EP、所有模型或所有量化格式已经可用。

每次启动使用独立 Linux 进程组。正常退出或超时后仅清理本次任务所属进程组，
避免失败的 scheduler 占住卡影响下一项。`--families` 可选择子集；
`--binding-manifest` 需与审查后的目标版本绑定一致。

## 第三步验证（2026-11-03）

- nmz28 单卡 GPU 0：Dense、MoE、MLA、Mamba2、GDN hybrid 五类两层
  dummy 模型，以及真实 `Qwen2-0.5B-Instruct` checkpoint，分别完成 eager
  与 graph 启动/生成，共 **12/12 PASS**。真实权重来自容器
  `/model/qwen2/Qwen2-0.5B-Instruct`，包含 safetensors 与 tokenizer。
- 每项两批、每批两请求、每请求 8 token；最终规则复核词表范围、有限
  logprob 与 token 对齐、eager 无 graph 分配及 graph 实际分配。六项 graph
  的 decode 分配均大于零；这不是只检查启动参数。
- 模型探测暴露旧 MLA 默认 `rope=0` 的覆盖缺口：新增受上游支持的
  `256+32`、`512+64` KV 布局，prefill/decode/真实 graph 三种模式，
  **6/6 参数化节点通过**。当前矩阵为 28 case；证据为第二步完整
  27 case 加本次新增 case 的增量验证，未声称重新完整跑过 28 case。
- MoE tiny 配方采用上游 fused RoPE 支持的 head64；MLA tiny 配方采用
  `512+64`，不再使用上游 grouped decode 不支持的任意小尺寸。
  这些是验证配方修正，未修改社区实现或放宽数值断言。
- 报告位于 `results/l15-readiness/`：`phase3-dense.json`、
  `phase3-dense-graph.json`、`phase3-models.json`、
  `phase3-models-retry.json`、`phase3-mla-final.json`，
  新 MLA 契约为 `phase3-mla-contracts.json`。旧失败保留在原报告，
  最终验证按 family/mode 选择成功执行，并用最终规则重新核验原始输出。
- 工具与失败判定回归 **91 passed**；包含真实 Linux 子进程及嵌套超时
  清理、空原因 XPASS 拒绝。未重新执行全量 adapter/vendor 套件。

## 第四步验证（2026-11-03）

- `ci/upstream_diff.py` 对不可变 Git 快照比较已声明的符号、签名/函数体、
  白名单补丁 needle 的位置/次数、上游测试 nodeid 与版本绑定文件。
  新路径复制同一个 needle 也要求审查。
- `tools.l15_upgrade_check` 串接静态审查、严格组件门、模型闭环；陈旧或
  其他 upstream commit 的报告不能使升级就绪，审查接受文件必须精确匹配
  baseline/candidate/changes。未完成全部阶段时保持 `ready=false`。
- nmz28 实际运行 `--baseline-ref HEAD --candidate-ref HEAD --audit-only`：
  **0 changes、15 baseline gaps、19 dynamic registration sites**；
  `same_revision=true`、`audit_only=true`、`ready=false`。这些缺口需运行时
  验证，不能将零差异解释为所有绑定完整或完成一次版本升级。
- 同容器工具组合回归 **91 passed**；补充 needle 新路径检查后，本地仅跑
  漂移/编排 CPU 测试 **12 passed**。L0 **GREEN（31 warn）**。
  故意修改签名、函数体、删除符号/nodeid/needle 的回归均能拒绝。
- 证据：`results/l15-readiness/phase4-workflow-audit.json`、
  同名 `-stages/upstream-diff.json`、`final-tool-tests.log`、`final-l0.log`。
  本次交付手动升级工作流，未配置定时巡检，也未验证新社区版本。
