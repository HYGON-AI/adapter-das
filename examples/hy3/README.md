# HY3 Channel W8A8 启动配置

本目录提供六份 `python3 -m sglang.launch_server` 启动示例。修改模型路径即可
复用命令；Python 环境需要安装配套的 SGLang 0.5.22、Adapter、Plugin 和计算库。
无需调用辅助脚本，也无需自己新建 YAML。

## 新增的三份 TP8 配方

以下三份对应 2026-11-24 补充的原始启动命令，未启用 DP attention 或 DeepEP。
与后面的三份 TP8/DP8 配方分别选择、分别保存。

| 包内 YAML（`hcu_adapter/launch_configs/`）与本目录同名 `.sh` | 原命令 | 保留的设置 |
|---|---|---|
| `hy3-int8-tp8-e4m3` | S7 INT8 | 显存比例 0.80、BF16、cache report、AITER MoE=false、融合 RoPE=0、W8A8 方法=2；不启用 NEXTN |
| `hy3-int8-tp8-e5m2` | 10.8 INT8 | 显存比例 0.80、BF16、cache report、相同四项计算开关；不启用 NEXTN |
| `hy3-fp8-tp8-nextn-e4m3` | S7 FP8 | 显存比例 0.9、BF16、NEXTN 2/1/3、关闭 radix cache、融合 RoPE=1 |

这些命令未增加 `--quantization slimquant_marlin`、DP、DeepEP、shared-expert
fusion 或 watchdog 等原脚本没有的参数。三份 TP8 YAML 明确设置
`HCU_FA_IMPL=vendor`、`SGLANG_KV_LAYOUT_HCU_FA=1`，沿用此前验证时的厂商 FA
及 HCU KV 布局组合。这是相对原始脚本补齐的两项配置，显式环境变量仍可覆盖；
它们不修改其他配方或 Plugin 全局默认值。DeepGEMM 开关保持原脚本的选择。
FP8 保留原脚本的 NCCL all-gather 同步 batch 开关。

10.8 的 INT8 TP8 E5M2 示例补充 `--dtype bfloat16`。原命令使用 `dtype=auto` 时，
抓图后的启动预热出现非法访存；仅显式指定 BF16 后，按原显存比例完成启动和生成。
该 auto 配置非法访存的具体实现原因尚未确定。

E5M2 配方需要配套 HY3 运行适配中的 Adapter FA3/E5M2 检查修复，才能在
显式 vendor 模式和支持的 HCU 上保留 FA3。仅安装本配置分支不能替代运行适配。

TP8 INT8 的 `W8A8_SUPPORT_METHODS=2` 由配套 Plugin 的 compressed INT8
实现读取。TP8 FP8 的 `USE_DCU_CUSTOM_ALLREDUCE=1` 按原样传入，当前
Adapter/Plugin 未发现该变量的读取点；它不构成自定义 AllReduce 已生效的证据。

两份 INT8 原脚本的动态库目录是部署路径，不属于模型配置。若环境依赖该目录，
先在部署环境设置正确路径，例如原 Python 3.10 环境使用：

```bash
export LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/amdsmi:${LD_LIBRARY_PATH:-}
```

换用其他 Python 或镜像时使用对应的真实目录。脚本本身不执行此机器路径设置。
FP8 原脚本的八卡可见性和 `kernel.numa_balancing=0` 也由部署环境提供。

## 原有 TP8/DP8 配置与启动命令

| 配置文件（位于 `hcu_adapter/launch_configs/`） | 本目录的同名 `.sh` | 对应原配方 | 融合 RoPE | FA 选择 |
|---|---|---|---|---|
| `hy3-fp8-tp8-dp8-e4m3.yaml` | `hy3-fp8-tp8-dp8-e4m3.sh` | S7 FP8，KV E4M3，显存比例 0.9 | 1 | vendor，HCU KV 布局=1 |
| `hy3-int8-tp8-dp8-e4m3.yaml` | `hy3-int8-tp8-dp8-e4m3.sh` | S7 INT8，KV E4M3，显存比例 0.9 | 1 | vendor，HCU KV 布局=1 |
| `hy3-int8-tp8-dp8-e5m2.yaml` | `hy3-int8-tp8-dp8-e5m2.sh` | 10.8 INT8，KV E5M2，显存比例 0.85 | 0 | vendor，HCU KV 布局=1 |

文件名使用量化、并行方式和 KV 类型，不绑定主机名。显存比例、NEXTN 参数、
page size 等仍由启动命令指定，选择器不重写这些参数。`--attention-backend fa3`
与 Adapter 的实际 FA 实现是两个选择；上表配方通过 `HCU_FA_IMPL=vendor`
明确选择厂商 FA。

三份 DP8 配方均显式选择 vendor FA 与 HCU KV 布局，确保 KV
写入和注意力读取使用同一种布局。配置 UI 与包内 YAML 同步维护；显式环境变量
仍优先于这些配方值。E5M2 配方需要配套 Adapter 的 FA3/E5M2 启动检查适配。

两类配方的自动选择仅覆盖本地 `config.json` 中的 `hy_v3` / `HYV3ForCausalLM`，且量化为
compressed-tensors Channel W8A8、激活为动态 per-token INT8 或 FP8 的模型。
原有 TP8/DP8 配方的启动参数必须匹配单节点 TP8 / DP8、DeepEP auto、DP attention、DP lm head、
FA3、NEXTN，以及上表中的 KV 类型。INT8 允许原脚本的 `slimquant_marlin`。
其他模型、量化和未收录的拓扑保持原有启动逻辑。

## 环境变量去了哪里

YAML 的 `hosts.sglang.env` 保存原脚本中的队列、拷贝、通信和计算开关。
Adapter 在安装框架桥之前选择配置，将选定环境传给重新执行的 Python，
使 GLIBC/HIP 参数在新解释器启动时已经存在。Linux 上重新执行保留 PID，
内部标记防止循环执行，随后生成的工作进程继承相同环境和配置路径。

优先级为显式 env > 配置文件 > 自动规则 > 默认值。没有外部配置时自动选择
包内 YAML，设置来源记为 `config`；显式 env 记为 `explicit`。启动日志示例：

```text
HCU SGLang config: builtin:hy3-int8-tp8-dp8-e4m3.yaml (auto)
HCU HY3 env: ... SGLANG_USE_FUSED_RMS_ROTARY=1(config) ...
```

显式 `HCU_CONFIG` 或自动发现的 `hcu_adapter.yaml/yml/json` 优先于内置配方。
自定义文件是完整配置，未填写的值不会自动从 HY3 内置 YAML 补齐。可复制上表
中的文件再修改，也可在 `tools/hcu_config_ui.html` 中选择六份 HY3 配方之一、编辑值并
导出。保存为运行目录的 `hcu_adapter.yaml`，或通过 `HCU_CONFIG` 指定路径。
同一个文件中 `hosts.sglang.env` 的同名值优先于 `ops` 映射。

INT8 配方将 FP8 专家开关设为 0；显式要求 INT8 使用 FP8 专家计算时明确报错。
旧名 `SGL_CHUNKED_PREFIX_CACHE_THRESHOLD` 的显式值仍优先于 YAML，正式名
`SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD` 同时出现时优先使用正式名。
`SGLANG_ENABLE_SPEC_V2=1` 暂按原配方保留；当前核对的 SRT 源码未找到读取点，
不能据此判断推测解码生效，NEXTN 由命令参数及实际运行路径决定。

显卡可见性、NIC/IB、Python/PYTHONPATH、LD_LIBRARY_PATH、计算库安装位置和
缓存目录由部署环境提供；主机 `kernel.numa_balancing` 由运维设置。内置配方
不写入这些机器路径或修改 sysctl，也不会安装或替换 DeepGEMM 等计算库。

## 基线与验证范围

本扩展基于公司 `release/sglang_v0.5.22` 的 `16666ac`，复用已合入的
PR #162 提供的包内 YAML、`hosts.sglang.env` 校验和 wheel 打包机制。
需要安装包含本次 HY3 配方及加载代码的 Adapter，并配套使用 HY3 运行适配
PR 的 Adapter/Plugin 与计算库；仅复制 YAML 到旧环境不会获得自动选择能力。

专项回归覆盖配方选择、外部配置和显式覆盖、未支持拓扑、启动顺序、真实进程
重新执行以及 UI 配方与 YAML 的一致性。

2026-11-26 在 gfx938、DTK2610、SGLang `baa9bef`、Adapter `42dbfd4`、Plugin
`1095795` 加本轮 weak-ref 和融合 KV 布局补丁的组合下，INT8 TP8 E4M3、FP8 TP8/DP8
E4M3、INT8 TP8/DP8 E4M3 已分别按 YAML 自动选择完成抓图和正常生成。显存比例
保持 0.8、0.9、0.9，两组 DP8 保持 NEXTN 2/1/3；配置回归 40 项通过。
三组各返回 `42` 和 `巴黎`（INT8 带句号），两组 DP8 另通过并发重复请求，均正常 stop。
实际参数核对包括新版框架规范化后的 `attn_dp_size=8、dp_size=1`，八个 scheduler
仍执行 DP attention。服务均已停止。

同日，10.8 的 gfx936 八卡环境使用相同源码组合和配套 DTK2610 计算库，完成两份
INT8 E5M2 配方验证：TP8 显式指定 BF16、显存比例 0.80；TP8/DP8 保持 BF16、
显存比例 0.85、DeepEP auto 和 NEXTN 2/1/3。DP8 YAML 补齐 vendor FA 和
HCU KV layout=1 后，实际后端保持 FA3。两组均通过抓图、顺序及并发生成，返回
`42`、`巴黎。`，全部正常 stop，测试服务已停止。GPQA 精度、性能和长期稳定性
仍待验证。
