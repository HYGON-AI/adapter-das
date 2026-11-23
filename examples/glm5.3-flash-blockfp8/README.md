# GLM-5.3-Flash blockwise FP8 TP8 确定性推理与 LightOp MoE

示例使用两节点各 4 张 gfx938 HCU、SGLang v0.5.22 和 adapter。权重使用 E4M3 FP8、128×128 权重块和 FP32 scale。验收使用权重修订 `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` 和 SGLang 提交 `baa9befa381511cd4d56f83980951e5a45fffc6d`。

默认加载同目录的 [`hcu_adapter.yaml`](hcu_adapter.yaml)。其中 `glm53_block_fp8_determinism: fixed` 为本模型打开 8 个确定性开关，`fused_moe: triton` 选择 Triton 专家 GEMM。

两节点使用相同的模型、源码路径和通信地址，分别传入节点编号 `0` 和 `1`。命令末尾是 adapter 仓库根目录，从仓库内运行时可以省略，从独立脚本包运行时需要填写。

```bash
bash tp8.sh /path/to/GLM-5.3-Flash /path/to/sglang-v0.5.22 10.0.0.1:25834 0 /path/to/adapter-das
```

默认服务端口为 25832，通信网卡可用 `NCCL_SOCKET_IFNAME` 指定。原有环境变量式启动仍可用。普通 adapter 对照加载 [`hcu_adapter.baseline.yaml`](hcu_adapter.baseline.yaml)，命令前可加 `DETERMINISTIC=0` 选择。自定义部署配置用 `HCU_CONFIG=/path/to/adapter.yaml`，优先于示例配置的选择。

对比专家 GEMM 时分别使用默认配置和 [`hcu_adapter.lightop.yaml`](hcu_adapter.lightop.yaml)。两者都启用 8 个确定性开关。LightOp 配置保留 Triton runner 并启用 adapter 的 LightOp 专家分发。节点需要已验证的 LightOp 运行包 `0.6.0+dtk2604.torch2110.2610261501.gf2f979`，加载后每个专家投影只常驻一份重排权重。

```bash
HCU_CONFIG=./hcu_adapter.lightop.yaml bash tp8.sh /path/to/GLM-5.3-Flash /path/to/sglang-v0.5.22 10.0.0.1:25834 0 /path/to/adapter-das
```

已有的 `MOE_BACKEND=triton` 和 `MOE_BACKEND=lightop` 启动方式仍可用。显式指定时，它们优先于 YAML 内的 MoE 选择。

把同一份 GSM8K test JSONL 提供给两个实例，文件每行含 `question` 和 `answer`。在 rank 0 节点的本目录运行下列命令，用同一份请求记录给两个实例采 profile。脚本会保存真实 prefill、EAGLE draft、target verify 和 draft extend 的 trace。测性能前重启服务，关闭 profiler。

```bash
python3 workload.py calls results/calls --port 25832 --dataset "$GSM8K_JSONL"
mkdir -p results/profile
python3 capture_continuous_graph.py --port 25832 --arm results/profile \
    --reference results/calls/call-responses.json --warmup none --no-recovery
```

使用相同并发、预热、输出上限，在同一组独占设备上依次运行普通对照和确定性版本。设置 `RESULT_DIR` 区分两组运行的结果。

```bash
python3 workload.py performance "$RESULT_DIR" --port 25832 \
    --concurrencies 1 2 --perf-repetitions 3 --perf-output-tokens 8
python3 workload.py accuracy "$RESULT_DIR" --port 25832 \
    --dataset "$GSM8K_JSONL" --sample-size 16
python3 workload.py cross-batch "$RESULT_DIR" --port 25832
python3 verify_cross_threshold_requests.py --port 25832 \
    --dataset-reference "$RESULT_DIR/gsm8k-results.json" \
    --call-reference results/profile/torch-graph-continuous/requests.json \
    --output "$RESULT_DIR/cross-threshold.json"
python3 compare_workloads.py results/baseline results/candidate
```

排查 BS2 时，在两个后端分别用下列命令先 tokenize，再将两条 token ID 序列作为一个批次发给调度器。计时从发送批次开始。常规性能命令使用独立文本请求。性能 JSON 会为每条输入保存首 token、输出进度、缓存命中和 MTP 计数。

```bash
python3 workload.py performance "${RESULT_DIR}-packed" --port 25832 \
    --concurrencies 2 --perf-packed --perf-repetitions 7 --perf-output-tokens 8
```

跨 batch 验收使用 greedy 和固定 seed 42，逐条对比输出 token、验证轮数、接受草稿数和提出草稿数。正式比较同时检查两组请求的 `cached_tokens`。

验收确定性候选时，可在两条跨 batch 命令中加 `--require-deterministic`。输出 token 或 MTP 计数不一致会保留原始结果并返回非零状态。

重放 GSM8K 第 447 题和同组第 457 题。fixture 来自上面的 16 题真实请求。脚本重新发出单跑、混跑、换序和单跑重试，逐条对比返回 token 和三项 MTP 计数。

```bash
python3 workload.py replay-pair results/replay447 --port 25832 \
    --fixture "$RESULT_DIR/gsm8k-results.json" --index 447 --require-deterministic
```
