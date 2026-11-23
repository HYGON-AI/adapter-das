# GLM-5.3-Flash blockwise FP8 TP8 确定性推理

示例使用两节点各 4 张 gfx938 HCU、SGLang v0.5.22 和 adapter。权重使用 E4M3 FP8、128×128 权重块和 FP32 scale。验收使用权重修订 `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` 和 SGLang 提交 `baa9befa381511cd4d56f83980951e5a45fffc6d`。

默认加载同目录的 [`hcu_adapter.yaml`](hcu_adapter.yaml)。其中 `glm53_block_fp8_determinism: fixed` 为本模型打开 8 个确定性开关，`fused_moe: triton` 选择 Triton 专家 GEMM。

两节点使用相同的模型、源码路径和通信地址，分别传入节点编号 `0` 和 `1`。命令末尾是 adapter 仓库根目录，从仓库内运行时可以省略，从独立脚本包运行时需要填写。

```bash
bash launch.sh /path/to/GLM-5.3-Flash /path/to/sglang-v0.5.22 10.0.0.1:25834 0 /path/to/adapter-das
```

默认服务端口为 25832，通信网卡可用 `NCCL_SOCKET_IFNAME` 指定。原有环境变量式启动仍可用。普通 adapter 对照加载 [`hcu_adapter.baseline.yaml`](hcu_adapter.baseline.yaml)，命令前可加 `DETERMINISTIC=0` 选择。自定义部署配置用 `HCU_CONFIG=/path/to/adapter.yaml`，优先于示例配置的选择。

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

跨 batch 验收使用 greedy 和固定 seed 42，逐条对比输出 token、验证轮数、接受草稿数和提出草稿数。正式比较同时检查两组请求的 `cached_tokens`。

验收确定性候选时，可在两条跨 batch 命令中加 `--require-deterministic`。输出 token 或 MTP 计数不一致会保留原始结果并返回非零状态。
