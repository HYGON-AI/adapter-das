# 真实模型 TP 配置验收

`tools.l15_model_smoke` 使用社区 Engine 执行真实 checkpoint 或显式
dummy 模型；二者的权重证据分开记录。默认 `--tp-size 1`，
可选择 1/2/4/8。每个 TP 规模、模型和 dtype 都须分别验收。

启动前确认所选设备均空闲，再通过 HIP/CUDA 可见性暴露恰好 TP 数量的
设备。预检会逐张分配、计算及同步，并记录 `runtime.devices`；
它验证设备可用性，不负责判断是否有其他用户作业。

`--host adapter` 默认禁止加载额外 plugin；显式使用
`--host adapter+plugin` 时，每个 TP 配置仍须检查公开 plugin loader
实际注册了 `hcu` hook。预检和 worker 同时绑定 host 与 TP 参数，
不能以多卡设备可用代替 plugin 加载成功。

```bash
HCU_ADAPTER_HOST=none \
HIP_VISIBLE_DEVICES="$DEVICE_IDS" CUDA_VISIBLE_DEVICES="$DEVICE_IDS" \
python -m tools.l15_model_smoke \
  --up-tree "$HCU_UP_TREE" \
  --families checkpoint --checkpoint "$HCU_CHECKPOINT" \
  --tp-size 2 --modes eager,graph --profile accuracy \
  --out results/model-tp2.json
```

不要继承与本轮选择冲突的 `ROCR_VISIBLE_DEVICES`。预检保留社区源码
绑定、导入来源及完整字节检查。控制器向 worker 和 Engine 传递 TP
数量，并核对实际 server info 和 scheduler 状态；请求 TP2 但运行
TP1、worker 缺少 TP 配置或设备数量不符均不能通过。

主报告和每个 worker 的 `tp_size` 必须一致。旧报告没有该字段时仅按
历史 TP1 解释，不能用于证明多卡。实际 graph 分配、三批固定输入、
缓存复用与变输入生成，以及 checkpoint 前后字节绑定继续按原合同验证。

精度参考从已绑定的报告读取 TP 配置，使用隐藏 GPU 的 CPU FP32 参考：

```bash
HCU_ADAPTER_HOST=none HIP_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= \
python -m tools.l15_checkpoint_reference \
  --smoke-report results/model-tp2.json \
  --out results/model-tp2-accuracy.json
```

精度报告以 `execution_tp_size` 标记受测配置，继续使用原版本化策略。
模型 smoke 的 `passed` 只证明启动、生成及对应证据完整，
必须同时检查独立精度报告；TP2 成功也不外推 TP4/TP8、EP 或其他模型。
量化参考仍须使用已验证的显式量化参考入口，不能作为普通 FP32 模型处理。

## 上游同步后的接口回归

对接 adapter 上游 `b3f4bb0250dbf4b84bd71473fe03817204e18293`
时，同时保留上游 host/plugin 检查和新增 TP 配置：
`preflight` 的第四个位置参数仍为 host，设备数量通过
`device_count=` 显式传递。控制器向 worker 同时传递 host 与 TP，
并核对报告中的两项配置。

远端隔离快照完成 269 项 CPU 回归，GPU 全部隐藏，同一解释器结束时
CUDA 未初始化；L0 GREEN，31 条已有警告。组合回归实际执行生成的
设备探针，覆盖 adapter/plugin 两种模式、两设备逐个分配，以及 plugin
未注册 hook 时拒绝启动。另覆盖 TP2 worker 参数遗漏、忽略或布尔值伪装。
该结果只证明工具与接口合同；新上游基点上的 HCU 矩阵、真实模型和精度
仍须分别验收，不能继承此前代码组合的通过结论。
