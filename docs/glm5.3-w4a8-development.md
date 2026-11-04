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

## 2026-11-04 editable 安装后的冷缓存预热自锁

节点 21 的独立 RoPE 检查在 tvm-ffi 预热时停住：父进程持有构建 flock，子进程即使移除 `PYTHONPATH`，仍通过 editable 安装找到 `sitecustomize`，重新进入 adapter 并等待同一锁。实际进程链和内核等待点为父进程等待构建子进程、子进程 `locks_lock_inode_wait`，未进入 RoPE 数学检查。

构建子进程环境现同时移除 `PYTHONPATH`，并显式设置公开及传播的 host 为 `none`；预热窗口中显式传入的子进程 env 也按此处理。父进程环境保持原样。重跑后冷缓存成功生成 `libtorch_c_dlpack_addon_torch211-rocm.so`，主诊断继续运行；证据是 `rope-jit-check-v1-deadlock.log`、`rope-jit-check-v2.log`。不改动镜像中的 Torch/DTK 算子库，模型生成验证仍待完成。

镜像的 `/usr/local/bin/ninja` 是 Python 入口，JIT 编译调用它时再次执行已安装的 `sitecustomize`。启动配方通过已安装 `ninja.BIN_DIR` 将同一 wheel 的原生二进制置于 PATH 前部，保留编译命令、工具版本和父服务的 adapter 配置。正在运行的模型继续等待其原有 JIT 完成，未为此重启。

## 2026-11-04 FlashMLA code object 兼容诊断

组合 v8 已越过 RoPE/concat，在镜像内 FlashMLA sparse prefill 报 `no kernel image`。同卡、同输入的原生镜像调用通过；指定 DTK26.10 镜像内的 8/64 heads、带/不带 `topk_length` 均失败。设备代码中有 gfx936，不能据此推断可执行。当前节点的 HYHAL 为 `6.3.31-V1.5.7 20261029-124220`；同一 DTK 编译器生成的最小 HIP 核中，code object v5 可查找函数并执行，v6 的模块加载成功但函数查找返回 500。

在隔离目录用镜像版本对应的 FlashMLA 提交 `e7b26ad186028b07b9267e296eaf0acaa5cc7069`、CUTLASS 子模块 `33e45645599f371c2382e67920085498f5905955` 构建，显式设置 `FLASH_MLA_CODE_OBJECT_VERSION=5`，未修改算子源码、Torch 或 DTK。隔离包的 5 组 BF16 sparse prefill（tokens 1/6/17/256，heads 8/64）及更换输入后的 graph replay 均通过预先声明的 FP32 参考误差检查；最大绝对误差小于 0.0044。原安装包保持原样，完整模型生成和 P/D 仍待验证。这是当前节点/镜像组合的证据，不代表所有 v6 运行时都不支持该格式。

启动脚本现保留调用者的 `PYTHONPATH`，使此类已核实的隔离包可在同一镜像中验证；三份任务源码仍排在前面。复现构建时，在上述固定源码与子模块目录执行：

```bash
export HCU_ADAPTER_HOST=none FLASH_MLA_CODE_OBJECT_VERSION=5 FLASH_MLA_OPT=1 MAX_JOBS=8
python3 -m pip wheel "$FLASH_MLA_SOURCE" --no-deps --no-build-isolation -w "$WHEEL_DIR"
python3 -m zipfile -e "$FLASH_MLA_WHEEL" "$VENDOR_OVERRIDE_DIR"
```

仅对需要该验证的启动设置 `PYTHONPATH="$VENDOR_OVERRIDE_DIR"` 与 `FLASH_MLA_ROOT_DIR="$VENDOR_OVERRIDE_DIR/flash_mla/asm"`。先确认实际导入的 `flash_mla.cuda.__file__` 和哈希，再继续服务验收；解包尚未完成时启动的一条早期探针导入了原包，该条未作为隔离包验证。证据清单为任务产物 `flashmla-cov5-provenance.json`、`code-object-check.json`、`flashmla-cov5-validation-v1.json`，原始失败记录保留。

## 2026-11-04 单机真实请求与 INT8 专家指令确认

组合 v9 完成 TP8/EP8 加载及 warmup。与原生使用相同固定输入 ID 和文档采样参数的三条请求均正常 stop，最终答案为 `323`、`H₂O`、`18`。水分子式的 Unicode 下标使第一版仅匹配 ASCII 的检查失败；保留原响应和失败记录，按 NFKC 表示等价重新检查同一响应后通过，没有重新生成或调整数值误差阈值。本轮仍为 MTP/graph 关闭的单机诊断。

额外抓取一条 33 输入、2 输出 token 的完整诊断请求，8 个 rank 的 gzip/JSON 均解析通过，本地归档内压缩文件 SHA256 逐项一致。trace 共命中 2400 次 `fused_moe_kernel_gptq_awq` 和 9312 次 `_per_token_quant_int8`。对应 8 份实际 Triton 编译产物中，A 为 `i8` 指针，packed B 为 `i8` 字节存储，矩阵指令为 `v_mmac_i32_16x16x32_i8`，输出为 BF16。结合 CT 的 INT4 权重布局，确认当前专家计算保留 A8；不能根据旧配置文件名中的 `w4a16` 或 BF16 输出类型判断量化降级。

独立抽查真实第 3 层一个专家，DeepGEMM masked 的 gate/up 和 down 两次 INT8 GEMM 对 QDQ 参考的 NRMSE 分别为约 `5.9e-9`、`4.5e-5`。在完全相同的 BF16 输入和实际 gate/up 中间结果上，原生与候选的输入 INT8/scale、SiLU-mul 后 INT8/scale 四组张量在 17 个有效行内逐元素一致；先前 CPU 舍入参考的两次严格检查失败仍保留。这些局部证据不替代完整模型中间结果与 teacher-forced 精度对照。

版本、trace 哈希和验收范围见 [单机候选 profile 记录](glm5.3-w4a8-candidate-single-profile.json)。目标四节点第一次启动停在 RCCL 初始化：镜像的 verbs/provider 不能打开当前 SHCA，且缺少所选网络插件；原生匹配的 verbs/provider 和网络插件通过隔离路径接入后，23/24 两节点实际 SGLang PyNccl 的 FP32、BF16、INT32 归约及两次更换输入的 graph replay 均精确通过，日志确认使用 4 个 SHCA 和 RDMA 插件。完整 DeepEP、Mooncake KV 传输、MTP、目标 graph、有限矩阵及候选 32-rank profile 继续验证。

## 2026-11-04 Mooncake SHCA 与 GPU RDMA

镜像的通用 Mooncake 包按标准 verbs 头文件构建，与当前节点的 SHCA u17 ABI 不匹配；仅替换运行时 verbs/provider 后，仍在 completion vectors 检查失败。采用镜像对应的 Mooncake 提交 `6eb1b21a06cad9e83569815847655ab1673c4e78`，按其 SHCA 变体设置 `USE_HYGON=ON`、`USE_SHCA=ON`、`USE_FAKE_HIP_RPC=ON`，使用与可用 verbs/provider 配套的 SDK 头文件重新编译。1741 个跟踪源码文件逐项 SHA256 与固定提交的本地副本一致，没有修改 C++ 源码。

构建只下载并解包隔离的 JSONCPP/YAML 等依赖，没有全局安装驱动或更换 Torch/DTK。SDK 作为 system include，避免其旧枚举警告被目标自身的 `-Werror` 提升为错误。产物是 CPython 3.12 的本地 Linux wheel；没有宣称 manylinux 可移植性。任务容器中原通用包的 146 个文件先逐项校验备份，再卸载通用 distribution，并通过隔离前缀加载 SHCA wheel，保留上游对重复 distribution 和头文件 ABI 的检查。恢复时先移除隔离前缀的启动引用，再从对应节点备份恢复原文件。

23/24 两节点通过公共 `mooncake.engine.TransferEngine` 验证 GPU→GPU RDMA：40 KiB 和 4 MiB 非均匀字节模式、128 字节偏移，接收内容及前后保护区均逐元素精确一致，两进程退出码均为 0。初始化日志确认 SHCA RDMA transport，未设置 `MC_FORCE_TCP`。控制消息使用独立 TCP socket；第一版使用共享盘控制文件，在不同 NFS 服务端之间未及时可见而超时，该失败记录保留，不能作为数据传输失败或通过的判据。

固定版本、产物哈希和范围见 [通信验证记录](glm5.3-w4a8-network.json)。四节点 v2 已采用这组依赖继续启动；本节只证明两节点通信，尚不证明完整 DeepEP、KV 交接、MTP 或目标 graph 已通过。

## 2026-11-04 连续 W4A8 设备代码与 DeepEP u17 ABI

四节点 v2 的 32 个 scheduler 均越过通信初始化和权重加载。prefill 在实际 DeepEP normal 计算中报 `W4A8 WASP kernel launch failed`；decode 在 low-latency 初始化时由 RocSHMEM 的 `ibv_alloc_parent_domain` 返回 95 并终止。四个进程均已退出，未将退出码 137 直接归因为显存不足。

镜像 DeepGEMM 为提交 `b061ec70272a001fca622ea814d3ee163e24bb05`。独立真实专家检查中，默认 WASP 1006 启动失败，原 HIPC 1002 明确报 `no kernel image`。为隔离 1002，诊断构建保留该实现的 kernel body，排除 WASP include/分发并明确拒绝 mode 1006，再使用上游标准 Torch hipify。相同 HIP 源码和编译参数仅切换 code object 格式：COV5 完成 gate/up 与 down 两次 INT8 GEMM，17 个有效行对 BF16 QDQ 参考均逐元素一致；COV6 编译成功后仍报 `no kernel image`。这只验证独立 HIPC 1002 模块，尚未接入完整服务，也未验收 WASP 1006。

DeepEP 采用镜像对应提交 `a755fa7e14631af7e8a6a0d6ead217101d3bfdda` 的 `BUILD_SHCA=ON` 构建，匹配 u17 SDK、`U16_LID=OFF` 和 COV5。其 build.sh 显式 checkout RocSHMEM `b71ac40294bb10c5ffb11b94bed7d7d20529ab02`，不同于 gitlink 的 `1c22be1a9e83d6500db171b5902e3b0e3d649481`，按实际构建契约记录版本。465 个非版本元数据的跟踪文件与固定源码逐项一致；setup.py 按已声明逻辑重写 `deep_ep/version.py`，保留原始严格字节检查失败记录，不宣称全部文件未改变。

隔离 SHCA wheel 保留原 DeepEP 安装。23/24 两节点共 16 rank、真实 hidden=6144/experts=384/topk=8，分别完成 3-token BF16 和 17-token INT8 low-latency dispatch/combine。所有路由都跨节点，检查每个有效 token、FP32 INT8 scale 和加权 combine，16/16 rank 逐元素精确通过。这不包含完整模型、MTP 或 graph 验收。产物与验证范围见 [后续 ABI 诊断记录](glm5.3-w4a8-vendor-abi.json)；后续需将连续 GEMM 的已验证实现作为可审查的 ABI 适配接入，再继续完整 P/D、有限矩阵和 32-rank profile。

## 2026-11-04 可审查的 HIPC COV5 接入

新增独立构建器 `tools/build_deepgemm_w4a8_cov5.py` 和可选机制 `hcu_adapter.deepgemm_cov5`。前者固定实际编译的四份 vendor 源码/头文件摘要，只生成独立 HIPC1002 模块并原子发布；后者仅在模块显式出现在 Python 路径时接入，验证完整 wheel 版本、wrapper 字节及扩展 ABI 元数据。以私有 op facade 和模块局部默认值保留原函数对象、签名、校验与已缓存别名，不改动共享配置或 masked 等其他算子。新 catalog、registry、UI 快照和已有 `HCU_DISABLE_OPS` 快关同步登记，没有新增环境变量。

真实专家检查经过原公共 API，默认选择 HIPC1002，gate/up 和 down 保持 INT8/FP32 scale 并通过原 QDQ 误差检查。两次 GEMM 各做两次变输入 graph replay，沿用原误差门限通过；缓存别名、函数 code、共享配置和 masked op 身份检查通过，显式 WASP1006 拒绝。任务产物为 `w4a8-contiguous-provider-mode1002-v1.json`；该组件验证仍不替代完整服务验收。配方统一采用公开 `--cuda-graph-max-bs-decode 16`，等价于先前兼容别名的目标设置。


## 2026-11-04 两节点 prefill 就绪及实际库核对

四节点 v3 的 prefill（21/22，PP2、TP8、CP8、EP8、normal）完成真实 CT 权重加载和 warmup，主节点 `/server_info` 返回 200。逐个读取 16 个 scheduler 的进程映射并校验已映射文件，确认使用同一 HIPC COV5 provider、FlashMLA COV5、SHCA DeepEP、SHCA Mooncake、配套 verbs/provider 和 RCCL 网络插件；不能仅根据启动环境变量认定实际库已生效。组件哈希及范围补入 [ABI 记录](glm5.3-w4a8-vendor-abi.json)，原始映射为任务产物 `candidate-pd/runtime-maps-node21-v3.json`、`candidate-pd/runtime-maps-node22-v3.json`。

同次 decode 在目标 graph 捕获时被原生 W4A8 消费者的 INT8/FP32-scale 校验拒绝，退出码为 137。分发配置修复归 companion plugin PR #11，不通过删除校验、关闭 graph 或降级 W4A16 处理。此阶段尚未完成一次候选 P/D 请求，也未交付候选 32-rank profile。

## 2026-11-04 避免重复包装 DeepEP vendor API

候选 decode v4 仍在目标 graph 捕获时被 INT8/FP32-scale 校验拒绝。仅运行 bootstrap、不加载模型的检查确认，native 与 TBO 均已使用预期 plugin dispatcher，但其 `_DEEPEP_LL_DISPATCH_USES_QUANT_TYPE` 为 false：旧 adapter bridge 将原生 vendor API 包装为 `use_fp8` 签名，隐藏了 `quant_type`，该旧接口也不能表达 INT8。配方现显式设置已有 `HCU_DEEPEP_BRIDGE=0`，由指定底座及 plugin 的既有 ABI 分支接管；保留旧 bridge 供其原有宿主版本使用，没有再叠一层 wrapper。

同时纠正路径记录：本底座的 CT + DeepEP + DeepGEMM 选择 `HCUW4A8Int8DeepEPMoEMethod`，复用其专家实现，但 CT 配置、checkpoint 名称和加载格式不变；其 normal 与 low-latency 均声明 INT8。plugin 修正 DeepGEMM 选择下的 W4A8 vendor 分支，撤回首次尝试中多余的 setter 配置补写。原消费者校验和目标 EAGLE/graph 设置保留，新的完整服务复验仍待完成。

## 2026-11-04 FlashMLA 静态池索引接口复用

decode v5 已越过 INT8 契约并完成一组 graph 捕获，随后 MTP 初始化导入 `FlashMLABackend` 时因 shadow 缺少 `hcu_create_flashmla_kv_indices` 退出。镜像没有注册对应 `torch.ops.sgl_kernel` 算子，LightOp 也未提供此出口；底座已有同语义的 `create_flashmla_kv_indices_triton`。新增薄接口复用该实现，保留原参数名、显式 stride、可选 start、page64、有效位置写入及尾部不变的语义，空批次/零宽输出直接返回；没有新写索引算法或更换 attention kernel。catalog、runtime registry、UI 快照和档位矩阵同步登记为 candidate。

节点 23 经实际公开 API 检查 6 组输入：连续/带 stride 源与输出、有/无起点、长度 0/1/63/64/65/128/257/4097，以及空批次/零宽输出。所有有效页索引与整数参考精确一致，未写尾部和输出保护列保持原值；4 个非空组合各完成两次更换源表、请求索引和起点后的 graph replay，仍逐元素一致。记录为任务产物 `candidate-pd/flashmla-indices-check-v1.json` 和同名日志。未新增单测，完整 MTP 与 P/D 请求仍待复验。

## 2026-11-05 原生 HCU top-1 MTP 随机采样选择

v6 的 32 个 scheduler 均已就绪，decode 完成 target-verify、draft-decode、draft-extend graph 捕获及 16 个 attention-DP rank 的启动 warmup。首条固定 33-token、temperature1/top-p0.95 的真实 P/D 请求完成 prefill，随后 decode 进入依赖 FlashInfer 头文件的 JIT 采样并退出；此前 warmup 不能代表此随机采样路径已通过。

指定底座已有 `sample_mtp_target_ids`：HCU top-1 MTP 保留温度、top-k/top-p/min-p、请求 seed/positions，先采样 target IDs，再进行树匹配。CUDA 平台 facade 令 `eagle_utils._is_hip=False`，已有 sampled IDs 没有被后续 `_verify_uses_greedy` 条件采用。`eagle_hcu_sampling.py` 通过正式 HookRegistry 在真实 HCU、原生能力及签名均符合契约时，补齐 top-1 非 rejection 的选择条件。它先保留原 predicate 结果，非 top-1 和 rejection 请求继续使用原结果；没有新增采样算法或改写 `eagle_sample`、logits、概率及随机数。

实际 bootstrap 和 Hook 应用检查完成 9 个 selector 组合，覆盖 top-k1/2/未配置、greedy 和 rejection。已缓存的 helper 别名随 Hook 传播，公开 `eagle_sample` 函数对象及 code 未改，模块平台常量未改。结果为任务产物 `candidate-pd/eagle-policy-check-v1.json`；这只是实际接线/选择检查，真实请求复验和完整 profile 仍待完成。未新增单测，SGLang 跟踪源码未改。
