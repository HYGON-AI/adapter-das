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
