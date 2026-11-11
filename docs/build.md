# 构建说明（wheel）

双 dist（governance §7.4）：`sgl-kernel` 纯影子内核 + `hcu-adapter` 适配层。
一次构建出两张 wheel，同戳互锁；构建期定版，上传环节不再改写版本号。

## 1. 前置条件

- python >= 3.9（验证栈为 py3.12）、pip、git。
- 纯 python 打包，不需要 DTK/GPU。
- 首次构建若当前环境缺 `setuptools.build_meta`，`build.sh` 会自动
  `pip install "setuptools>=64" wheel`；也可手动安装。
- 装 `hcu-runtime` extra 的机器需要配好 DTK nightly 归档的 pip 源
  （torch/triton/vendor 包从该归档解析）。

## 2. 一条命令构建

```bash
bash packaging/build.sh
```

`hcu-adapter` 的共享基础依赖仅为 `packaging>=23` 和配对的 `sgl-kernel`。
内核包保留原有 `torch>=2.0` / `triton>=2.1`，不另猜最低版本或固定某条
Torch/Triton 组合。实际组合由所选 Core/provider 的真实 METADATA 与部署
constraints 共同选择；共享包装不按宿主、构建参数或戳文本推断运行时栈。

`sgl-kernel` 的安装依赖与 vLLM 运行时是否调用 `sgl_kernel` 算子是两回事：
保留双包互锁，不因这条安装依赖而启用 SGLang 专用桥。

产物（同一个戳）：

```text
dist/sgl_kernel-0.0.0+hcu.<YYMMDDHHMM>.g<commit6>-py3-none-any.whl
dist/hcu_adapter-0.5.22+hcu.<YYMMDDHHMM>.g<commit6>-py3-none-any.whl
```

adapter 的元数据精确依赖同戳内核，两张必须成对安装：

```text
Requires-Dist: sgl-kernel ==0.0.0+hcu.<同戳>
```

## 3. 版本与戳规则

| 项 | 来源 | main 当前值 |
|---|---|---|
| 内核基线 | `sgl_kernel/version.py` 的 `__version__` | `0.0.0+hcu.dev0` |
| adapter 基线 | `packaging/hcu-adapter/VERSION` | `0.5.22+hcu.1` |
| 构建戳 | `git rev-parse --short=6 HEAD` + 当前时间 | `<YYMMDDHHMM>.g<commit6>` |

渲染规则：去掉基线里的 local 段，再拼 `+hcu.<戳>`。基线在 main 上不固定
（内核 `0.0.0+hcu.dev0`），拉 release 分支时定死；重配对上游时改对应文件即可。

`DAS_BUILD_STAMP` 仅记录构建来源、生成版本与配对内核 pin。即使戳中包含
`torch` 文本，也不改变依赖。构建入口不提供运行时版本 profile 或自动宿主
探测；空白/点号被规范化为空的显式戳会在构建前拒绝。

固定戳（复现某次构建）：

```bash
DAS_BUILD_STAMP="2611111025.ga64ad6" bash packaging/build.sh
```

## 4. 不带戳构建

```bash
bash packaging/build.sh --no-stamp
```

产物退回基线原值，适合本地开发/可编辑安装：

```text
dist/sgl_kernel-0.0.0+hcu.dev0-py3-none-any.whl
dist/hcu_adapter-0.5.22+hcu.1-py3-none-any.whl
```

此时 adapter 只钉内核公共版本 `sgl-kernel==0.0.0`（PEP 440 下可匹配任何
`0.0.0+local` 内核）。
`--no-stamp` 忽略已有构建戳；带戳与无戳构建使用相同的共享依赖策略。

## 5. 安装

按构建输出的确切两条路径成对安装，使用部署选择的真实 DTK provider
constraints 正常解析依赖（路径为示例）：

```bash
pip install -c /path/deployment-constraints.txt /path/sgl_kernel.whl \
  '/path/hcu_adapter.whl[sglang-runtime,sglang-diffusion,hcu-runtime]'
```

- constraints 属于具体部署，须与实际 Core/provider METADATA、原生 ABI 和
  包来源匹配；共享 Adapter 不固定某条 Torch/Triton 线。不能用 `--no-deps`
  把未满足依赖当成通过。
- 三个 extra 的分工：`sglang-runtime` 官方 sglang 运行时、`sglang-diffusion`
  扩散入口、`hcu-runtime` DTK vendor 栈（aiter/flash_attn/deep_ep/nixl/vllm 等）。
- `hcu-runtime` 另有三个可独立安装的子 extra：`hcu-runtime-kernels`
  （aiter/flash_attn/deepgemm 等内核）、`hcu-runtime-serve`（vllm/tilelang/
  lmcache/mooncake/nixl）、`hcu-runtime-aux`（torchvision/torchaudio）。
  `hcu-runtime` 通过自引用这三个子 extra 组合，包清单只在子 extra 维护一处
  （L0 校验组合关系）。镜像构建按子 extra 分 `RUN` 层、最后补装
  `hcu-runtime` 收尾，可避免单个 docker layer 超过仓库上限。Torch/Triton
  组合仍由部署 constraints 和实际 provider 依赖选择。
- 官方 sglang 本体以 `--no-deps` 安装，依赖由 adapter extra 提供。
- 从旧单 dist 版升级：先 `pip uninstall sgl-kernel`，否则同名残留与
  entry-points 双注册。

多次构建共用 `dist/` 时，不能分别挑最新 wheel；两张必须匹配实际 metadata
中的同戳互锁。构建戳只是来源信息，不能代替依赖解析或运行时能力验证。

## 6. 自检

```bash
python3 ci/l0_gate.py      # 双 dist 结构门禁（版本源、互锁、entry-points）
bash packaging/verify.sh   # A/B/C 三组合安装验证，需要 python venv
# 多对产物必须显式指定；此选项仅核对 METADATA，不创建环境/不安装
bash packaging/verify.sh --check-pair /path/sgl_kernel.whl /path/hcu_adapter.whl
```

`verify.sh <kernel.whl> <adapter.whl>` 可对指定 pair 跑原有 A/B/C；这些
`--no-deps` 结构/半装测试不证明依赖闭包。闭包须使用所选组合的真实
provider METADATA/产物，正常依赖解析并检查所有活动约束。既有组合的
Torch/Triton 版本只是验收来源，不限制其他可用组合，也不证明另一组合的
原生 ABI 或模型质量。包装解析、安装身份、原生 ABI、Graph 与模型质量分别
留证，不能用 import/help 或一项绿色替代。

`verify.sh` 需要目标机 python 带 `venv`/`ensurepip`；基础镜像里若缺 venv，
在镜像内执行或改用 `python3 -m venv --system-site-packages` 验证。

查看 wheel 元数据：

```bash
python3 -c "import zipfile,sys; z=zipfile.ZipFile(sys.argv[1]); print(z.read([n for n in z.namelist() if n.endswith('METADATA')][0]).decode())" dist/hcu_adapter-*.whl | grep -E "Version|Requires-Dist"
```

## 7. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `BackendUnavailable: Cannot import 'setuptools.build_meta'` | 环境缺 setuptools；重跑 `build.sh` 会自动安装，或手动 `pip install "setuptools>=64" wheel` |
| 解析到非预期 Torch/Triton 组合 | 检查部署 constraints、实际 Core/provider METADATA 与索引来源；共享 Adapter 不替部署选择版本 |
| `No matching distribution found for sgl-kernel==...+hcu....` | 先本地装同戳 kernel wheel；本地文件安装时 pip 不会去 index 找已满足的依赖 |
| `dist/` 多版本混装 | 构建前 `rm -rf dist build`，或按戳显式指定 wheel 文件名 |
| 构建日志的版本与 dist 文件不符 | 检查是否残留旧的 `DAS_BUILD_STAMP`；`--no-stamp` 会忽略它 |

## 8. CI

main 线的 nightly 由
[.github/workflows/nightly-wheel.yml](../.github/workflows/nightly-wheel.yml)
在 push 时构建两张同戳 wheel 并一次性上传 nightly 归档（L0 结构门禁已在 MR
阶段执行，发布流程不重复）；`workflow_dispatch` 只构建并保留 artifact，
不推 nightly。release 线同理：
workflow 在构建期写入同一个 `DAS_BUILD_STAMP` 后直接上传，不做上传后版本改写。
