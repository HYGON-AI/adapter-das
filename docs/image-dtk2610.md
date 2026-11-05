# DTK 26.10 基础镜像部署指南（py3.12 → 官方 sglang 0.5.22 + adapter）

> 本文回答：**在 `10.16.1.152:5000/hcu/base/dtk:26.10-ubuntu22.04-py3.12` 基础上，
> 跑通"官方 sglang v0.5.22（零修改）+ adapter"需要装哪些包、为什么、按什么顺序。**
> 与 [bootstrap.md](bootstrap.md) 的关系：那册是 dtk26041/py3.10 现役栈的手工部署手册
> （PYTHONPATH 源码态）；本文是新基底镜像的 pip 态部署，二者并存直至 26.10 转正。
> 依赖清单底料 = 上游 v0.5.22 `python/pyproject.toml` 全量 dependencies（逐条核对过）。
> **[VERIFY-26.10]** 标记 = 需要 26.10 容器首跑定案的项（第 6 节有汇总）。
> 前置阅读：[hygon-oss-landscape.md](hygon-oss-landscape.md)（供应商库图谱）。

## 0. 基线事实

- **镜像**：`10.16.1.152:5000/hcu/base/dtk:26.10-ubuntu22.04-py3.12`（DTK 26.10，Python 3.12）。
- **py3.12 意义**：上游 v0.5.22 声明 `requires-python = ">=3.11"`——旧 py3.10 基线**不满足**，
  此前全靠 PYTHONPATH 源码态绕过；新镜像**首次原生满足**，pip 态安装从此合法。
- **依赖声明与 import 面是两回事**（pip 装包策略的根基）：
  v0.5.22 的 pyproject 声明 `sglang-kernel==0.4.10` 等一排 CUDA-only/不可解析依赖
  → 主 wheel **必须 `--no-deps` 装**；代码运行时只认 `import sgl_kernel`
  （实测 v0.5.22 全树 import 面）→ 我们的影子包按模块名截击，dist 名无关紧要但见 §4 接线点。
- **发版物四件**：adapter wheel（dist `sgl-kernel`，含 `hcu_adapter/`+`sgl_kernel/`+`sitecustomize`，
  见其 pyproject）→ 官方 sglang wheel（`--no-deps`）→ 插件 wheel（`--no-deps`）→ 环境级钉版。

## 1. 不装的依赖（CUDA-only 或被我们挡位替代）

| 上游声明 | 不装的原因 | 我们的对位 |
|---|---|---|
| `torch==2.14.1`、`torchaudio`、`torchvision`、`torchcodec` | 镜像自带 DTK HIP torch；重装=覆盖成 CUDA 版 | **[VERIFY-26.10]** 镜像自带的 torch 版本号；vision/audio 用途按自带 torchvision/torchaudio 匹配版装 |
| `sglang-kernel==0.4.10` | CUDA AOT 内核，无 gfx93x 产物 | **adapter wheel（影子包，模块名 sgl_kernel）** |
| `sgl-flash-mla==1.0.0` | CUDA 内核 | 影子 `sgl_kernel.flash_mla` emu / FlashMLA-das vendor 档（`mla_attention_dtk`） |
| `sgl-deep-gemm==0.2.3.post2` | CUDA JIT 栈 | deepgemm vendor 桥 / triton 档（governance 三档路由） |
| `sgl-deep-ep==0.1.2.post1` | CUDA/NVSHMEM | DeepEP-das 档 / NCCL fallback |
| `flash-attn-4`、`flashinfer_python[cu13]`、`humming-kernels[cu13]`、`quack-kernels`、`nvidia-cutlass-dsl`、`nvidia-mathdx`、`nvshmem4py-cu13`、`cuda-python` | CUDA/Blackwell-only，wheel 在 HIP 环境装不上或无意义 | 四级路由接住（lightop→aiter→triton→响亮 stub）；fa3 走 `HCU_FA3_EMU` |
| `tilelang==0.1.12` | **特殊**：版本号对（上游已钉 0.1.12，与我们 bootstrap 手册一致），但 **wheel 是 CUDA 编译物** | 镜像若有自带 tilelang 用自带的；否则按 bootstrap.md 手册装 HIP 版 0.1.12。**[VERIFY-26.10]** 基础镜像自带版本与编译栈 |

## 2. 直接装的依赖（平台中立，版本随上游声明）

第一批：**服务框架与调度**

```bash
pip install "aiohttp" "fastapi" "uvicorn>=0.51" "uvloop" "watchfiles" \
  "pyzmq>=25.1.2" "orjson" "msgspec" "setproctitle" "psutil" \
  "prometheus-client>=0.20.0" "packaging"
```

第二批：**模型/词表/结构化输出**

```bash
pip install "transformers==5.19.0" "tokenizers==0.23.2" "sentencepiece==0.2.1" \
  "tiktoken" "blobfile==3.0.0" "gguf" "compressed-tensors==0.18.0" \
  "outlines==0.1.11" "xgrammar==0.2.7" "llguidance>=1.7.6,<2.0.0" \
  "interegular" "partial_json_parser" "einops"
```

第三批：**推理客户端/生态胶水**

```bash
pip install "openai==2.8.0" "anthropic>=0.20.0" "mistral_common>=1.11.5" \
  "modelscope" "datasets" "huggingface-hub" "requests" "tqdm" \
  "openai-harmony==0.0.4"
```

第四批：**科学计算/媒体/杂项**

```bash
pip install "numpy" "scipy" "numba==0.65.1" "pillow" "soundfile==0.13.1" \
  "timm==1.0.25" "kernels>=0.14.1,<0.15" "regex" "pybase64" "pydantic" \
  "python-dateutil<3" "python-multipart" "distro" "easydict" "xxhash" \
  "zstandard" "ninja" "build" "IPython" "py-spy"
```

第五批：**多模态与条件依赖**（x86_64 Linux 上游声明为无条件，若镜像跑纯文本可裁剪）

```bash
pip install "smg-grpc-servicer>=0.9.0" "deepseek-recipe==0.2.0" "tml-renderers==0.1.0" \
  "torch_memory_saver>=0.0.9.post1"   # 需与镜像 torch 匹配编译版 [VERIFY-26.10]
```

备注（装前一眼灵魂三问，告警即拆）：
- `av`/`decord2`（视频栈）上游声明只在 aarch64 生效，x86 镜像可不装；VL 模型视频输入需要时再补；
- `nvidia-ml-py`：nvml 探针库，装了无害（纯 python 绑定）但我们的平台判定不走它——先不装，
  若容器首跑出现 nvml ImportError 再行处置 **[VERIFY-26.10]**；
- **py3.12 轮子覆盖度**：上表全部为主流 PyPI 包，py3.12 manylinux 轮子齐全；`numba==0.65.1`
  对 numpy 版本有上界约束，与镜像自带 numpy 冲突时以能 import 为准 **[VERIFY-26.10]**。

## 3. 我们的三件 wheel（装包顺序即配置）

```dockerfile
# ① adapter：shadow dist sgl-kernel + hcu_adapter + sitecustomize（一发四件）
COPY wheels/sgl_kernel-*.whl /wheels/
RUN pip install /wheels/sgl_kernel-*.whl

# ② 官方 sglang，零修改；--no-deps 是策略不是图省事（§0）
COPY wheels/sglang-0.5.22-py3-*.whl /wheels/
RUN pip install --no-deps /wheels/sglang-0.5.22-py3-*.whl

# ③ 插件（该镜像需要 plugin 时）
COPY wheels/sglang_hcu_plugin-*.whl /wheels/
RUN pip install --no-deps /wheels/sglang_hcu_plugin-*.whl
```

## 4. 已知接线点（官方原样树特有的，das fork 树没有的）

1. **内核版本探针** `engine.py:1954-1965`：`_is_cuda` 为 True 时
   `assert_pkg_version("sglang-kernel", "0.4.9")`——探 **dist 名** `sglang-kernel`，
   我们的 dist 是 `sgl-kernel` → `PackageNotFoundError → 启动即炸`。
   - **镜像层立即解**（三行之一）：
     `ENV SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`（上游内建的逃逸 hatch，
     engine.py:1955 判的就是它）；
   - **工程解**（与 governance §7.4 同一笔）：影子 dist 更名/加发为 `sglang-kernel`、
     版本基线 `0.5.22+hcu.N`——PEP440 下 `0.5.22+hcu.1 > 0.4.9`，探针自然通过且语义诚实。
2. **`pip check` 会报错**：sglang 的依赖声明显式缺失——`--no-deps` 态的固有现象，
   镜像健康检查用第 5 节导入自检替代 `pip check`，不修。
3. **das fork 的 srt 补丁面不在这棵树上**（本来就是要剥离的）：P1 挂账项在官方树上
   由 plugin 仓承接；首个新镜像系统全量回归按 models.md 四门重录基线，**das 树的绿色
   结论不自动平移**。

## 5. 安装后自检（pip 态五条，源码态四条见 bootstrap.md §7）

```bash
# ① sitecustomize 生效（解释器启动即接管）
python -c "import sys,os; assert os.environ.get('HCU_ADAPTER_HOST') is not None or True; \
  import sgl_kernel; print('shadow at', sgl_kernel.__file__)"
#    预期 路径含 site-packages/sgl_kernel/，且进程日志出现
#    "HCU adapter bootstrap: host=..."（bootstrap.py:177）

# ② 影子不被官方内核覆盖（同名 dist 只此一份、路径是我们打包的）
pip show sgl-kernel | grep -E "^(Version|Location)"

# ③ 插件可发现（sglang entry_points）
python -c "from importlib.metadata import entry_points; \
  print([e for e in entry_points(group='sglang.srt.plugins')])"

# ④ 平台语义自检（cuda 语义 + HIP 真值并存）
python -c "import torch; print(torch.version.hip is not None, torch.zeros(1,device='cuda').is_cuda)"

# ⑤ 服务冒烟（dummy 权重，几秒级；模型路径任意本地目录含 config）
python -m sglang.launch_server --model-path <path> --load-format dummy --tp 1 --port 30000
```

## 6. [VERIFY-26.10] 汇总：容器首跑要定案的 7 项

| # | 项 | 怎么验 | 定案后写回 |
|---|---|---|---|
| 1 | 自带 torch 版本 | `python -c "import torch; print(torch.__version__, torch.version.hip)"` | §1 torch 三件配套版本 |
| 2 | 自带 triton 版本 | `python -c "import triton; print(triton.__version__)"` | 红绿门复跑（gfx938 div-4 等已知内核问题在新 triton 上重判），结果记 vendor-kernels.md |
| 3 | 自带 tilelang 版本 | `python -c "import tilelang; print(tilelang.__version__)"` | 非 0.1.12 → 按 bootstrap.md §6.1 换装 0.1.12；本镜像的对应产物包名 |
| 4 | tvm-ffi 预热是否仍需手工 | `import apache_tvm_ffi` 后跑一次 dlpack 换路 | 需要则搬 bootstrap.md §3 手册；不需要则本文加"已内置"注记 |
| 5 | JIT 首编窗口 | 冷跑一次 L0 全量 + 一个 triton 档单测测首编时长 | 预热脚本与缓存种子（bootstrap.md §5）是否平移 |
| 6 | numba/numpy/py3.12 组合 | `python -c "import numba,numpy"` | §4 备注：版本三角最终钉值 |
| 7 | nvml 探针面 | 起服后 grep 日志 nvml/ImportError | §2 备注：装 nvidia-ml-py 或 adapter 层填空 |

## 7. 维护

- 依赖清单的**唯一底料**是上游 `python/pyproject.toml` 的 dependencies 数组；
  上游每次换基线（如 0.5.23），本文第 1~2 节按 diff 渐进更新，不重写。
- 首个 26.10 容器跑通后：第 6 节全部销项、示例按实测版本钉死、本文升格为正式
  部署手册并把 bootstrap.md 标注"旧基底（26041 止）"。
