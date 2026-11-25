# 打包与版本号怎么读（release 线双 dist）

> 只讲**当前 release 线**的产物形态与版本语义；门禁分层见 [ci.md](ci.md)，
> 部署/配置见 [config.md](config.md) 与 [env-vars.md](env-vars.md)。

## 两条产物

| wheel | 内容 | 版本单一来源 |
|---|---|---|
| `sgl_kernel-<public>+<platform>[.<stamp>]-<py>-linux_x86_64.whl` | native 内核（`setup_hip.py` 编译 `csrc/`，`import sgl_kernel` 的 AOT 后端） | `sgl_kernel/version.py` |
| `hcu_adapter-<public>+<platform>[.<stamp>]-py3-none-any.whl` | 纯 python 适配层：`hcu_adapter/`（含内置启动配方 `launch_configs/*.yaml`）、`sitecustomize.py`、vllm entry-points | `packaging/hcu-adapter/VERSION` |

一条命令出两张：`bash packaging/build.sh`（`--no-stamp` 去掉时间戳后缀）。
两张共用同一个平台标签和同一个构建戳，adapter 的 `Requires-Dist` 精确互锁同戳内核
（`sgl-kernel==<内核版本>`），所以**必须成对安装**，不能混用不同次构建的产物。

## 版本号怎么读

格式统一为 `<public>+<platform>[.<stamp>]`（PEP 440 本地版本段）：

- `<public>`
  - 内核：`0.1.0`，内部内核线版本，和上游 sglang 版本无关；
  - adapter：配对的上游 sglang release（当前 `0.5.22`）。**带平台标签的构建里
    `VERSION` 文件的 `+hcu.N` 增量段会被平台标签替换**，只有不带的开发构建
    （直接 `pip wheel packaging/hcu-adapter`）才会看到 `0.5.22+hcu.1`。
- `<platform>`：`dtk<DTK>.torch<TORCH>`（例：`dtk2610.torch2130` = DTK 26.10 /
  torch 2.13.0）。这是编译期 ABI 标签，**部署侧按它对齐容器**：容器里的 DTK 与
  torch 必须是这一段，否则扩展加载/数值都不保证。
- `.<stamp>`：`<YYMMDDHHMM>.g<commit6>`（构建时刻 + 源码提交号），只在
  `build.sh` 默认路径出现，用于把包精确追溯回某次构建。

对应关系一句话：**adapter 的 public 对齐上游 sglang release，内核的 public 走内部
内核线；两张的 `dtk/torch` 段必须与部署容器一致，`stamp` 必须两张相同。**

## 包内数据文件

拆分后每个 dist 只带自己的数据，且都**显式声明**在各自的
`[tool.setuptools.package-data]` 里：

- `hcu_adapter/launch_configs/*.yaml`：`HCU_CONFIG=builtin:<name>` 的运行时配方
  （见 [env-vars.md](env-vars.md)），由 `packaging/hcu-adapter/pyproject.toml` 声明；
- `sgl_kernel/diffusion_rope_csrc/*.cuh`：JIT 编译用的头文件，由根
  `pyproject.toml` 声明。

不能依赖 `include-package-data` 自动发现：adapter 的 `package-dir` 指向仓根
（在构建根之外），自动扫描看不到这些文件，漏声明时 wheel 会**静默**少文件，
`builtin:` 配方在运行期才报 `Missing built-in launch config`。

契约由 `tests/adapter/test_packaging.py` 兜底（`pytest tests/adapter/test_packaging.py`）。
