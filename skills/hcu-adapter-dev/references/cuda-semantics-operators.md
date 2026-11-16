# CUDA 平台语义下的算子兼容

适用于 HCU/其他 HIP 设备通过 adapter 进入 SGLang CUDA 路径，但底层库、算子实现或资源要求不适用的情况。CUDA 平台语义决定上层分派，不代表硬件变成 NVIDIA，也不把实际 HIP Tensor 改成 CUDA Tensor。PyTorch HIP 仍使用 `torch.cuda` API 和 `CUDA` dispatch key；这是框架约定，不是设备身份伪装。

## 先辨认实际缺口

沿模型 caller → forward_cuda → 包装/注册 → 数值实现核对实际导入路径与版本。

- **符号缺失或响亮 stub**：import 成功、callable、hasattr 都不能证明有计算实现。确认模型确实调用及其 ABI，再查厂商包/已有算子。
- **接口已存在、实现不支持硬件**：例如 CuTeDSL/TVM FFI 要求 CUDA Tensor，却收到 HIP Tensor。这是实现后端不兼容；不应改 Tensor 身份或随意改平台分支。
- **接口漂移**：签名、可选参数、输出格式错误，先对比固定版本及当前上游的调用方与接收方，按主 skill 的接口排查规则处理。
- **资源超限**：例如 Triton 每 CTA shared memory 超过设备 LDS 上限；区别于全卡显存 OOM。先查可用后端或合法 launch 配置，多卡不会增加单个 CTA 的 LDS。
- **依赖或二进制问题**：包缺失、ABI/动态链接或错误构建先处理环境，不凭模块名判断 CUDA/HIP 实现。厂商库可保留 `*_cuda` 的符号名，需检查实际链接和执行。

## 选择实现与接线位置

遵守项目的 adapter/plugin 分工及用户授权。社区已有行为的兼容实现归 adapter；新增模型策略或框架没有的能力按治理归 plugin。保持模型类、接口、配置和原有 forward 路线；确需改变结构时仍须先说明并获同意。

先查语义相符的现成实现：lightop、aiter、厂商 FlashAttention 等，再查社区其他平台的 Triton 路线。不要仅因名称相似就接入；确认精度、布局、广播、alias/mutation、返回值、stream/Graph 和依赖。社区 Triton 可移植不等于原样逐位一致；核对 wave、归约、FMA 和中间舍入，保留来源与差异。没有匹配实现时再评估新 Triton/HIP kernel。

按真实调用边界选窄 seam：

1. **已注册 Torch custom op**：优先替换该 op 的目标设备实现。例如 `torch.library.Library(namespace, "IMPL", "CUDA").impl(op_name, implementation)` 可保留 CustomOpDef、schema、fake/meta 和早先缓存的函数别名。保留 Library 对象生命周期；验证注册与 schema 定义的先后顺序、重复 install、其他设备实现及 spawn worker。不要注册新 schema 覆盖原 op，不改 mutates_args 来迎合错误实现。
2. **外部库 ABI/影子库**：在现有影子符号绑定处接到算子实现。数学放算子层，桥只做绑定；缺少实现继续明确报错，不能用 zeros/空结果伪造成功。
3. **现有 attention/backend 选择器**：在已授权的平台选择 seam 接入现成后端，保留其他请求的 original。使用框架正式插件生命周期及实际 registry；SRT 注册不等于 diffusion worker 已生效。Hook classmethod 时检查方法实际定义类，挂到继承子类可能丢失 descriptor；覆盖类和实例调用。
4. **普通函数直调**：优先正式 registry；否则采用项目允许的窄运行时绑定并覆盖预导入别名。仅修改 module 属性可能遗漏消费者，不能靠社区源码写回来掩盖。

CuTeDSL 融合 norm 的替代需单独审计 residual、gate、weight/bias、scale/shift。返回 `(normalized, residual_out)` 的无输入变更接口，不能套用原地 RMSNorm。公式相同不保证相同数值：确认 residual 保存的 dtype、norm 实际读取的精度、归一化与仿射后是否低精度舍入、modulation 类型提升；独立 reference 明确这些阶段，无法执行原实现时如实记录对齐边界，不能宣称原 CUDA 逐位一致。

## 验证与收尾

先建真实消费者反例，再测实现与安装。覆盖生产 shape/dtype、独立广播、空/非连续输入、None/整数 gate 转换、返回顺序、输入修改和输出别名；按原契约验证数值，保留 exact gate，不把容差测试称为 exact。运行 warmup 后同地址更换输入的 Graph replay，与对应 eager 比较；拦截旧后端证明新路线实际命中。安装测试保留真实 custom op/选择器/forward，不 mock 掉接入边界。

之后回到固定模型请求：保存完整日志、输入和 job 最终状态，确认未 native fallback。一个算子通过仅代表越过该点，后续融合算子失败单独记录；生成质量与性能需要各自证据。清理本次服务并更新项目记录、算子 catalog 与测试范围。

具体包构建、版本、硬件、请求、耗时和源码快照放模型项目记录，避免把单机经验写成 skill 的通用支持承诺。
