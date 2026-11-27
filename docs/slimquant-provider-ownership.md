# SlimQuant注册归属

当SGLang按公开插件配置选择已安装的`hcu_sglang:register`入口时，adapter跳过legacy SlimQuant的完整activate步骤，包括权重加载shim与量化注册，由plugin提供该格式支持。部署无需显式设置`HCU_SLIMQUANT=0`。

判断遵循`SGLANG_PLUGINS`逗号分隔白名单和`SGLANG_PLATFORM`对未选中平台发行包的排除规则；不导入plugin或提前执行其注册入口。空白名单按SGLang默认发现规则处理。未安装或未选择该plugin时保留adapter原有实现。`HCU_SLIMQUANT=0`仍作为手动关闭legacy实现的调试选项；默认值仍为1，选中plugin时该默认值不再造成重复注册。

不将plugin入口加载失败解释为可安全回退到legacy；两者的权重处理及MoE路径不同。此变更不替换底层GEMM kernel，不改变plugin的SlimQuant实现。

验证范围：独立CPU测试覆盖入口发现、白名单、平台排除、plugin路径不执行legacy activate、adapter单独使用时执行legacy activate。真实模型启动、生成精度及性能必须分别验证，不由CPU测试推断。


## Vision single-patch convolution diagnostic compatibility

The separate kernels/conv2d_projection_emu.py provides single_patch_conv2d for one spatial patch, zero padding, unit dilation, groups=1, stride=kernel and zeros padding mode. It preserves the Conv2d weight/bias and uses flattened linear projection. All other geometry calls the supplied Conv2d object. The GLM5-Next plugin invokes it only on HCU. This addresses a recorded old-DTK MIOpen hipFree wait during multimodal warmup; it is not a global torch patch and is unrelated to SlimQuant provider selection. CPU/HCU/Graph component tests passed, while full image-model and OCRBench validation remain pending. Catalog/promotion and broader release gates are pending; this is an uncommitted diagnostic candidate.

The 2026-11-27 single-patch visual convolution diagnostic candidate did not resolve model warmup. After correcting private plugin visual parallel-group ownership, native Conv2d completed warmup and three real image requests. The unused projection helper was archived in the task evidence and removed from adapter runtime; it is not a promoted adapter operator.


## Companion plugin and DeepEP dispatch ABI

The companion plugin branch is suiyizhe2/sglang-plugin-das:
feat/glm53-flash-int4-fixes-20261127. The validated combination starts from
adapter 00eaf494, plugin 43bb7b8c and official SGLang baa9befa (0.5.22).

The DeepEP wrapper preserves vendor calls carrying quant_type or the positional
vendor ABI, and retains upstream use_fp8 dispatch translation for upstream calls.
Vendor quant_type combined with use_fp8=True is rejected instead of silently
changing the quantization contract. The wrapper also preserves the caller's
legacy topk_weights handoff needed by the existing upstream compatibility path.
No communication kernel implementation is replaced here.

Targeted dual-ABI and provider selection CPU tests pass, as does the L0 gate
(with 32 warnings). The tested old-image combination completed GPQA 172/198
and OCRBench 895/1000; dataset protocols, tokenizer concurrency handling and
precision limits are described in the companion plugin development record.
These results do not certify arbitrary newer adapter/plugin heads.
