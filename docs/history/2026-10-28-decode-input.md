# 2026-10-28 Decode 输入融合（历史组合记录）

> 以下记录来自迁入前的 adapter + plugin 组合，保留用于追溯实现来源；本分支在新 adapter-das 基线上的运行、精度与性能尚未验收。

基于DeepInfer 774ce244，共用adapter c2f659f及模型plugin v53，冻结组合记为v54。官方源码、原生扩展和vendor未修改；工作区原有HCU_CONFIG修复保留。

## 实现与归属

五个原有tilelang出口由bridge选择`zos_metadata_triton`或原Torch参考。Triton路径直接写入输出，合并位置/上下文/缓存索引等逐元素运算；不读取GPU标量到CPU。框架签名保留在bridge，张量实现不依赖DeepInfer。公共registry、catalog及UI快照同步。

CPU metadata打包上传属于独立模型plugin的执行策略，不在本adapter改变调度器或宿主源码。

## 验证

- N=1/3/8/48/65，nextn=0/5，五个出口合计50组；与Torch参考整数逐项相等，每组3次换输入Graph回放一致。覆盖变长、padding、非紧密输出stride、压缩边界、Engram sentinel。
- CP8EP8和DP8EP8，112道独立题各109正确；16重复提取答案一致，与SGLang参考提取答案等价。随机采样回退与target/draft Graph均实际执行。逐token不完全相同，不将答案等价扩写为数值逐位一致。
- 相同DP8EP8 decode prof，rank0五步：prepare_input每步189次kernel启动降至4次；原8次H2D+3次D2D变为8次H2D、0次D2D。新路径4个kernel为metadata、blocks、ids、engram；seqlens修正在外层另行执行。
- profiler区间耗时与未采样服务延迟分别记录，不相互替代。原始证据目录：`prof/deepinfer-input-acceptance-20261028`，含metadata-qualification、逐题结果、分阶段8卡CPU-stack trace及归因脚本。

未新增单测；按用户要求执行实际CI的`python3 ci/l0_gate.py`、额外strict模式与`git diff --check`。未推送或修改其他工作树。

## v56 上下文增长补充

metadata和blocks的块表长度NB改为运行时参数，并显式禁用该参数特化。受控增加8种块表长度，旧版两个kernel各新增7个编译条目，新版新增0；整数对照及50组Graph数值探针全部通过。该改动处理持续增长触发的JIT停顿，端到端重采集结果见下方v56记录。

组合模型plugin的批量上传后，v55 trace实际为4 kernel、1 H2D、0 D2D。CP/DP完整回归v56继续各109/112，16重复提取答案一致；具体token变化与原始分段耗时保留在任务证据目录。

## v56 精度门禁之后的性能复验

prefill CP8EP8 global8、32768输入/1输出，三次6.418530/6.430302/6.424631秒，中位6.424631秒（v53 6.427123；SG6.514072）。decode DP8EP8 global48、65536上下文/512输出预算，三段75.075847/73.919890/73.922192ms/iteration，中位73.922192ms（v53 74.931436；SG74.789800）。有效每输出token14.784438ms（v53 14.986287；SG14.957960）。相对原版decode中位降低1.35%；只有该固定负载的证据，不扩写到所有批次/长度。

第三段旧版JIT尖峰在v56消失。三段稳态接受率均80%。prefill8条、decode48条性能输出均与v53逐token相同；与SG的对应关系沿用已核验的固定输入/输出基线。每rank、每窗口原始值见performance-comparison.json。

v56八卡40个prepare_input区间全部4 kernels、1 H2D、0 D2D。完整CPU调用栈trace分prefill/decode保存；查看版另行验证。
