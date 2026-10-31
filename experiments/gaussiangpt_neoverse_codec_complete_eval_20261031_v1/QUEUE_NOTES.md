# 队列与CPU比较实现说明

本目录当前只有待审核源码。此实现没有部署、启动评测或向现有训练进程发送信号。三臂按同一v7 profile使用float64 decoder/head/multiplier，encoder/state与原渲染器输入保持float32；ME始终False/SEPARATE/globalNone，独立manager直接/重建，无全局切换。

v7真实RESULT SHA `15eae21a080a4cddec7e19ff660cd007f16e586fd4302c3a77104fc1f8242789` 的float64两帧fresh/historical全部通过，float32仍失败。pipeline新增CPU-only准入，必须在创建queue或worker之前验证冻结v7源码、precision helper、passing CPU、实际RESULT及float64转换/运行state/fresh重放；两帧通过不表示完整评测或总codec验收。

## 队列

- 入口为 `pipeline.py --root <dedicated-root>`。GPU0依次执行baseline和decoder_only，GPU1等待独立full_codec训练成功后执行full_codec评测。
- 等待固定外部PID 479060、start_ticks 40929788及精确worker/root命令。PID被复用、命令改变、失败收据、进程退出但无成功DONE/TRAIN_DONE均拒绝full_codec评测。成功收据与checkpoint绑定再次核验；外部进程退出后才准入GPU1。
- 外部训练只读观察，永不登记到owned-child registry，永不向外部PID或进程组发信号。仅加载并复用SHA固定的 `postfit_queue.py` 的 `run_child` / `terminate_queue`。
- 单臂异常记录到该臂OUTCOME，继续其他臂；baseline失败后仍尝试decoder_only。全局信号或20000秒总时限只终止队列自建子进程组。
- 每个评测进程8500秒，含其初始化；worker自身还保留7200秒内部保护。训练等待8500秒，CPU比较1800秒；每个子任务时限与剩余总预算取较小值。无重试、不减帧、不降分辨率、不改点数。
- 每次评测启动前测量指定GPU实际空闲显存至少16GiB及输出盘空闲至少100GiB。每worker设置6个CPU线程，12GiB显存保护由worker实现。CPU比较禁用CUDA可见设备。
- 仅三臂子进程退出成功且各自有完整9027帧EVALUATION后运行比较。任何臂缺失都不执行完整比较，输出PARTIAL或FAILED；三臂评测完成但CPU比较失败也为PARTIAL。

## 比较

- 入口为 `compare.py --root <dedicated-root>`。不导入Torch/MinkowskiEngine、不反序列化checkpoint/state tensor。
- 先校验新评测代码manifest和精确CPU验证证据。复用旧固定manifest下CPU比较器的来源、数据池、采样计划、decoder训练记录和基础哈希函数；不调用其旧重放门槛、旧baseline复用逻辑或bootstrap比较。
- 验证原baseline为新增step0，decoder_only为旧固定4426 checkpoint，full_codec为独立重训成功DONE绑定的4426 checkpoint。复核完整训练trace与采样计划；baseline/decoder encoder digest必须匹配旧固定证据，full_codec encoder必须已变化。full checkpoint内部tensor的源码绑定校验由worker在GPU评测前完成，CPU比较不冒充再次反序列化校验。
- 每臂严格要求9027个唯一帧，覆盖50个TRAIN片段共8850帧及已暴露tower14的177帧。校验PTS、split、source_group、归一化、source cache SHA、checkpoint/encoder/decoder、新旧训练来源及新评测来源。
- 验证EVALUATION中targets/state_manifest/screenshot_manifest三个JSONL SHA；逐9027个实际state和1275个实际PNG验证bytes、SHA、路径，以及PNG 448×252尺寸。拒绝外部路径及符号链接。
- 严格验证三臂SOURCE/EVALUATION/targets/state_manifest采用同一profile与诊断来源；纯CPU逐文件验证实际ME四源码SHA。每帧direct/replay前后必须None/False/SEPARATE、没有global变更、两个实际manager对象身份不同；隔离证据不能替代图像重放通过。
- 验证各模型conversion报告的castback_exact/encoder_exact/nonselected_state_exact、float64选定模块及float32其余状态，最终运行state未变。baseline按转换前checkpoint_model_state_sha256核验，运行精度摘要另列，避免将dtype变化当权重改写；helper与legacy两套摘要不混用。
- v7诊断带剪枝观察钩子，完整评测不挂。历史不同输入比较与float32对照保留但不作为float64 fresh准入要求，也不能将v7完成冒充9027帧完整重放验收。
- 指标保持完整：codec-real、teacher-real、codec-teacher、伪视角-teacher，各自全图和ROI MAE/PSNR。按每片段、50 TRAIN片段等权、19来源组等权、每来源组及已暴露14分别报告。三种成对模型差值均为参考减候选。
- 双视角纯重放仍使用原阈值MAE≤1e-5且PSNR≥60。超过阈值的完整评测仍能出描述报告，每个失败帧及其视角、数值、PTS、解码点数差明确列出；伪造通过标志被拒绝。
- 不计算任何置信区间，不做自动质量gate，不据此判定动作Z或LDM成功。`quality_evaluation_complete`、`replay_passed_all`和`codec_accepted`是三个独立字段；最后一项始终为false，等待单独视觉与重放审查。

## CPU检查

`python -B -m unittest -v test_queue_compare` 使用小型临时文件和少量帧检查身份变化、等待截止、失败训练资格、资源准入、部分结果、配对和分组均值；不构造9027份虚假state或GPU见证。最终源码及全部test_*.py由主任务的validate.py另行冻结验证。
