# 完整池 codec worker 接口与证据范围

`worker.py --root <本轮目录> --mode preflight|train|evaluate|baseline`。仅 train/evaluate 必须指定 `--arm decoder_only|full_codec`。preflight 在一个进程内执行两臂、各两次完整 B4 丢弃更新；baseline 为原始 3000 步权重的完整 9027 帧评测。正式入口、GPU 分配、磁盘准入、总队列与外层超时由 root 的 pipeline 管理。

本模块通过 SHA256 校验的旧 worker 复用模型加载、训练模式/冻结检查、AdamW、梯度检查、latent state packet、编码与双视角落盘重放、图像指标和 48 GiB 资源检查。旧 worker SHA 为 `f210f2f0961d87ed0f4e9d448b54b34118da00e9c958ae32ac1fb70a6335e3f7`，旧 sources SHA 为 `7e8eaa327d1c05e7d00fa07121832c1f66d617c6e76a016eb711f1453606b249`。模型和 renderer 源码没有复制或改写。

运行时要求旧 multisource 的 DATA_BINDING.json 已存在且 SHA 为 `695959d1dce88089293ce795d8a962ecb284654fe530b0e8f286c61e73d02715`，然后调用旧只读验证/加载路径。旧 binding 已存在的前置条件防止该旧 loader 在旧目录创建新绑定。旧 bundle 只供原模型/renderer/损失依赖和 tower14 评测；新 50 段 TRAIN 全部由 pool_index 的 FrameRef 逐帧加载。不同片段保留各自已绑定 normalization，没有全池 scale=0.5 假设。

两臂都从原始 codec3000 checkpoint `a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917` 开始，丢弃旧 optimizer。两完整 epoch 的种子分别 20261031、20261032；每 epoch 8850 次完整帧曝光、2213 次更新，最后一批两帧，按两帧平均。第一完整 epoch 使用目标结构强制，第二完整 epoch 不使用；共 17700 帧曝光、4426 次更新。B4 是顺序逐帧反向的逻辑有效 batch，不是把四个稀疏场景拼成物理 batch。

必须冻结两个预定 B4 压力批次，通过新 protocol.preflight.batches 提供。两批使用相同四个 TRAIN 帧：tower17_176:117（全池最大输入）、tower18_352:151（最大 latent/candidates/GS）、tower7_0:135（唯一 scale=0.25 片段的候选/GS 最大帧）、tower0_176:166（候选第二高）。每批内部四帧必须不同，允许跨批重复。每臂连续执行 force=True、force=False 各一次丢弃更新，让两种模式均覆盖同样的输入极值和尺度情况；这是资源与梯度冒烟检查，不是隔离的 forcing 效果比较。

预检检查同初值双视角一致、两种 forcing、两臂所有必需参数组非零有限梯度、full encoder 更新及 decoder-only encoder 参数和 buffer 冻结。实际 GPU 前反向和新场景峰值尚未执行，CPU 通过不替代这些证据。

正式训练保留 step_2213.pt、step_4426.pt，只有 4426 可评测。每臂训练预算 7200 秒；初始化/数据 SHA 准入还需 pipeline 的外层有界时间。失败保留 trace、FAILURE.json，能够保存时另保存 FAILED_DO_NOT_EVALUATE.pt，不自动续训。protocol 的固定字段由 validate_protocol 显式核对。

评测全部 50 TRAIN 的 8850 帧加此前已暴露、梯度排除的 tower14 177 帧。每帧都重新 encode，保存连续 64 维 state packet，再从磁盘加载这个 packet、通过原 decode_state 重放两个视角；阈值保留原实现 MAE≤1e-5、PSNR≥60。**这是 latent state 的落盘重放，不是脱离模型的 native GS 包独立重放。** 收据明确 saved_latent_state_replay=true、independent_native_gs_replay=false。不使用旧 state 缓存作为新 encoder 的输出，不按 tower14 挑选 checkpoint，也不声明独立 holdout 泛化。

每个评测模型输出 targets.jsonl、state_manifest.jsonl、screenshot_manifest.jsonl、EVALUATION.json。按 clip 汇总，并单列 TRAIN 与 exposed EVAL-only 均值；来源组均值可从逐帧行另算。每段固定截图帧 0/44/88/132/176，五种图各一张，共 1275 张；没有按效果挑帧。数值改善或 codec 成功不等同于动作 Z 学成。

首次本地 CPU 数值测试 9/9 通过，工具 session 55529、终态 chunk a2c005；增加失败权重保留逻辑后最终复查仍为 9/9、零跳过（chunk e4ab1c）。覆盖两 epoch 无放回与独立尾批、forcing 精确边界、B4/尾 B2 的真实梯度均值、encoder 冻结与全模型更新、量化器约束、非法批次及非有限损失。仍需最终冻结后的 fresh CPU 文档见证及真实 GPU 预检。此阶段未 SSH、未运行训练或 GPU。

压力批次修订后，新增“允许跨批重复、拒绝批内重复和评测来源”检查；完整本地 worker 测试 10/10、零跳过（chunk 564009，2.454 秒）。
