# 完整质量评测与重放验收分别记录

当前状态：条件接入采用v7固定精度方案。主任务取得的真实RESULT SHA为 `15eae21a080a4cddec7e19ff660cd007f16e586fd4302c3a77104fc1f8242789`：float32对照失败，float64两帧的fresh、historical和全部lane通过。这是短诊断准入证据，不是9027帧或整体codec验收。本轮仅准备源码与CPU回归，正式冻结、部署、执行由主任务负责。

入口 `worker.py --root <本目录> --arm baseline|decoder_only|full_codec` 只评测指定模型。本模块导入时不加载Torch或MinkowskiEngine；没有训练、部署、等待循环、评测队列或自动模型替换。三个arm各自独立目录，不联动终止。

## 来源和加载

旧fullpool manifest固定 `01505377362bf47c89ae23aeaa5de6ca0c5663fe9119e35f6c01c03524351376`。只读复用其 `load_runtime`、完整50片段数据、tower14评测来源、`state_packet`、图像指标及原渲染器。旧预检只作为旧来源见证，不生成新预检。

baseline使用原codec3000检查点，SHA固定 `a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917`，additional_codec_updates=0。它按相同9027帧完整GPU推理路径fresh encode并重放，不借用旧CPU缓存指标冒充本轮结果；是否实际运行由新协议/队列另行决定。

decoder_only使用旧4426成功TRAIN_DONE与原加载门槛，检查点SHA必须为 `dcdcd0be5f999b09086bbc31c91aea62898600c123a962791845314898d6eef3`。旧评测失败不抹去训练成功，也不能被本轮改写。

full_codec必须来自独立重跑run，manifest固定 `479e75989ca7c5cde716dfc9d842a5f703a735837ac1a11b8faad15de638034b`。要求根DONE与TRAIN_DONE一致、无FAILED/FAILURE/失败权重、4426更新、两轮各8850覆盖、正确新旧source、完整epoch计划及最终检查点SHA；检查点还须标记failed=false、eligible_for_final_evaluation=true并匹配全部来源。尚无DONE时立即拒绝，等待由外部队列负责。三个模型加载后共用完全相同的评测路径。

## 评测路径与阈值

每个模型评测50个TRAIN片段加已暴露的tower14 EVAL-only，51×177=9027帧。每帧fresh encode，执行direct sparse decode→CPU state落盘→磁盘读取→重建。直接与重建两路均显式调用固定helper的 `prepare_sparse→ae.decode(target_key=None)→_world→to_render_points`：encoder与保存的Nx64 state保持float32，解码输入仅F变为float64且key/manager不变，world GS浮点属性转回float32交给原渲染器。GS键、点数和分辨率不变。两个view均比较原float32 `image_metrics`：MAE≤1e-5且PSNR≥60才通过。ROI指标保持原实现并保留，但不新增ROI阈值。

只有有限且结构正确的重放指标越过上述阈值时继续后续帧：`targets.jsonl`与`state_manifest.jsonl`保存原 `replay.real_view/nearby_view`全部指标，同时写 `replay_passed=false, failed_views`。也记录两路径GS数量与差值；数量差异不能归为渲染器容差已解决。NaN/Inf、异常结构、磁盘包与fresh state不一致、来源变化、显存超限及超时都立即FAILURE。

三臂统一采用协议固定的 `decoder_float64_default_me_independent_replay_v1`。ME始终False/SEPARATE/globalNone，不调用set/clear、不启用SHARE。每帧direct返回dense渲染和CPU clone包后释放其SparseTensor局部对象；保存/load同包，再创建独立manager重建。暂留旧manager对象引用验证新旧对象不同（避免Python id复用），创建新稀疏输入后立即释放旧引用，再执行decoder。全程no_grad并禁用CUDA autocast。

准入必须验证冻结v7 manifest `d45b63505b25bf3121dfad420200fbb42af3fd175eb20bab76654661d7b23261`、probe SHA `6f28cfbafa5ce75eec293cf08383edca6d16ce0bcc0c02d0fa758ad1cebf3dc4`、precision.py SHA `2b6d4513afed3bc1d5f6d31d4a076a341c49a23ba500a49f257b65319609d7e8`、passing CPU和protocol.inference中真实RESULT SHA。要求float64两帧fresh_same_state通过原阈值，精度转换报告castback_exact/encoder_exact/nonselected_state_exact全部通过，运行state_verification.unchanged通过。float32对照不要求通过；历史不同输入比较单列。诊断挂剪枝观察钩子而完整评测不挂，不能冒充同条件重复；完整评测自己检查9027帧独立manager重放。

只转换ae.decoder、ae.feature_heads和ae.gaussian_multiplier。转换前记录legacy算法的checkpoint_model_state_sha256与原encoder/decoder摘要，转换后记录运行model_state_sha256/decoder_sha256，并直接验证encoder摘要未变。baseline原始身份用转换前摘要对照旧initial_model_sha。helper另有覆盖全部参数和注册buffer（包括非持久buffer）的摘要，其编码算法不同，不与legacy摘要混用；最终assert_unchanged验证该运行状态未被推理改变。SOURCE、packet metadata、targets和EVALUATION绑定各自conversion报告与三种dtype。

固定所有片段的0/44/88/132/176帧保存real、teacher、codec、nearby、nearby_teacher五类截图。保留原分辨率、全点、归一化、pts、源cache SHA、checkpoint/encoder SHA及原state_path语义，另增decoder SHA、`evaluation_source`与`training_source`，旧 `source`仍表示原数据/实现绑定。固定inference_profile在state保存前写入metadata；两作用域实际恢复记录只在退出后写targets/state_manifest，绝不修改已经落盘的packet metadata。`cold_manager_replay_verified`仅表示实际完成独立manager重建；图像重放是否通过由独立replay字段决定。

## 产物与资格

产物位于 `evaluations/<arm>/`，目录独占创建且不覆盖重试。完成9027帧后写 `EVALUATION.json`：`status=evaluated, quality_evaluation_complete=true` 表示质量数据完整；另有 `replay_passed_all`、`replay_acceptance`、失败帧数量/清单。任一阈值失败必为 `replay_acceptance.status=failed`；`every_packet_saved_replay_passed`不得写成true。

即便重放全通过，本worker也保持 `codec_accepted=false, codec_acceptance_status=pending_separate_review`，整体codec质量/动作Z验收由后续独立审查决定。失败写 `FAILURE.json`，标记quality_evaluation_complete=false和codec_accepted=false；不把部分评测做成成功收据。

CPU6线程，7200秒alarm包含加载与评测，allocator限10GiB；Torch峰值及约每10秒本进程显存采样上限12GiB。采样不是连续峰值。现场设备分配、空闲显存/磁盘准入和外层timeout由队列负责；worker仍要求实际可见CUDA设备恰好一个，且不控制其他进程。

## protocol.json与CPU测试

`schema=codec_complete_evaluation_v1`；`source`字段为old_root、old_manifest_sha256、decoder_checkpoint_sha256、full_retrain_root、full_retrain_manifest_sha256。`evaluation`冻结frames_per_model=9027、frames_per_clip=177、snapshot_frames=[0,44,88,132,176]、max_seconds=7200、replay_mae_max=1e-5、replay_psnr_min=60、continue_on_replay_threshold_failure=true。`resources.cpu_threads=6, peak_limit_gib=12`。

新manifest需绑定所有根级Python文件及protocol；CPU_VALIDATION必须passed、gpu_executed=false并绑定准确新manifest。`python -B -m unittest -v test_worker.py` 使用真实纯CPU数据验证边界阈值、非有限值失败、完整覆盖及重放失败不可隐藏、未来full DONE/检查点资格，不模拟CUDA或ME执行。

`test_inference_profile.py`导入精确冻结v7 helper但不加载Torch/ME，使用小型CPU对象检查独立manager身份/释放时机、默认global状态不被改动、异常记录、精度准入与两套摘要区分。它不是CUDA/ME可重复性见证。RESULT SHA为null或真实float64结果不合格时，pipeline在创建任何worker之前即拒绝，worker自身也在import Torch之前复核。
