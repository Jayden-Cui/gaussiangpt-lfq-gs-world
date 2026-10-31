# 完整50片段连续GS状态codec训练

两来源codec实验已证明扩展来源可改善训练场景，但暴露的tower14仍有明显形状、颜色错误。当前动作Z也未通过黄条抬升验收。本阶段处理跨来源状态编解码基础：完整50段训练数据已于09:40完成，先训练连续状态codec并重新编码，再由完整结果决定接回LDM。它不训练动作QFormer/LFQ，不自动替换正在运行的LDM，更不把codec得分改善当作动作Z成功。

## 数据和训练范围

精确输入为完整池DONE SHA `920a7f7c030733b2ff6d1be62aafb243bf581ea0a9bcc53ba3fe775b3d0f694b`、POOL_COMPLETE SHA `8ffe7055cba065a49c233245b106631665dcd710b3ebee2fd08368ab543224df`。50段/19来源组，每段177相邻原帧，448×252，112896源GS/帧，共8850片段内帧。跨片段物理帧去重数未知。另加已暴露的tower14仅评测177帧，禁止进入训练或用其选择checkpoint。

每次只载入一个完整帧，按原缓存字节先验SHA再Torch解包，不把49,191,686,647 bytes缓存整体驻留。新适配器保留各片段已绑定的第一帧归一化：49段scale0.5，tower7_0为0.25；不新增全池归一化。详细接口见DATA_ADAPTER.md。

两臂从同一原codec3000 checkpoint `a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917` 初始化、使用新的AdamW。decoder_only训练原decoder/feature_heads/gaussian_multiplier，encoder参数及buffer保持冻结；full_codec解冻完整原codec。保留64维连续状态、Identity状态量化、grid0.008/stride4、原完整点数上限和原六项损失权重；不新增状态LFQ。

有效B4通过完整帧顺序反向累积实现，按本批实际帧数平均，不拼接不同场景。两遍各完整8850帧无放回覆盖：第一遍目标结构支持开启，第二遍关闭；每遍2213次更新，最后一批2帧，共4426次更新、17700次帧曝光/臂。两个固定种子20261031/20261032。保存2213和4426，仅最终4426可进入正式评测，不选best或用暴露eval来源挑选模型。

原六项损失为real L1=1、VGG19=0.1、teacher L1=0.5、nearby teacher L1=0.25、workspace ROI L1=0.25、occupancy BCE=1。保持原实现，不把ROI矩形称作物体分割。此处完整codec训练的损失和当前v10b动作模型的新增监督对照是两个独立实验。

## 基线与压力预检

CPU基线复用49段原冻结codec准备时的float32重建，补齐当前所有图像指标；tower17及tower14从旧完整baseline的各177行取得。所有引用仍核确切文件字节SHA，不运行重复8850帧GPU baseline。两条指标路径仅在明确浮点容差内一致，不能称位级相同。49段准备只验证保存张量内容，未将磁盘重载state再渲染，因此基线整体saved_latent_state_replay=false；只有17/14共354帧继承真实历史重放证据，不能伪造全量重放通过。

预检同一组四帧各执行force=True、False两次完整B4更新，两臂均从同一初值独立开始；每臂内部两次更新连续、最终全部丢弃。这是资源/梯度检查，不是隔离的forcing效果比较。四帧由全池已有资源规模字段预先确定：tower17_176:117（输入体素最多）、tower18_352:151（候选、latent及GS最多）、tower7_0:135（唯一0.25尺度片段的候选最大）、tower0_176:166（另一高候选片段）。固定点集、分辨率和每帧全部输入不减少。

需要实际CUDA证明双臂初态双视角一致、64维state往返、有限非零梯度、指定参数组更新、decoder_only的encoder参数和buffer不变。CPU绿色测试不能代替此项；预检失败不得进入正式训练，失败后不自动重试或降配。

## 环境、冻结与新鲜CPU文档见证

134温启动复用 `/data/CJH/envs/gaussiangpt/bin/python`，环境spec SHA `5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f`。既有ledger位于 `/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_vw1_alignment_20261030_v5/.aris/compute/ssh-134.md`。不安装/升级包，不改旧模型或旧数据源码。原生算子GPU证据来自既有真实训练；新场景资源须由本轮预检测量。

全部实现和本文完成后，将根级全部.py、protocol.json及本轮说明.md的SHA冻结为CODE_MANIFEST.json。生成的SUMMARY.md/结果/回执不属于源码清单，不得在完成验收时误报为新增未绑定源文件。冻结后由没有参与实现的新agent逐字执行一次：

```powershell
C:\Python314\python.exe -B E:\MoViesGS\experiments\gaussiangpt_neoverse_fullpool_codec_20261031_v1\validate.py --root E:\MoViesGS\experiments\gaussiangpt_neoverse_fullpool_codec_20261031_v1
```

命令只做CPU合成测试，最多900秒，写CPU_VALIDATION.json。要求实际测试数大于零、全部通过、零skip、gpu_executed=false、manifest/protocol SHA准确。若工具返回session，只接续同一session，不重新执行命令；失败不自动修复重试。见证者保留原始输出、退出码与工具标识。CPU证据只准入下一真实GPU预检，不代表本轮训练或动作Z完成。

## 唯一有界队列

入口一次核两张分配GPU实际空闲各至少55GiB、当前磁盘至少80GiB。worker保持48GiB allocated/reserved上限，CPU每进程6线程。并行执行GPU0预检和CPU基线，二者完整成功后，GPU0训练decoder_only再评测，GPU1训练full_codec再评测。最后CPU比较所有结果。共享服务器上不停止、接管或修改其他训练任务。

阶段上限：预检内部900秒/进程2130秒，基线CPU计算3600秒/进程4830秒，两臂各训练内部7200秒/进程8430秒、评测内部7200秒/进程8430秒；外层含每次初始化/数据准入1200秒。比较1800秒，总队列24600秒。OOM、文件变化、非有限值、重放失败或超时都保留FAILED和可保存的失败权重，不能评测失败checkpoint、续训、减点/帧/像素或自动重试。

同源manifest与CPU证明部署到远端全新同名目录后，仅提交一次独立screen：

```bash
env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=20s 24620 /data/CJH/envs/gaussiangpt/bin/python -B /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fullpool_codec_20261031_v1/pipeline.py --root /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fullpool_codec_20261031_v1
```

日志写新root/launch.log；启动只确认一次准确PID/start_ticks/命令与queue/RECEIPT。程序等待阶段，模型不反复tail/ps/nvidia-smi盯训练；使用一个有界observer等待READY_FOR_TRAINING及最终DONE/FAILED，接续原handle，不因观察超时重启。终态应有7个正常子阶段记录。

## 验收边界

每个新训练臂完整评测9027帧，每帧fresh encode连续64维state、保存并从磁盘加载、再经原codec双视角解码重放，要求MAE≤1e-5、PSNR≥60。这是latent state重放，不能称脱离模型的native GS独立重放。全量新state绑定相应checkpoint和encoder SHA；以后LDM只能用新编码重新建立特征统计，不能沿用旧encoder的states或统计。

每段固定0/44/88/132/176截图，两新臂各1275张，不挑最好帧。比较每段、TRAIN片段等权、19来源组等权与暴露tower14分开，不能把有重叠的50片段当独立样本，也不把训练数据上的改善外推为独立泛化。数值结果之后仍需检查固定帧物体位置、颜色、结构和空洞。这里只训练状态codec；动作Z是否可靠仍须后续完整LDM差分训练、正确/打乱/固定Z对照和动作视觉验收，goal_complete始终false。
