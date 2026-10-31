# 原 codec 基线 CPU 复用合同

本实现只复用原3000步连续状态codec的历史基线，与正确Z、动作模型和未来预测无关。入口为：

```text
python -B baseline_adapter.py --root /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fullpool_codec_20261031_v1
```

默认时间上限从冻结协议 `baseline.max_seconds=3600` 读取；可显式设置更小的 `--max-seconds`，不能突破协议。外层队列另有包含数据准入的4830秒硬上限。内部检查在文件/帧边界执行，整段 `load_pool_index` 有界性也依赖外层进程期限。脚本不调用 `worker.load_runtime / setup_gpu / gate / preflight`，不需要GPU预检先完成，可与GPU预检并行。两者均成功才能开始训练，由root队列负责。

## 实际来源与计算

- **49片、8673帧**：48个新准备片加原tower10。读取已绑定 `states/<clip>/<frame>.pt` 中的 float32 `[2,252,448,3] reconstruction` 和相应完整cache，在CPU上重新计算全部四类×四指标。
- **tower17与历史已暴露tower14，各177帧**：读取精确绑定的旧multisource `evaluations/baseline/targets.jsonl`。保留原指标浮点数，检查对应原cache与存盘连续state的实际字节SHA、尺寸、状态metadata、同checkpoint/normalization/target-free模式。tower14标为`EVAL-only`，从不进入TRAIN。
- 全部 **9027 cache + 9027 state** 都在本次生产执行中流式读取实际字节并验证SHA。每帧张量hash通过后才从同一bytes进行 `torch.load(BytesIO(...),map_location='cpu',weights_only=True)`。缓存与状态不跨帧积累；聚合只保留数值字典。
- 加载新pool_index时会先逐文件核对8850训练cache；因此生产CPU阶段包含额外完整磁盘扫描。本实现没有声称零I/O，也没有执行这次真实扫描。
- 原DATA_BINDING SHA固定 `695959d1dce88089293ce795d8a962ecb284654fe530b0e8f286c61e73d02715`；原权重SHA固定 `a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917`。既有source静态文件与权重读实际字节核SHA；不构建codec、不导入原native模型模块、不渲染、不初始化CUDA。
- 49份准备inputs_hashes中的全部静态声明须与POOL_COMPLETE绑定匹配；与原multisource共享的静态文件须完全一致，且必须显式包含原codec权重、selection、protocol、codec/common/train代码。额外历史审计文件不因不在旧multisource DATA_BINDING就误判模型不同。
- 每片normalization来自该片已准入数据；保留合法的tower7_0 scale0.25，其他片scale0.5，不重新归一化。主/nearby相机、teacher targets与real直接使用同一cache，不改视角、点数、帧数和分辨率。

四类指标是`codec_real / teacher_real / codec_teacher / pseudo_view_teacher`，每类为`mae / psnr / workspace_roi_mae / workspace_roi_psnr`。公式与旧worker.image_metrics一致：float32全图/工作区绝对误差均值，PSNR=-10log10(max(MSE,1e-12))；ROI为`[67,38,380,227]`。teacher先clamp[0,1]；历史预测已clamp，必须是float32且在[0,1]内。不得用PNG补算，工作区矩形不能称语义物体分割。

49片的历史quality缺少workspace_roi_mae。新CPU结果除补齐该项，还对原有mae/psnr/workspace_roi_psnr逐帧核对，绝对容差分别为 **2e-6 / 2e-4 dB / 2e-4 dB**。超出即失败，不自动放宽或启动GPU重算。记录每项最大差异，`bitwise_equivalence_claimed=False`。已知tower10两次GPU实现之间也存在约1.043e-7 MAE和5.722e-6 dB差异，不能叫位级相同。

## 产物与比较接口

只在当前新run的 `evaluations/baseline` 新建产物；目录已存在就失败，不覆盖，不自动恢复。异常写本次拥有目录的FAILURE.json，不写旧来源。仅成功写EVALUATION.json可进入比较。

- `targets.jsonl`：9027行，按50个TRAIN片排序再tower14，每片0..176；保留新worker的clip/frame/split/source_group/pts、四类四指标、latent_positions/64通道、input_gaussians112896、input_voxels、decoded_gaussians、原checkpoint/encoder SHA、additional_codec_updates0。另有baseline_reused与逐帧baseline_origin。
- `state_manifest.jsonl`：9027行，记录**外部历史state的绝对路径、SHA、bytes**、shape、source cache SHA、normalization、checkpoint等。`state_path_is_external=True`、`fresh_encode=False`。没有伪造新state文件。
- `screenshot_manifest.jsonl`：空文件，明确本次不生成新截图。旧图片仍在原实验目录；本适配器不以截图替代float32输入。
- `EVALUATION.json`：`status=evaluated/arm=baseline/frames=9027/train_frames=8850/exposed_eval_only_frames=177`，含与新worker相同的summary_by_clip、summary_train_frames、summary_exposed_eval与三个manifest/targets SHA。source由`worker.verify_root`及相同make_epochs/plan_summary产生，与GPU预检source保持一致，但不要求预检先完成。
- 聚合保持逐帧指标均值，不能换为池化MSE之后再转PSNR。比较器可自行进行19组等权和bootstrap；本适配器没有添加显著性或结果成功判断。

## 重放证据边界

49片prepare曾reload坐标/特征并验证tensor相等，但没有reload后重新渲染。其行级`replay=null`、`baseline_origin.saved_latent_state_replay=False`。本CPU脚本也没有执行任何渲染。

tower17和tower14的354个旧baseline packet曾真正落盘reload后经原codec解码、主/nearby两视角渲染，PSNR≥60且MAE≤1e-5；这部分继承原始逐行两视角replay字典与旧EVALUATION/SHA绑定，不说本次又运行了重放。

因此总体必须为：

```text
baseline_reused = true
reconstruction_cpu_metric_frames = 8673
exact_historical_metric_frames = 354
every_packet_saved_replay_passed = false
all_frames_freshly_encoded = false
saved_latent_state_replay = false
historical_saved_latent_state_replay_frames = 354
saved_latent_state_replay_unavailable_frames = 8673
replay_performed_in_this_run = false
independent_native_gs_replay = false
gpu_executed = false
training_executed = false
screenshots_generated = false
```

连续latent+codec重放不等于无模型的native GS+SH重放；teacher native_render_parity与归一化similarity_parity也不等于预测GS重放。不能复制新训练两臂的全量fresh_encode/replay成功标志到本baseline。

## 本地验证与尚未执行事项

合成CPU测试命令：

```text
C:/Python314/python.exe -B -m unittest -v test_baseline_adapter.py
```

18项测试覆盖：与原worker的CPU公式逐值一致、ROI/PSNR下限、float32/完整分辨率/finite、teacher clamp、显式数值容差、同bytes hash后反序列化和篡改拒绝、0.25归一化、错误checkpoint/cache/PTS/64维坐标拒绝、完整9027覆盖与TRAIN/EVAL隔离、旧531行绑定和两视角重放、CUDA初始化拒绝、JSON重复键和digest拒绝。首次18项通过，耗时0.823秒；后续修改须由fresh CPU见证再验。

这些是合成数据测试，不是全池实际字节准入、CPU指标一致性或生产baseline成功证明。本次实现阶段没有SSH、真实全池读取、GPU、部署、冻结或启动新实验。
