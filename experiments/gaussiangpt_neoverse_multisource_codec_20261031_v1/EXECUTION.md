# 两来源连续 GS 编解码对照

本轮解决状态编解码器的跨来源失真，不训练Q-Former或动作LFQ，也不自动替换正在使用的LDM底座。现有完整E/D在tower17的177帧上实际训练3000步；冻结后处理tower14约17.8dB。该失败尚不能单独归因于过拟合、编码器或解码器。本轮使用两个TRAIN来源，比较解冻范围，同时保留完全相同初始权重的冻结基线。

固定TRAIN为 `buildingblock_tower_17_176` 和 `buildingblock_tower_10_176`，各177个相邻帧。tower17历史为DEV、已获用户授权用于TRAIN；tower10原本就是项目TRAIN。`buildingblock_tower_14_0`仅评测，不进采样器、不参与梯度；它历史为TEST且已经多次评测接触，不能称全新独立测试集。三片各保留全部112896个teacher GS的聚合信息、448×252和各自已验证的首帧固定归一化。不得将旧17帧或93帧采样替代177帧。

初始checkpoint是 `gaussiangpt_neoverse_state_motion_20261030_v1/state/codec_final.pt`，SHA `a1c32dbcb29d23da9bc65cbf16cc49d40cdb36954952b8ac91580fd3eeeb2917`。两臂完全相同初始化，并使用相同fresh AdamW、lr1e-4、betas(0.9,0.95)、weight_decay0、clip1；不恢复旧优化器。每步分别处理两个TRAIN来源各一完整帧，累积平均梯度，逻辑batch为2；相同seed20261031计划，前177步覆盖全部354训练帧。固定追加300步，保存150/300，不按效果挑checkpoint。前240步保留原target occupancy forcing，后60步关闭；所有评测关闭forcing。

- `decoder_only`：整个 `ae.decoder`、`ae.feature_heads`、`ae.gaussian_multiplier` 可训练；编码器、输入embedding、set adapter及其buffer保持不变，编码模块始终eval。
- `full_codec`：原E/D、输入embedding、set adapter、GS heads和multiplier均可训练；不改变任何网络宽度或GS表示。

完整复用旧 `train.terms`：real L1=1、VGG19=.1、teacher L1=.5、pseudo-view L1=.25、固定工作区L1=.25、occupancy BCE=1。保持连续Nx64、grid0.008、stride4、每输出voxel四个GS、无状态量化；保留原输入voxel80000、候选600000、输出GS2400000及48GiB上限。附近视角来自teacher，只是重建一致性证据，不是实测三维真值。

所有来源由 `sources.py` 核对。tower10数据准备必须已经在指定manifest下真实完成，随后绑定实际DONE、SUMMARY及177份cache SHA；tower17/tower14也检查其权威索引、PTS、归一化和全部缓存。`DATA_BINDING.json`只在真实来源完整验证后生成。每阶段复核来源和源码。新encoder下绝不读取旧latent作为当前状态；各模型评测全部重新编码并保存状态，磁盘重读后核对两个视角的渲染重放（PSNR≥60dB、MAE≤1e-5），packet绑定对应checkpoint和clip/frame。

环境warm reuse：134 `/data/CJH/envs/gaussiangpt/bin/python`；原GS spec SHA `5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f`，provider ledger继承 `gaussiangpt_neoverse_vw1_alignment_20261030_v5/.aris/compute/ssh-134.md`。旧全E/D训练以及v8真实GPU更新已成功执行相同环境；本次仍须自己的真实预检。不安装、不升级，原VGG19缓存文件也已固定SHA并检查，不允许静默下载。

独立文档见证者读取 `E:/Codex/.codex/skills/run-experiment/SKILL.md` 及其 `../shared-references/compute-env-contract.md`，原样执行以下CPU命令一次，不修复或重试：

```powershell
C:\Python314\python.exe -B E:\MoViesGS\experiments\gaussiangpt_neoverse_multisource_codec_20261031_v1\validate.py --root E:\MoViesGS\experiments\gaussiangpt_neoverse_multisource_codec_20261031_v1
```

该命令核对源码manifest、语法和四个CPU测试文件，写CPU_VALIDATION；不访问远端数据、不运行ME/GS/GPU，不能证明新训练已成功。保存原始exec/session/exitcode；观察超时只沿用原句柄，不重新执行。

通过后部署完全相同的源码和CPU证明到远端同名新目录，核对哈希，仅提交一次screen `gg_multisource_codec_20261031`：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=20s 22420 /data/CJH/envs/gaussiangpt/bin/python -B /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_multisource_codec_20261031_v1/pipeline.py --root /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_multisource_codec_20261031_v1
```

程序先等tower10准备队列PID437557/start_ticks38609998，根目录 `gaussiangpt_neoverse_tower10_replay_20261031_v1`，manifest `f9d8db6b8f6c77636dc78cc11c7cbdd5da91d710e5a0675620fb4dd6facd0418`。必须真实退出、唯一DONE complete、两个子任务成功且SUMMARY哈希相符；失败或缺终态拒绝训练。等待上限15000秒、间隔60秒；不停止父实验或共享进程。

顺序固定：CPU来源准入（硬上限300秒）→GPU0完整双臂预检→GPU0共享冻结基线→两臂分别在GPU0/1训练→两臂并行评测→CPU比较。两GPU一次性检查各空闲≥55GiB。预检每臂两次丢弃B2更新，分别forcing开/关，检查初始预测一致、四个真实帧的Nx64和重放、梯度、冻结参数/buffer与full encoder更新；共4次trial、0次production。任意PREFLIGHT产物存在则拒绝原地再试；任何预检失败都禁止正式训练。

预检内900/外930秒；共享baseline、每臂train/eval内1800/外1830秒；比较300秒；队列总上限22400秒。两臂训练都必须实际完成300步才进入最终比较。任一失败或超时立即停止本队列及其自有子进程组，写失败证据；不自动重试、不减帧/点/分辨率，不改变划分或预算。

结果：`arms/<arm>/`保存SOURCE、trace、checkpoint和TRAIN_DONE；`evaluations/{baseline,decoder_only,full_codec}/`保存三片各177帧，共531条评测、新states及逐包重放证据。固定0/44/88/132/176截图包含真实图、teacher、codec及附近视角对照。比较按clip配对，block16/2000次/seed20261031，报告两臂相对共同基线及相互差异；不能把531帧当独立来源泛化样本。

完整结果还须人工核对积木颜色、位置、破碎和附近视角。原codec绝对质量门槛只作参考。工程完成、像素指标改善、甚至两来源训练成功，都不等于动作码Z学好；不自动选择生产codec，`goal_complete=false`。NeoVerse teacher是离线全视频估计，本轮不证明感知因果性或AR预测能力。任何后续LDM必须使用与所选encoder严格匹配的新状态缓存，并重新验证动作贡献和跨来源效果。
