# GS 时间差分与继续训练的四臂实验

这是尚未验收的前瞻实验。四组为 `z_rgb`、`constant_rgb`、`z_delta`、`constant_delta`；两种码各配原损失和原损失加图像差分。不是已成功动作码的声明。

从 v6 对应的共同 step100 恢复完整模型、原生 GaussianGPT 权重、优化器和随机状态。每组最多继续300步至400步，B4、相邻10帧、168完整窗口、177真实帧，全部初态latent点和原252×448画面保留。64维、动作LFQ16bit、原生编解码器解冻范围与学习率不变。未扩大候选点，也未混入严格前缀因果结构修改。

新增项为预测帧相对预测初态的RGB变化与真实变化之差，取未来9帧的全图L1与固定活动ROI L1，再取batch均值，权重1。初态及未来图像均收到完整梯度。当前整窗条件仍可能使未来码影响h0；此项限制没有被隐藏。

`temporal_step.py` 的6项CPU toy验证已由独立实现代理执行一次全部通过，比较完整图总损失的一阶梯度和优化器更新，并单独确认差分项的初态梯度。CPU测试不代表原生GS/GPU结果。首版真实B4 benchmark测得约2.50倍速度，但原方法与分阶段方法的重复性检查均在27个解析零梯度的注意力key偏置上失败；首版不采用、失败结果保留。新独立v2候选只固定这27个softmax冗余偏置的当前值，保留其优化器历史与所有参数组顺序，训练四臂一律执行同一策略；constant分支原本没有这些优化器成员，仍保持没有。该改造不改变64维或前向参数值，所有原数值门槛不变。新基础分阶段反传只有在独立真实B4 benchmark 的数值、完整性、资源和至少1.2倍提速条件全部满足后才能采用，绑定记录为 `BENCHMARK_ACCEPTED.json`。

环境沿用134的 `/data/CJH/envs/gaussiangpt`，spec SHA256 `5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f`。不安装或修改vendor。每进程6CPU线程，启动至少55GiB空闲，分配峰值48GiB以内，原600000候选/2400000GS guard保留，不结束其他任务。

冻结新 `CODE_MANIFEST.json` 后，新鲜文档见证代理必须原样执行一次以下唯一GPU预检命令，不执行训练队列：

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=15 172.19.58.134 'CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 timeout --signal=TERM --kill-after=10s 650 /data/CJH/envs/gaussiangpt/bin/python -B /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_temporal_z_20261031_v7/runner.py --root /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_temporal_z_20261031_v7 --mode preflight > /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_temporal_z_20261031_v7/preflight_process.log 2>&1'
```

本预检仅在临时模型上对真实 `z_delta`、`constant_delta` 各执行一次B4×10更新，检查有限性、初态梯度、每个B×h的Z梯度、三组原生参数更新及冻结encoder不变。输出绝对路径为 `/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_temporal_z_20261031_v7/PREFLIGHT.json`，只有 `status=passed` 且绑定当前manifest才可提交训练。

训练由 `pipeline.py --root <本目录绝对路径>` 一次提交：GPU0顺序运行 z_rgb、constant_delta；GPU1顺序运行 constant_rgb、z_delta，以平衡后验编码开销。每臂训练内部1800秒/进程1950秒，评测内部2400秒/进程2500秒，整队列9600秒；失败保留，不自动重试。父进程超时使用已经验证的原postfit进程组终止实现，只停止自身子组。程序等待，模型不轮询训练。

所有组训练结束后，按预先指定的200/300/400步选择共同存在的最大检查点，不能按质量选择。保留真实final步数和文件；不把中间checkpoint改名final。每组评测168×9个未来曝光，并分别记录初态、各自预测初态相减的时间差分，以及隐藏真实未来、只重复初态特征但保留时间位置的对照。盲化复用v6已验证helper，合法ID经同解码器重建，属于分布偏移诊断；冻结参考没有差分指标，必须为null而非0。

结果在 `evaluations/step_NNNN/<arm>`，终态比较在 `SUMMARY.json/md`。沿用原14项描述性区间规则及逐h非恶化规则；重叠窗口先按176物理目标帧归并，再做块长16、2000次bootstrap。差分方案还须改善差分ROI/运动区间，且普通全图/ROI与h0均值不恶化，并胜过同损失constant。所有筛查即使通过，也不等于动作迁移、泛化或AR目标完成。
