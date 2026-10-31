# v10b：固定架构，验证有效区域伪视角监督

v9的Z对照改善数值但44/132关键动作仍失败；位置与特征目标拟合约30–31dB仍会破碎。此轮真正训练LDM，不继续目标变量oracle。唯一科学变化为在原RGB、差分与LFQ辅助损失上加入0.25倍附近视角teacher L1。原codec相应权重为0.25，但有效像素归一化后不是原全图损失的数值等价复制。

四臂rgb_z/gs_z从同一v9 spatial_z600恢复，rgb_constant/gs_constant从同一spatial_constant600恢复。各200步到800，恢复优化器与RNG，沿用make_plan(seed20261031,800)第600:800批，逐项核对前600与原来源。B4、10相邻帧、177全帧、448×252、全初始latent行、64维连续状态、九个16位LFQ动作ID、原空间调制与native部分解冻保持。没有状态LFQ、VGG、新头、学习率/特征范围改动；原全窗归一化仍在，不宣称严格前缀因果或AR已完成。

teacher来自原始112896 GS/帧。先逐帧校验源NPZ与缓存SHA，按原归一化、相机、完整点集重渲染附近视角，浮点RGB相对缓存target须MAE≤1e-5且PSNR≥60，另记录max_abs。alpha≥0.90再3×3腐蚀、外边界无效，产生177个固定bool mask。mask只用于损失、不进模型且不依赖预测。新增项为每帧有效像素RGB L1均值，再对B与10帧等权平均。alpha只证明覆盖，不是真实几何；薄物体边缘可能被排除，手仍可能主导损失。

环境warm reuse134 /data/CJH/envs/gaussiangpt/bin/python，spec SHA5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f。Provider ledger：experiments/gaussiangpt_neoverse_vw1_alignment_20261030_v5/.aris/compute/ssh-134.md；该块仅CPU见证、GPU环境证据继承早期ledger。本轮不安装、不重建，不以import成功宣称GPU可用。

独立新见证agent只需本文件、run-experiment/SKILL.md及其shared-references/compute-env-contract.md、上述ledger，并原样执行一次：

```powershell
C:\Python314\python.exe -B E:\MoViesGS\experiments\gaussiangpt_neoverse_gs_supervision_20261031_v10b\validate.py --root E:\MoViesGS\experiments\gaussiangpt_neoverse_gs_supervision_20261031_v10b
```

预期11项CPU测试通过，写CPU_VALIDATION.json；涵盖分段/完整图梯度一致、loss只用有效附近像素、零权重原step直通、阈值腐蚀与无梯度mask。CPU命令不访问服务器、不执行CUDA/ME，不自证真实GS训练。失败保留、不得由见证者修复或重试。观察超时继续同一句柄。

通过后将相同冻结源码与CPU证明部署到远端新目录。固定一次提交screen gg_gs_supervision_v10b_20261031，命令：

```bash
PYTHONDONTWRITEBYTECODE=1 timeout --signal=TERM --kill-after=20s 14420 /data/CJH/envs/gaussiangpt/bin/python -B /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_gs_supervision_20261031_v10b/pipeline.py --root /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_gs_supervision_20261031_v10b
```

一次NVML检查两GPU各free≥20GiB，每个worker入口重复验证、PyTorch分配/保留上限16GiB。新resource adapter明确代替历史55GiB准入，不改原core.setup/config；不降B/帧/点/分辨率。可与fulltrain_pool并存；只终止自己的进程组。

固定阶段：GPU0 teacher资产≤900秒 → GPU0四臂完整B4T10预检各一次丢弃更新≤900秒 → 程序等待teacher视觉验收≤1800秒 → 两GPU各顺序两臂训练/评测 → CPU汇总。GPU0为rgb_z、gs_z；GPU1为rgb_constant、gs_constant。每臂训练/评测各≤2400秒，进程外额外30秒；总队列≤14400秒。正式800次更新，另4次预检丢弃更新。任何阶段失败或超时停止自身兄弟进程、不自动重试。

teacher视觉门槛须实际查看44/132的teacher图与mask：关键黄条、绿梁主体保留，附近黑洞边界被排除；覆盖不能当几何真值。审阅者在同root写TEACHER_VISUAL_ACCEPTANCE.json（status=accepted/rejected、reviewed_frames=[44,132]、teacher_assets_sha256、preflight_sha256、观察说明）。只有绑定匹配且accepted继续训练，失败保留全部证据。该文件是人工/agent视觉验收，不是用户授权门槛。

预检恢复原600渲染两完整窗，与原结果MAE指标≤1e-4/PSNR≤.02；每臂真实一更新，检查native与LDM更新、原encoder冻结、Z各h重建梯度。正式训练新进程重新恢复600，不继承预检更新。保存固定800，不挑最佳步。

评测每臂168完整窗、1512未来曝光；先按176物理未来帧汇总，移动块16帧、2000次、seed20261031。保留三个打乱、常量、遮蔽未来与当前codec保持初态对照；每类14项动作贡献检查和逐h退化单列。额外object_box_mae使用现有[140,25,320,235]矩形，不冒充物体分割。新增监督对照须运动CI正改善、full/ROI/物体矩形/h0不劣；最终仍必须人工验收44/132物体位置、颜色、形状与重影。全部数值pass不自动等于可靠Z。

每臂保存两个关键完整窗×10帧×三个label的原生GS/SH包，格式可独立加载；本队列不声称已经独立重放。真实未来状态codec参考与Z输出明确分开。单个已接触TRAIN来源不能证明跨来源泛化，50clip数据池训练另行进行。goal_complete始终false。

校准修订：v10在正式训练前被视觉门槛拒绝，保留原FAILED与4次丢弃预检更新。alpha0.99筛选使132帧绿梁候选颜色像素仅保留27.45%。对两个固定teacher帧（没有训练模型）检查0.5/0.75/0.9/0.95/0.99后选0.90，黄条/绿梁保留90.45%/87.85%，仍排除侧面空洞及右边界。颜色候选只是覆盖校准、不是物体分割，也没有按模型效果选择阈值。本独立目录重新冻结并预检，不改原v10源码/结果；四臂训练仍同源600到800。
