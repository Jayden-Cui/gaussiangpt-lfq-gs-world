# 连续 Z / LFQ 同条件对照：有界源码审计与待实施协议

日期：2026-10-31。状态：**实验协议仍为草案；核心 adapter 已实现并通过 9 项真实依赖 CPU 测试，尚未训练、尚未验收新 codec**。实际实现与证据以 `ADAPTER_NOTES.md` 和 `ADAPTER_PREPARATION.json` 为准；本文件下文保留设计时的拟定预算与待执行检查，不表示它们已经执行。所有旧实验的源码、manifest、结果均保持原样。研究顺序为 GS codec replay → codec 验收 → 同条件连续 Z / LFQ → 物体运动验证；不训练 AR。

本轮以 v10b 及其绑定代码为依据，未重新查论文。任务交接状态是：v10b 四臂到 800，附近视角监督的运动指标改善约 1.84%，但 44 帧黄条与 132 帧绿梁动作仍失败；fullpool 的 decoder_only 到 4426 但评测 replay 失败，full_codec 在 3940 中断、仅 2213 正式检查点。上述运行状态来自主任务交接，本审计没有把完成训练等同于验收通过，也没有重新启动其中任何任务。

## 1. 已核对的真实调用链和张量

| 层级 | 当前实现与语义 |
|---|---|
| v10b | `../gaussiangpt_neoverse_gs_supervision_20261031_v10b/worker.py:58` 加载 v9；`loss_step.py:54` 调用 `encode_windows` / `decode_latents`，加入 RGB、差分与固定 teacher mask 损失 |
| v9 | `../gaussiangpt_neoverse_spatial_action_20261031_v9/worker.py:113` 绑定 v7 的终态加载器；`spatial_action.py:49` 在未来槽加入初态位置与各自动作码的乘性空间调制 |
| v7 → v6 | v7 复用 v6 `code/native_experiment.py`、`anchored_ldm.py`、`aligned_ldm.py`、`model_core.py`、`multiscale64.py`；不是另一个 512 维图像模型 |
| state codec | `../gaussiangpt_neoverse_state_motion_20261030_v1/state/code/codec.py:106` 输出连续 `[M_t,64]`，`ae.vq=Identity`；state 没有 LFQ |
| LDM | `AnchoredLDM` 的 encoder / decoder 各 4 个保留全部坐标行的稀疏多尺度时空块；64 维。不是 20 块完整图像架构 |
| 动作 | 9 个独立 QFormer，各 2 层、4 heads、1 个 64 维 query；共享一个 16-bit LFQ |

原生 GS 每帧输入为 112,896 个点，图片为 448×252。codec 已有体素集合聚合与原生预测剪枝，latent 行数 `M_t` 和 decoder 输出 GS 数 `K` 均可变；**112,896 不是 latent 行数，也不是每个输出必须强制补齐的 GS 数**。新实验保留全部原始 GS 的现有聚合统计、全部初态 latent 行、现有候选上限与完整分辨率，不增加抽点、截断、top-k、降分辨率或截短窗口。`.008` 为 voxel grid、stride=4、`.032` 为归一化坐标中的 latent 间距，不能直接称为米。

以逻辑 batch `B=4`、窗口 `T=10` 为例：

| 张量 | 真实形状 | 来源 |
|---|---|---|
| 单帧 state | coordinates `[M_t,3]`；features `[M_t,64]` | 冻结 state encoder，所有行 |
| posterior 展开 | `[13,N_union,65]` → `[13,N_union,64]` | 3 个左侧 padding、10 个真实 state；额外 1 维为 presence |
| QFormer h keys | `[B,L_h_max,64]` 加 valid mask | 本窗口 `0..h` 的有效编码行；`L_h=sum(M_t,t=0..h)`，不是统一抽样的点数 |
| QFormer h query / output | `[B,1,64]` | 第 h 个独立 posterior |
| `z_prequant` | `[4,9,64]` | 9 个输出拼接 |
| `project_in` | `[4,9,16]` | 无 bias 的 Linear(64,16) |
| 硬码 / ID | `[4,9,16]` 的 ±1 / `[4,9]` 整数 | 每 h 一个 16-bit ID，范围 0…65535 |
| `project_out` | `[4,9,64]` | 无 bias 的 Linear(16,64) |
| 解码输入，每个样本 | `[13,M_0,64]` | 初态 4 槽＋9 个动作向量各广播到全部 `M_0` 行 |
| 解码输出，每个样本 | 10 个 `[M_0,64]` | `F0 + output_std * 3 * tanh(head(hidden))` |
| 原生包 | `gs [K,14]`、`sh [K,4,3]` | 原生 decoder 与属性转换；K 可变 |

当前 ID 是**窗口内的 9 码后验动作束**。QFormer h 可读 `S_0…S_h`；这不是每对相邻帧独立编码的单一动作，也不是已训练的 AR token。B4 是 LDM 后验 batch；不能把 fullpool codec 的顺序逐帧 B4 与此混为一谈。

数据保持完整 177 帧、frame_gap=1、168 个 start=0…167 的完整 10 帧滑窗、1512 次未来曝光、176 个不同未来帧。不得为模仿某篇模型改成不重叠采样。

LFQ 实际依赖通过只读 SSH 核对，未安装软件：

```text
/data/CJH/envs/gaussiangpt/lib/python3.10/site-packages/vector_quantize_pytorch/lookup_free_quantization.py
version: 1.31.1（已有环境记录）
SHA256: 9fb68bdcb3763ac580c2a0f931640a70a14a0fed65e0b29a8800ee015b4452eb
```

该源码实际执行 `u=10*tanh(project_in(z)/10)`，随后 `where(u>0,+1,-1)`；训练采用 identity 直通梯度（STE），推理直接硬符号；`codebook_scale=1`、`spherical=False`、`orthogonal_rotation=False`。`project_out` 将 16 维返回 64 维。熵项分 9 次对各 horizon 的 B4 计算，不能合成 36 token 的跨 horizon 熵。当前 commitment 权重为 0，LFQ 内 entropy 权重 .1，外部 `aux_multiplier=.1`，逆温度 1，softplus offset 5。

## 2. 归一化审计：明确是哪一种未来依赖

### 2.1 已知初态与坐标归一化

`state/normalization.json` 明确 `source_frame=0`、固定 center=[0,0,0]、scale=.5。各片段新 fullpool 也采用各自第一帧确定的二次幂尺度，不能在跨片段情况下硬写 .5。这不是用本窗口未来几何计算的归一化。

`AnchoredLDM.initialize_output_units` 仅从明确绑定的 TRAIN frame0 features 计算每 channel 的 population std，之后为固定 buffer；它也不是每个窗口未来目标决定的尺度。若换 encoder，须重新绑定该来源和数值；若仅换 decoder，须核对 encoder/features 身份，不能默默改变尺度。

### 2.2 当前破坏解码前缀因果性的两个位置

1. `v6/code/multiscale64.py:_SpatialTemporalBlock64` 对 decoder 设置 `TemporalResidual64(across_time=True)`。`model_core.py:22` 的 `MaskedGroupNorm64` 此时在 `(time,node,group_channel)` 三轴统计，即 13 槽一并求均值/方差。
2. 四个 decoder stage end 的 `AdaptiveWindowNorm64` 既执行上述整窗 norm，又在 `model_core.py:54` 对原始 `condition` 的全部 `(time,node)` 池化，生成共享 affine scale/shift。

这两个位置的条件张量是 `cat(initial_context[4,M0,64], action[9,M0,64])`。动作来自 `project_out(code)`；没有在这里直接送入未来原始 state features 或未来真实图片。因此准确结论是：**输出 h 以及初态 h0 可依赖 h 之后的后验动作码，违反动作解码前缀因果性；不能据此声称发现了真实未来 state 绕过 Z 的直接旁路**。恒定或打乱 Z 后仍有初态、位置与时间结构，必须分别测量贡献。

v9 空间支路本身只读 slot3、当前动作 slot 与初态坐标；其两个 LayerNorm 只沿 64 channel，未新增跨未来统计。问题仍在其 `base`。

### 2.3 posterior 读取未来的边界

`aligned_ldm.py:105` 用全窗坐标并集建立稀疏展开，但 presence mask 在每个时间生效；encoder norm 不跨时间，时间卷积只左填充，QFormer h 只取 `0..h` keys。源码设计意图是前缀不受后续 state 影响。**并集坐标本身不够证明旁路，仍须对坐标、features 和缺失行做前缀不变性见证**。

posterior 的目标正是从已经观察的变化中提取 Z，因而 QFormer h 读取 `S_1…S_h` 是允许的。不得把“修因果性”实现成 QFormer 只能读初态、全部未来 mask 掉或只能读相邻两帧，否则同时改变了问题和 token 语义。

### 2.4 两臂共同采用的最小修正

在新目录实现并绑定 `InitialReferenceNorm64`，保留每个现有 norm 的 64 维 weight/bias 和 eps=1e-6；将 decoder 中上述跨时间的 norm 替换为：

```text
输入 x: [13,M0,64] → reshape [13,M0,4,16]
reference = x[:4]；reference_active = active[:4]
每个 group 只在初态4槽的 (time,node,16channel) 求 mean/var
该 mean/var 广播用于全部13槽
AdaptiveWindowNorm 的 affine pooled 也只从原始 condition[:4] 求得
```

每层 x[:4] 在左因果卷积与同样初态 norm 的归纳保证下只依赖初态。不要把 `reference` 换成 pooled future code，不要在当前 batch 或当前评测窗口更新 running statistics。spatial block 的逐帧规范化、空间支路当前码 LayerNorm 可保留，它们不会跨读取后续 h；若实施时改变它们，须作为共同架构变化明确登记。

`encode_windows` 还应显式用 `encode_initial(window[0])` 构造 decoder contexts，以使模型接口层面初态支路只接收 S0；posterior 保留 `prepare(window)` 和原 h-prefix keys。若后续为了计算复用使用 `prepare(window)['initial']`，必须先证明两者在值和梯度上等价，不可只信注释。

使用初态参考统计是本草案的默认选择；不同时引入新的 TRAIN 全池激活统计。若改用固定 TRAIN 统计，必须只由已准入 TRAIN 数据、固定 checkpoint/层/采样规则生成后冻结；不使用 tower14、当前测试窗口、真实未来图片或在线 batch 重估，也不能在看结果后选择两种 norm 中更好者。

共同修正后的比较回答“在同一初态归一化架构中，离散化是否损害动作信息”；它不能直接把相对旧 v10b 的变化全部归因于 LFQ。

## 3. 最小公平动作瓶颈

主比较只有两种训练模式：`lfq16` 与 `continuous16`。均保持 QFormer、query 数、64→16→64 投影、decoder、空间调制、损失、native 解冻范围相同。**不把原始 `[B,9,64]` QFormer 输出直接交给连续 decoder**；那会取消 16 维瓶颈与两组投影，无法只比较量化影响。两组 projection 合计 2048 个可训练参数，无 bias，两臂完全相同。

为避免连续码和 ±1 的幅值不匹配，默认预注册以下一个共同的固定尺度规则：

1. 在共同初始化、验收通过的 state 以及固定 TRAIN 168 窗上，no_grad / eval 计算全部 `u=10*tanh(P_in(z)/10)`。
2. 计算唯一标量 `s_ref=sqrt(mean(u^2))`，覆盖所有窗/h/channel；不能每窗、每 h、每 batch 重估。要求 finite 且 >1e-6，否则预检失败，不自动换规则。
3. 保存 calibration starts、state/encoder/model hash、原始 RMS、`s_ref`。两臂加载同一 buffer、同一值；训练和评测都冻结。
4. 共同定义 `v=u/s_ref`。连续重建用 v；LFQ 重建用 `v + (where(v>0,1,-1)-v).detach()`，推理用硬码。两臂均用同一个 `P_out` 映射到 64 维。

初始 TRAIN 总体 RMS 的尺度均为 1，且连续 16 维瓶颈保留方向/幅度信息。训练后尺度可能漂移，应记录 mean/RMS/p95/max，不得根据评测表现重新校准或仅给一臂加 clamp。固定公共增益会改变旧 LFQ 的重建梯度尺度，必须披露；其硬码 forward 不变。这里比较的是**同一公共尺度规范下的连续与符号码**，并不声称逐梯度复现旧 v10b。

最小实施方式是在新 `action_bottleneck.py` 中复用已绑定 LFQ 的投影参数与熵函数/调用。为避免不同正则项混入主比较，**两臂都保留同一 LFQ entropy auxiliary**（同 u、同温度、同 per-h B4、同两层权重），只有 reconstruction 的 v→hard/continuous 分支不同。可调用原 quantizer 取得相同 auxiliary 和诊断 IDs，再用显式分支构造重建 token；该额外调用在两臂一致，未用的 quantized 输出不参与重建梯度。应验证 `s_ref=1` 时 LFQ 支路与绑定库在 train/eval 的值及梯度一致。

这样测试离散化本身，**不能声称得到无量化正则的最优连续模型**。若主结果显示连续受益或两者都失败，可另行预注册两臂同时 aux=0 的成对复核；不能训练完成后只给连续臂去掉 aux，再沿用主比较结论。

连续 `indices` 仅是诊断性的 `sign(u)` ID，不能把它作为实际连续 action 保存或解码。建议新返回字段为 `action_latents [B,9,64]`、`action_codes [B,9,16]`、`diagnostic_ids [B,9]`、`aux_loss`、`initial_contexts`；旧 `z_quantized` 字段若暂作兼容别名，receipt 必须标明其 mode。`decode_ids` 只允许 LFQ；连续 replay 读取浮点 code 后走相同 `project_out`。

## 4. 初始化、更新预算与共同训练条件

默认首轮采用**共同随机 LDM 初值＋同一已验收 codec**，保存一个 `common_init.pt` 后克隆两臂；两臂 fresh AdamW，optimizer state 为空，保存参数、参数组和 RNG hash。不能让 LFQ 从 v10b800 继承已训练模型而连续从随机开始，也不能把两个不同 v9 源臂当成同初值。

同一个旧 LFQ checkpoint 分叉再训练虽可作短期机制诊断，但两臂都继承了对离散路径的历史适配，它不应替代本主比较；若采用须另起实验名并标明共同预训练步数、分叉步数和偏向。

建议在 codec 门槛通过后冻结以下正式预算：每臂 **800 次 optimizer 更新**，共同 `make_plan(seed=20261031, steps=800)`，B4、T10；一周期 42 步无放回覆盖全部 168 起点。两臂各 3200 个窗口曝光 / 32000 次帧曝光（重复保留）。固定报告 step800，不按 44/132 或评测最优挑步。800 是从头比较的拟定预算，不能与 v10b 的“600+200”混写为同一训练轨迹。

共同超参数沿用：motion/spatial action lr=3e-4，native lr=1e-5，AdamW betas=(.9,.95)、weight_decay=0、grad_clip=1、residual_scale=3；state encoder 参数和 buffer 全冻结，codec 保持 eval、无目标强制 pruning；只训练 `ae.decoder.out`、`ae.feature_heads`、`ae.gaussian_multiplier`。如果验收指出必须冻结整个 codec 才能保持质量，则两臂共同改为全部冻结并在训练前登记，不能运行中单臂切换。

共同损失默认取 v10b 的 gs recipe：全 10 帧 RGB L1＋固定 hand_activity ROI L1＋9 个未来 motion-weighted L1＋delta_weight=1＋nearby teacher 有效区 L1 权重 .25，alpha阈值 .90、3×3 erosion；固定 target mask 只用于 loss、不进模型。所有真实未来目标只监督损失。teacher 资产须独立绑定 raw GS / cameras / normalization / renderer；codec 变化不自动赋予旧 teacher 资产新来源。若 codec 验收要求重新选择 recipe，须先统一两臂并在跑前冻结，不能继承旧 v10b 四臂结果充当新条件基线。

预检丢弃更新两臂各至少 2 步，记录为额外开销，正式模型重新从 common_init 加载。零初始化 residual head 使第 1 步 upstream reconstruction gradient=0 是已知行为；第 2 步要求所有 h、每个样本、投影和 QFormer 有 finite 非零重建梯度，不能误判第 1 步并改初始化。两臂更新步数、窗口曝光、损失归一化相同；wall time、FLOPs、显存单列，不把更快一臂多跑步当作公平。

## 5. codec / replay 准入与来源绑定

在 replay 原因查清和 codec 验收完成前，允许写代码草案、做 CPU 形状/因果测试；不开始上述正式训练，也不冻结“使用最新 fullpool decoder”这一依赖。

验收要区分三条路径：

1. 原始 GS teacher 的两视角渲染，与缓存目标核对；它检查数据和相机。
2. fresh encode 得到的连续 state → decoder，及 state 落盘再加载 → 同一 decoder；它检查 state/codecs/稀疏结构重建。旧 `multisource_codec/worker.py:345` 的 `encode_replay` 比较的是 direct sparse decode 和 rebuilt sparse decode，两视角阈值 MAE≤1e-5、PSNR≥60；失败不能被低训练 loss 抵消。
3. 原生 gs/sh/cameras 包落盘后，在独立进程只加载包和 renderer 重放；它不依赖 encoder 或 motion model。v10b 包清单的 `independent_replay_performed=False` 不是此项通过证明。

数值 replay 一致只证明重放一致，codec 能否保留黄条/绿梁形状、位置、颜色和遮挡还需独立质量验收。不得为通过而放宽阈值、选择有利帧、重标失败为通过或使用目标强制剪枝。若发现 coordinate map / row alignment 修正，必须在新文件中绑定修正源码与 witness，保留旧失败记录，然后重新评估受影响检查点。

**decoder_only 的 encoder 不变并不意味着可以沿用整份旧 source metadata。** `v6 NativeBridge` 当前固定加载 `STATE_ROOT/codec_final.pt`，`gs_bridge.verify_native_sources()` 固定旧 `state_checkpoint_sha256`；新入口必须显式接收验收后的 codec binding，不能替换内存权重后仍写旧来源。

新 binding 至少分别记录：

```text
raw_gs/cache/camera/normalization SHA
encoder_parameters_and_buffers_SHA
state_producer_checkpoint_SHA / state_packet_SHA / source_cache_SHA
decoder_checkpoint_path + file_SHA + decoder_parameters_and_buffers_SHA
codec selection/protocol/source-code SHA、grid、stride、state_vq=Identity
fresh_encode witness / cached_state_equivalence witness
latent replay receipt SHA / native packet replay receipt SHA / quality acceptance SHA
LDM common_init、action adapter、normalization adapter、train plan、optimizer-group SHA
每臂最终 codec 与 LDM checkpoint SHA（partial 解冻后 decoder 已再次变化）
```

若采用已验收 decoder_only：先核对 encoder **参数和 buffers** digest 与 state producer 一致；允许复用经逐字节校验且坐标/features 语义一致的旧 state 作为特征来源，但必须保留旧 state producer 身份并另记新 decoder 身份，不能伪标为 fresh encode。默认使用新验收生成的 fresh state，避免模糊语义。完整 fullpool codec 验收仍要求协议规定的数据覆盖，不能因 tower17 可复用就宣称 9027 帧评测完成。

若采用改变 encoder 的 full_codec：必须重新编码全部受影响 state；旧 state、output_std、尺度校准和缓存 manifest 都失效。仅有 2213 阶段检查点、3940 中断状态或 FAILED_DO_NOT_EVALUATE 权重，不满足原 4426 终态规则，不得自动转正。

codec 选择只基于其已注册重放和重建质量门槛；不可查看哪一个让 LFQ 或 continuous 更好后再选。若多个完整 codec 都通过，先固定唯一准入者，再构建两臂共同初值。未通过时保持路线在 codec 阶段。

## 6. 评测：正确码、固定码、打乱码与物体运动

每臂最终保存全部 168 窗的实际动作。LFQ 保存 IDs 和 ±1 code；连续保存 float32 的 16 维 codes 和其 SHA，不舍入成 ID。固定 `decode_actions` 输入后只允许 initial state、相机和动作进入解码；再次跑 posterior 不能冒充落盘 replay。

| 控制 | 两臂共同定义 |
|---|---|
| correct | 当前窗口自己的完整 9 码束 |
| shuffle42/84/126 | 使用 `(start+offset)%168` 的完整 9 码束，保留 h 顺序；接收方初态、相机、decoder 不变 |
| fixed_bundle | 使用本臂 start=0 的 9 码束，所有窗口恒定；供体在协议中预定，不按效果选择 |
| fixed_single | 使用本臂 start=0,h=1 的一个实际码广播到所有 h、所有窗；连续不使用任意 zero-vector，LFQ不把随机 ID 当在分布内证据 |
| blinded_posterior | 用初态重复替换窗口中 future state，时间/PTS 保持原递增；不得保留真实未来坐标或 features |
| adapted_held | 本臂最终 codec 解码真实初态，在每个目标相机下保持该初态 |
| decoded_actual_target_state | 本臂最终 codec 解码真实目标 state，明确标为 codec reference，不当成预测或 Z 成绩 |

固定供体/打乱均在**训练后的同一模型**进行，测量该模型是否使用 action 信息；它们不等于从头训练的恒定码模型。旧 v10b constant 模型具有不同 norm、codec 和训练史，不能用作本轮 matched trained constant。若需要排除共同 decoder 适配能力，可另加共同初值/相同步数的 `trained_constant` 第三臂，但必须先登记预算；不属于最小两臂主比较，也不可用旧 constant 冒充。

两臂均对全部 168 窗 / 1512 个未来曝光、176 物理未来帧评测，h0 单独报告。先在同一物理帧内均匀平均重叠窗，再以物理帧为统计单位；保留每 h 结果。沿用 paired moving-block bootstrap：block16、2000次、seed20261031，报告连续相对 LFQ 与各自 correct 相对控制的 paired improvement/95%CI。单一已使用 TRAIN clip 只支持表示/拟合诊断，不是跨来源泛化。

报告 full RGB、hand_activity ROI、motion-weighted MAE、delta、固定 `[140,25,320,235]` object_box MAE；矩形不是物体分割，运动权重也不是语义物体 mask。关键验收必须查看 s35→44 与 s123→132 两个完整 10 帧窗及 44/132 两视角：黄条/绿梁位置和位移方向、颜色、形状、重影、背景/手占主导程度。数值优势不能替代此项。

原生 GS 包至少保存上述两关键窗的全10帧、correct/shuffle84/fixed_bundle/decoded_actual_target_state，以及绑定图像、相机与normalization；两臂均做独立进程 replay。LFQ 记录 per-h entropy、不同 ID 数、bit occupancy；连续记录 per-h variance、距离、RMS 与 fixed/shuffle 敏感性。连续诊断 ID entropy 不能作为连续信息容量结论。

结论边界：连续明显优于 LFQ 且通过对象动作验收，才支持“当前 16-bit 符号瓶颈损害此任务”；两臂都失败需继续检查 codec、解码器/优化或后验表示，不自动增加码本/维度；两臂接近且都成功，才说明此设置下离散化不是主要瓶颈。任何结果都不自动证明可预测 AR 动作或目标已完成。

## 7. 实施位置与正式启动前证据

建议在新的实验目录维护自己的源码与 manifest，不编辑旧 frozen 文件。最小接口变动：

| 新文件/入口 | 职责 |
|---|---|
| `action_bottleneck.py` | 公共投影、固定尺度、continuous/LFQ 分支、同 auxiliary、保存 actual codes |
| `initial_norm.py` | 初态参考统计和初态 affine pooling；保留参数形状 |
| `model_adapter.py` | 在 AnchoredLDM 的 encode/decode 接口注入两项共同变动；显式初态 contexts；保持 v9 空间支路 |
| `codec_binding.py` | 显式加载验收 codec/state 来源，拒绝旧 source metadata 冒用 |
| `worker.py` | 用同一 init/plan/recipe 跑两臂；不能直接沿用 v10b restore600/restore800 的旧科学契约 |
| `evaluate.py` | 模式明确的落盘 action replay、固定/打乱控制和独立 native 包 |

预检须产生以下针对性证据，不能只凭 import 成功：

1. 每个固定 witness 的 shape、所有行/点来源与 bound hash；连续实际 action 并非 ID replay。
2. `s_ref=1` 时 hard adapter 对库的 forward/backward 等价；校准后两臂参数名、shape、公共初值、RNG、optimizer groups 和计划完全一致。
3. posterior 修改 `S_{h+1:}` 时 `z_h` 不变，修改 `S_{1:h}` 时允许且应能改变；显式 context 只读初态。随机遮蔽未来后不能复用原后验输出。
4. decoder 固定 initial、改变 `Z_{h+1:}` 时输出 `0..h` 不变，h0 对全部 Z 不变；相应未来 Z 梯度为零。改变本 h Z 后允许影响本 h 与更晚输出。测试多个非零权重模型，不能只拿零 residual head 的“恒等输出”通过因果测试。
5. 冻结 encoder 参数/buffer、decoder eval、无 target_key、全量 frame/point/resolution；两步真实 B4T10 丢弃更新，native 与 motion 各组梯度和更新证据。
6. 分段反向与小图直接完整反向的损失/梯度等价，尤其保留 delta 对 initial render 的导数；已有 v10b 的该项证明不能自动覆盖新 adapter。
7. fixed/shuffle donor mapping 无 identity 意外、整束移动、连续浮点 serialization round-trip、两视角 latent/native replay 与覆盖计数。

这些测试为实现前的约束；本文件没有声称已经执行它们。

## 8. 哪些可以自动推进，哪些必须等待证据

**现在可自动推进**：精确源码审计；只在新目录实现公共 adapter 与记录格式；合成 CPU shape/梯度/前缀测试；制定固定配对计划、供体映射、评价协议；准备只读 checkpoint/数据 hash 核对。无需因普通实现选择再次请求用户许可。本分工仅产出 DESIGN.md，因此此处未执行这些实现步骤。

**replay / codec 结果出来前不能定案或启动**：实际 codec checkpoint 选择；旧 state 是否可复用；是否需要重新编码和校准；replay 修复适配器的最终来源；若 codec 质量不合格是否需要继续 codec 阶段。禁止把 decoder_only4426、full_codec2213/3940 自动当作验收通过。

**可在门槛通过后按预注册规则自动执行**：绑定唯一合格 codec，固定上述两臂共同 norm/尺度/初值/损失/800步协议，做独立 CPU/GPU witness，随后一次正式配对训练和完整评测；失败保留证据，不能自动降 B/T/点数/分辨率、挑另一 checkpoint、改单臂超参、重试到成功。资源准入或超时须在正式协议冻结前明确，不能临场用缩减科学条件解决。

**必须由结果触发、另行记录而非悄悄执行**：增加 trained_constant / aux=0 配对、改变维度/码本、换 norm、扩多来源动作训练或延长步数。物体视觉验收是证据门槛，不是让用户重复批准已授权工作的许可门槛。任何情况下本阶段不训练 AR。
