# 连续 / LFQ 配对 runner：最小接入方案

2026-10-31。**本文件是 CPU 源码准备，不是实验协议冻结或启动记录。最终 codec 尚未选择，800步预算仍待冻结。**本次只新增本文件；已阅读 `DESIGN.md`、`ADAPTER_NOTES.md`、三个 adapter，以及 v10b→v9→v7→v6 的真实训练调用链。没有修改已有 agent 文件、部署源码、执行 GPU 或提交实验。

最小方案是新增一个从共同随机初值训练的薄入口，复用已验证的模型与损失原语；不调用旧续训 runner 的 runtime / restore600 / restore800。先保留现有单片段 `buildingblock_tower_17_176`、完整177帧和168个T10窗口，不将50片段codec训练池自动扩成动作训练范围。

## 1. 复用位置与实际需要改的接口

| 部分 | 复用方式 | 必须新接的部分 |
|---|---|---|
| 模型 | `model_adapter.load_pinned(parent)` → `AnchoredLDM`、`attach_spatial_action`、`attach_adapters` | 按共同随机初值构造，不恢复v9/v10b训练状态；先attach再创建optimizer |
| 数据 | 保留v6 `Data.window/loss/metrics/motion_weight` 数学语义 | 新 `BoundData(codec_binding)` 显式加载准入state/raw records，不执行旧 `Data.__init__` |
| native bridge | 沿用 **`native_experiment.NativeBridge`** 的 `codec.eval`、`prediction_gs`、`render`及部分解冻语义 | 新 `BoundNativeBridge(binding, partial=True)` 用明确checkpoint参数替代STATE_ROOT硬编码 |
| RGB/teacher损失 | v10b `teacher_loss`、`render_views`可按SHA加载调用 | 新的薄 `step_native` 接受依赖和action_mode，保留原权重与计算顺序，改正日志/实际code处理 |
| 分段反向 | 直接复用v7 `temporal_step.staged_temporal_backward` | 用新adapter小图验证梯度等价；保留所有未来delta对初始render的导数 |
| 计划/控制 | v7 `make_plan(seed,steps)`；固定168窗供体公式 | 新计划文件及code映射，不能带入旧400/600/800续训来源约束 |
| 保存/评测 | 现有adapter actual-code packet；原native gs/sh/camera包形式 | 新外层来源字段、两模式实际码回放、完整控制和最终codec身份 |

两个易错点：

- `gs_bridge.py:NativeGSBridge` 是另一条带 `SoftOccupancyTracker`、软opacity gate且默认全codec可训练的路径；本轮不引入它。v10b真正使用的是v6 `native_experiment.py:NativeBridge`，其普通 `decode_state` 路径仅允许 decoder.out / feature_heads / gaussian_multiplier 部分解冻。
- v10b `loss_step.step_native` 最后在 **optimizer.step之后** 执行 `encoded['indices'].detach()`。continuous的indices刻意为None，原样调用会先更新再报错；不能用捕获异常、伪填diagnostic IDs或补个日志wrapper解决。应在新薄step中明确保存actual codes，调用原teacher/render原语和原staged backward，去掉对 `native_experiment` 全局runtime及indices-only日志的依赖。

## 2. 已核对的源码SHA

以下为本地实际文件SHA256，供新入口按字节绑定；不是待选codec的验收声明。路径均相对 `experiments/`。

| 来源文件 | SHA256 |
|---|---|
| `gaussiangpt_neoverse_z_ablation_20261031_v1/model_adapter.py` | `112bf3cc14ee7517f8052e06467c0a8674e00a6e65b108fbbb1017c6da163662` |
| 同目录 `action_bottleneck.py` | `4111735d1575b5fd0e71046b8d38e2f4191797e898cb417911bd6a4da44dfb20` |
| 同目录 `initial_norm.py` | `21e1cc366aebe48ae78fc5f1460c3ed765095157d655854d1c57f9a9b8e90b80` |
| `gaussiangpt_neoverse_gs_supervision_20261031_v10b/loss_step.py` | `ce304e2d861e303f1780c1267cf05146e7d5d4a13f8b30f54b28368608ea16ad` |
| 同目录 `teacher_assets.py` | `c79f11360eb1d1ad7fa935c7cdf101b6b0993e9f71d15933b57b9f39cc2dfc8b` |
| 同目录 `worker.py`（参考，不直接调其runtime） | `fb189f9ff654962323621c3d5393bd94b27adea355cfbf48070e088c1a881f2f` |
| `gaussiangpt_neoverse_spatial_action_20261031_v9/spatial_action.py` | `16d7e315eea1ab4f818359c079017b1dc9f7319133d98dca0a847e0a3d82539f` |
| `gaussiangpt_neoverse_temporal_z_20261031_v7/temporal_step.py` | `6a1de737b3e87d88c6678352236258c031f9e8d66f17d79a04e2b944560562ef` |
| 同目录 `runner.py`（只借用计划/数学原语） | `f039edcb98d82f4f2de3545b0b8b8aa17b8bb71313aa43d7775413ab96c609ac` |
| `gaussiangpt_neoverse_native_z_20261031_v6/code/native_experiment.py` | `375b4e3c7b7e8ed0f7e0a8d3a0e69ba0e39bc41459980ebb81ef05cd8837c331` |
| 同目录 `gs_bridge.py`（硬编码审计依据） | `84cb0b52f1e727190b9471baef3b979f53d08aa5be453ea99568ac88b77af359` |
| 同目录 `sequence_data.py` | `d61890e66d7575b2296e14c8c71abcb946d32519649992a1183bd7fd3ea314a5` |
| `gaussiangpt_neoverse_state_motion_20261030_v1/state/code/codec.py` | `e5eb7b948a70e9abea5fc0428ba7c37a35a71a17b5a49ed5da0578df3a719a2a` |
| 同目录 `common.py`（prediction_gs / native renderer） | `38a83d114718f42ed81989b5f770db323e18949a444da418a22184df847b66ff` |
| `gaussiangpt_neoverse_step_acceleration_20261031_v2/null_key_bias.py` | `7ef41a420de732a11f6938303f18bfc69cc7655c85b2ad870a1105a1585f83fb` |

模型内部的spatial64/model_core/upstream/aligned_ldm/multiscale64/anchored_ldm及vendor源，直接复用 `model_adapter.PINNED/VENDOR` 的完整绑定，避免再维护一份不一致清单。真实LFQ库SHA仍是 `9fb68bdcb3763ac580c2a0f931640a70a14a0fed65e0b29a8800ee015b4452eb`；不安装、替换或mock它。应在干净进程先调用 `load_pinned`，不能先导入旧native_experiment造成未标记同名模块碰撞。

## 3. codec_binding、初始state与native GS接口

新 `codec_binding.py` 最小提供三个对象/函数即可：

1. `load_binding(path)`：返回明确选定的codec checkpoint、架构/renderer源码、state manifest和raw/teacher来源；缺少完整准入结果或选定checkpoint就拒绝构造训练runtime。文件“存在”或 `quality_evaluation_complete=true` 不是codec验收；须分别绑定latent replay、质量验收与native包独立重放证据。若最终选择包含ME解码设置或修正，其源码/设置也作为binding输入，两臂与评测一致；本计划不先指定开关。
2. `BoundData(binding, clip)`：校验177个state与raw frame的clip、frame、PTS、normalization、大小/SHA、encoder身份，暴露 `states/records/roi/teacher_targets/teacher_masks/window/loss/metrics`。允许CPU存储、按B4窗口搬运；不要求像旧runtime一样一次将全部RGB/teacher常驻GPU。
3. `BoundNativeBridge(binding, partial=True)`：用 `load_codec(architecture_root, checkpoint_path, device)` 加载选定权重；所有codec先冻结并保持eval，再仅解冻 `ae.decoder.out`、`ae.feature_heads`、`ae.gaussian_multiplier`。暴露 `codec/prediction_gs/render_fn/last_count/render`，满足旧render_views的鸭子类型接口。encoder检查覆盖其参数和buffers、embedding及set_encoder，不只检查 `ae.encoder` 一个子模块。

Binding至少保存：选定checkpoint路径/文件SHA、encoder参数+buffer摘要、decoder参数+buffer摘要、codec selection/protocol/source SHA、grid=.008/stride=4/Identity VQ；每帧state path/SHA及state producer checkpoint、source_cache SHA、原始PTS/normalization；raw GS/cameras/intrinsics/real/teacher来源；独立验收收据SHA；运行时最终codec另记SHA。不能用未来动作成绩挑codec。

**初态不是一个训练好的LDM checkpoint**：每窗S0来自该窗口start对应的准入state。新fullpool state packet将pts/clip/frame放在metadata中，而旧 `sequence_data._validate_state` 要求顶层pts。因此新增只读模型输入视图，从已核对的metadata提升pts/frame_index/clip，不改文件、不改变coordinates/features顺序，不伪造时间，也不把该视图SHA冒充原packetSHA。`make_condition(S0)`及 `encode_initial(S0)` 只接收该初态；真实未来state仅进入各h允许的posterior前缀。

真实RGB、相机、teacher RGB和alpha mask与encoder无关，可以在**raw GS、normalization、相机、renderer、帧身份逐项一致**时只读复用，但保留旧producer身份。v10b旧候选来源为：`state/prepare/buildingblock_tower_17_176.json` SHA `7b63c8c50eb9a3e7f3e2f4b55576894f272c6c0f72de3502b6b7636c56dd702a`，raw cache为同clip177帧，native原始GS为 `neoverse_tower177_fps30_20261029_v1/native448/gaussians/`。本次没有读取远端teacher包，故其实际包SHA留待来源绑定，不编造新资产身份。若不相同，只重建受影响teacher资产；不能从模型预测生成有效区mask。

## 4. 共同初值、初态4槽norm与同一scale

按一个固定共同seed构造 `AnchoredLDM(blocks=4, checkpoint_blocks=拟冻结设置, residual_scale=3, inv_temperature=1, code_mode='posterior')`；用准入TRAIN frame0 `[M0,64]` features初始化固定64通道output_std；attach v9 spatial分支，再 `attach_adapters`。训练前的checkpoint_blocks设置两臂一致，并须以完整B4T10资源见证决定，不能运行中单臂切换。

初态参考norm已实现：四个decoder块各替换两个temporal norm及一个condition norm；仅用 `x[:4]` 的活跃行/组通道求统计，affine仅池化 `condition[:4]`。原weight/bias和eps保留。不是全13槽统计，也不是在线批次或未来目标统计。新contexts由单独 `encode_initial(S0)` 构造，posterior仍读取S0…Sh。

校准仅在共同初始化的eval/no_grad下，以准入TRAIN全部168窗、完整9h和16通道计算一个 `s_ref=sqrt(mean(u²))`，`u=10*tanh(P_in(z)/10)`；共24,192个标量，按固定42个B4批流式累计。记录starts/state manifest/encoder/common-model SHA、RMS/元素数，要求finite且>1e-6，只绑定一次。不能按臂、batch、h或评测结果重估。然后 `clone_pair(common)`，两臂state_dict逐值相同但Parameter对象独立；保存一个common_init与共同RNG，mode作为外层metadata保存。

两个fresh AdamW从空状态开始；motion/空间参数lr3e-4、native部分参数lr1e-5，betas(.9,.95)、wd0、clip1。空间参数在创建optimizer前已attach，不执行v9为续训设计的append旧optimizer逻辑。沿用现有27个softmax无效key-bias固定策略：先建完整optimizer，再对两臂同样调用已绑定 `freeze_null_key_bias`，保留成员、无旧moments；不要把这些解析无效方向误列为必须非零的重建梯度。

两臂同为64→16→64无bias投影；continuous使用v=u/s_ref，LFQ使用sign(v)与既定STE。两臂都保留同一真实LFQ per-h B4辅助熵，库内权重.1、外部aux_multiplier=.1、温度1；不是36-token混合熵，也不是只给LFQ额外正则。预检两臂至少各2步后丢弃，正式训练重新加载共同初值/RNG；第1步零head导致上游重建梯度为零是已知行为，第2步验证各样本×h、QFormer、投影及native/空间组的有效梯度。

## 5. 拟定800步B4计划与real/teacher监督

候选计划保持 `v7.make_plan(20261031, 800)`，内部实际 `random.Random(seed+17)`。本次仅在CPU用AST取出该原函数及plan_hash运行，结果：

- 800更新，每步4个完整T10窗口；每臂3200窗口曝光、32000帧曝光、28800未来曝光。
- 19个完整168-start周期占798步，另2步含8个start；160个start各19次、8个各20次。不能把800步称为整数个完整周期。
- 候选plan SHA：`d80f965a9fbd7c4906327691e0a671488b68d5c8d32f62a8842107163c58c984`；首批 `[116,77,61,94]`，末批 `[27,117,67,155]`。这不是正式冻结或训练证据。

两臂使用同一完整plan，正式计数从0至800；没有v10b“600+200”继承。只评估固定最终800，不根据44/132帧挑checkpoint。时间/显存上限待真实B4T10资源测量后统一冻结，不能临时减B、T、点数或448×252分辨率。

损失按v10b `gs_z` recipe共同固定，不能误搬fullpool codec的六项重建loss：

- 每窗10帧real full L1 /10 + fixed hand_activity ROI L1 /10；h1…9 motion-weighted L1 /9，权重仅由真实RGB差分构造、detach，禁止进入模型。
- h1…9 delta loss比较 `(prediction_h-prediction_0)` 与 `(real_h-real_0)`，full+ROI，delta_weight=1；保留对prediction_0的完整导数。
- 每帧teacher两view权重是 `[0,.25]`，新增监督实际只在nearby view固定alpha有效区；alpha≥.90并3×3 erosion，按10帧平均。real view仍有真实RGB监督，不能错误解释为两view各.25。
- staged backward内部统一除B，并仅对delta另除9；辅助熵在一次重建VJP之后加入一次；一批40帧完成后统一clip/step。新薄step不能再次除B/T造成双重缩放。

## 6. 完整动作控制、实际码与参考

每臂最终checkpoint提取全部168窗actual float32 `[168,9,16]`；LFQ同时存实际IDs和±1码，continuous的indices为None，diagnostic_ids只能诊断。外层包加入mode、s_ref、start/PTS、状态manifest、model/最终codec SHA、adapter SHA及包SHA。评测先读回实际codes，再 `decode_action_codes`；不能再次跑posterior冒充动作包重放。

| 控制 | 确定映射 / 含义 |
|---|---|
| correct | 接收方start自己的9码束 |
| fixed_bundle | 所有窗使用本臂start=0完整9码束，h顺序不变 |
| fixed_single | 本臂start=0,h=1实际码广播9h及全部窗；continuous不改用零向量 |
| shuffle42/84/126 | 供体 `(start+offset)%168` 的完整码束；接收方S0、PTS、相机不变 |
| blinded_posterior | 重做posterior：各未来槽coordinates/features替换为S0副本，保留各槽时间/frame identity；不能保留真实未来坐标并集，也不能复用原posterior输出 |
| adapted_held | 本臂最终codec解码真实S0，在各目标相机保持初态；不走动作模型 |
| decoded_actual_target_state | 本臂最终codec解码真实S_h；显式“真实未来codec参考”，不是预测或Z成绩 |

先评估全部168窗×9未来=1512曝光，h0另168行，覆盖176个不同未来帧；按物理帧平均重叠窗口后做配对统计。保留每h、full/hand ROI/motion/delta和固定object_box `[140,25,320,235]`。沿用设计中的block16/2000次/seed20261031配对bootstrap，不把ROI或矩形叫语义物体mask。

完整记录s35→44与s123→132两个T10窗、44/132两view以及黄条/绿梁方向、位置、形状、颜色和重影。至少correct/shuffle84/fixed_bundle/真实未来参考保存两窗全部native gs/sh/cameras包，并在独立进程只读包+renderer回放。新codec部分解冻后必须用各臂最终codec身份重新验证latent/native replay；旧codec准入不能代替这一步。

## 7. 不可直接复用的旧内容

- v10b `runtime/restore600/restore800`、v9 `restore400/restore600`、v7 `restore_source`，以及旧LDM/optimizer/RNG和旧constant臂成绩：都带旧训练史与固定来源，不属于共同随机初值主比较。
- v6 `STATE_ROOT/MOTION_ROOT/V4_ROOT`、`verify_native_sources()` 固定的codec_final.pt、V4 cache_manifest/source_binding；旧 `Data` 强制读取177旧motion states和354源文件及旧reference指标。这些构造函数须替换，不能修改模块全局变量后继续写旧binding。
- 新encoder下全部旧state、output_std和action-scale校准失效。即使仅换decoder，旧state复用也须encoder参数+buffers及坐标/features语义见证并保留producer；本方案默认新codec完整评测生成的fresh state。
- 旧state packet常含顶层pts，新fullpool packet通常只在metadata中有pts；不能直接调用旧loader或用frame/fps补造时间。旧 normalization=.5不能外推到其他片段。
- v10b/v7只保存IDs、只支持decode_ids的动作包与评价分支，旧fixed/constant模型、旧 `held_initial/frozen_target` 指标缓存，均不能当作本轮continuous控制或最终codec参考。真实未来参考必须现算。
- 旧teacher mask可能可复用其数值，但旧run的TEACHER_ASSETS收据不能改名充当新来源；必须单列producer SHA与当前raw/camera/renderer兼容检查。

## 8. 可立即做的最小任务清单

1. **`codec_binding.py` + CPU单测**：定义明确的未选择/已准入状态；实现收据/SHA/帧身份校验和metadata→模型state视图；测试错encoder、错decoder来源、缺帧、重复坐标、非单调PTS、错误mode均拒绝。先用合成CPU包测接口，不加载候选codec。
2. **`runner_core.py` + CPU单测**：实现共同计划、数据loss原语、actual-code外层保存/读取、全部控制映射、覆盖计数。直接用当前候选plan hash作待冻结一致性见证；检查整束搬移、无shuffle自供体、fixed_single真正跨h恒定、blinded无未来几何残留。
3. **新薄 `loss_step.py` + CPU梯度见证**：以真实adapter小图比较staged与完整反向，含B平均、双view teacher、delta对初始render导数和一次aux；验证continuous不访问indices，LFQ实际ID/sign吻合，零head第1/2步行为与两个模式日志均可写。图像替身只用于链式求导测试，不称原生renderer/GPU成功。
4. **`worker.py` / `evaluate.py` 骨架**：只接受显式binding/common_init/plan；保存model+最终codec+optimizer+安全格式RNG/游标，mode外层强校验；两臂独立目录、异常保留失败、没有自动续训/替换底座。先写无准入binding即拒绝的CPU测试，暂不启动。
5. **codec确定后补齐唯一实际输入**：绑定最终codec、177 fresh states与可复用或新建teacher；共同校准s_ref/output_std、保存共同初值，执行两个模式实际完整B4T10两步丢弃见证；据实冻结800步及资源上限，再进行一次配对正式训练和全控制评测。

已有adapter的9项真实CPU小图见证可复用其明确覆盖范围；不重做整套治理流程，不增加第三训练臂、维度搜索、额外来源或AR。当前缺的是准入codec输入和薄runner的上述具体接口，不能靠旧成功日志补齐。
