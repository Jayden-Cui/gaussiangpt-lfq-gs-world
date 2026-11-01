# 连续动作码进展（2026-11-02）

目标是在给定初始 GS 状态时，从真实未来状态提取紧凑的连续变化码，再恢复主要物体的空间运动。当前数据为一段 NeoVerse 搭积木视频的 177 个相邻帧，形成 168 个重叠的 10 帧窗口，每窗口码为 `[9,16]`。当前实验不使用 LFQ、FSQ 或 AR；自由码直接利用真实未来优化，不是在线预测。

## 已完成：阶段 A

固定 9560 步 decoder 和原 codec，仅优化全片段自由码。两个分片各完成 3,780 次更新；第 5、10、20 轮全片段渲染及各 104 份 GS 包、208 个视角的独立重放均完成。最终自由码运动 MAE 为 0.060828，原 Q 码为 0.062531，真实未来状态 codec 参考为 0.029464（176 个物理未来帧等权）。

[最终动态审查与视频](experiments/gaussiangpt_neoverse_action_stage_a_eval_correction_20261102_v3/EXECUTION.md)：黄色条拿起时仍变细或消失，绿色梁搬运时基本消失；小运动和静止结构多能保留。由此按计划进入阶段 B，不能据此断言固定 decoder 的理论容量不足。[审查记录](experiments/gaussiangpt_neoverse_action_stage_a_eval_correction_20261102_v3/REVIEW_ASSETS/STAGE_A_REVIEW.json)。

原评估器重复初态条件曾错误复用自由码输出；v2 修正程序又在汇总时报索引错误，失败记录已保留。独立 v3 完成逐条修正、完整三检查点汇总测试及视觉审查。修正后原 Q 的重复初态运动 MAE 为 0.068632，是同一模型的消融，不是独立训练的基线。

[占据替换诊断](experiments/gaussiangpt_neoverse_action_stage_a_eval_correction_20261102_v2/EXECUTION.md) 仅用真实目标格点选择预测特征做离线定位；不能作为真实预测输入。[第 5／10 轮结果和八窗口观察](experiments/gaussiangpt_neoverse_stage_a_interim_render_20261102_v1/EXECUTION.md) 是中间检查。

## 正在训练：阶段 B

[共享 decoder＋全窗口自由码](experiments/gaussiangpt_neoverse_action_stage_b_20261102_v1/README.md) 从原两窗口成功的模型出发，固定原 codec、初态编码和 context，训练共享 decoder 与 168 组自由码。真实 GPU 预检已通过；启动验收时已完成 25／7,560 次更新。训练完成后自动评估并独立重放第 5、10、20 轮；最终结果仍需逐窗审查。[两窗口起始模型独立回归](experiments/gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1/EXECUTION.md) 的运动 MAE 为 0.02977625250，与旧结果 0.02977625270 一致；另有事件监听器将在第 5、10 轮产生后预览这两个窗口。

## 后续阶段

- [阶段 C 准备](experiments/gaussiangpt_neoverse_stage_c_training_prepare_20261102_v1/ENTRYPOINT_V2.md)：如果阶段 B 得到可靠教师，冻结同一个 decoder，对比冻结 memory、可训练后验时空编码器，以及显式时间／变化 memory；[C0/C1/C2 真实 GPU 梯度／显存预检](experiments/gaussiangpt_neoverse_stage_c_gpu_preflight_20261102_v1/EXECUTION.md) 已通过；预检使用合成目标并撤销更新，正式教师和训练尚未完成。
- [同初态运动／静止评估](experiments/gaussiangpt_neoverse_stage_bc_evaluation_prepare_20261102_v1/DRIVER_V4_NOTES.md)：接口已准备，尚无真实模型验收结果。
- [跨视频数据检查](experiments/gaussiangpt_neoverse_multisource_action_data_20261102_v1/CROSS_SOURCE_NOTES.md)：第二训练源的部分未来格点超出原半径 4；依据训练源选择半径 6 后三段候选节点数均低于 25,000，但完整跨视频训练的 GPU 显存和效果尚未验证。第三段曾被观察，不是从未接触过的最终测试集。[两训练源均衡调度与按物理帧聚合](experiments/gaussiangpt_neoverse_multisource_sampling_20261102_v1/EXECUTION.md) 已通过 CPU 和真实数据检查，尚未启动跨视频动作训练。 [第二训练源 GPU 单窗口检查](experiments/gaussiangpt_neoverse_multisource_gpu_probe_20261102_v1/EXECUTION.md) 已通过半径 6 与码梯度预检；[同帧真实状态 codec 参考](experiments/gaussiangpt_neoverse_multisource_codec_reference_20261102_v2/EXECUTION.md) 显示原 codec 已出现颜色错置。此为单窗口动作路径诊断，不是总体跨视频性能。[两训练视频各 177 帧的状态 codec 记录审计](experiments/gaussiangpt_neoverse_multisource_codec_train_audit_20261102_v1/EXECUTION.md) 显示平均图像 MAE 分别为 0.016599 和 0.079611；已有 [decoder-only 状态 codec 对照](experiments/gaussiangpt_neoverse_multisource_codec_reuse_audit_20261102_v1/EXECUTION.md)：第二视频 MAE 降至 0.024533，当前视频升至 0.017179；其编码器侧 155 个张量与原版完全相同。[相同阶段 A 预测状态的 decoder 交换检查](experiments/gaussiangpt_neoverse_motion_state_decoder_swap_20261102_v1/EXECUTION.md) 中，两窗口运动 MAE 从 0.081061 升至 0.087234，关键动作仍未恢复；该结论仅覆盖 18 个已有预测。当前阶段 B 不切换。

仓库不含原始视频、GS 缓存、训练权重与服务器环境；代码中的绝对路径和哈希依赖原 134 服务器。

补充：对[此前成功的两窗口自由码预测状态](experiments/gaussiangpt_neoverse_motion_state_decoder_swap_warm_20261102_v1/EXECUTION.md)逐帧交换 codec decoder，黄条和绿梁的粗运动在两种 decoder 下均可辨认，但 18 帧运动区域 MAE 从 0.029776 上升至 0.039747。该检查未进行优化器更新或独立 GS 冷重放，仅覆盖两个已知训练窗口，不能外推为全片段兼容或跨源泛化。

另已根据源画面变化冻结[四组窗口评估分层](experiments/gaussiangpt_neoverse_motion_groups_20261102_v1/EXECUTION.md)：每组 42 窗口。修正后的阶段 A 最终自由码，在最高变化组的运动 MAE 为 0.081317，最低变化组为 0.042979；两个已知关键窗口位于最高变化组。分组在 B 训练期间建立，包含手部和遮挡变化，不等同物体跟踪或动作成功率，待 B 完整评估后复用。

[CPU 分组监听器](experiments/gaussiangpt_neoverse_motion_groups_watch_20261102_v1/EXECUTION.md)已启动，等待 B 的三个完整评估与独立重放后自动生成分组统计；教师选择仍需动态审查，不会自动声称动作码已完成。

阶段 B 第 5 轮[两窗口预览与逐帧审查](experiments/gaussiangpt_neoverse_stage_b_sweep05_review_20261102_v1/EXECUTION.md)：18 帧运动 MAE 从起点 0.029776 升至 0.086623；黄条出现细弱、缺失与残影，绿梁难以辨认。该中途预览未覆盖全部窗口或执行独立重放；原 20 轮训练继续，已完成不更新模型的起点／第 5 轮 decoder 与自由码交叉诊断。

[交叉诊断已完成](experiments/gaussiangpt_neoverse_stage_b_cross_decode_20261102_v1/EXECUTION.md)：起点 decoder＋第 5 轮码为 0.031984，第 5 轮 decoder＋起点码为 0.094129；两种原配分别为 0.029776 和 0.086623。图像和指标均表明，两个已知窗口的阶段性回归与 decoder 变化高度相关；交叉组合可能不在训练分布内，不代表全片段的最终结果。

[全部 168 窗口的 h9 交叉诊断](experiments/gaussiangpt_neoverse_stage_b_cross_h9_20261102_v1/EXECUTION.md)已完成：同用第 5 轮码时，起点／第 5 轮 decoder 的运动 MAE 为 0.196871／0.084201；第 5 轮 decoder 在 166 窗口误差更低，但两个例外正是此前成功窗口 35 和 123。当前是全片段适配与锚点保留的权衡，不能直接冻结起点 decoder；这仍只是一条视频的 h9 离线诊断。

已完成[关键窗口重放的两组匹配小试](experiments/gaussiangpt_neoverse_stage_b_anchor_replay_pilot_20261102_v1/RESULTS.md)：两锚点运动 MAE 从 0.083637 降至 0.071302，高变化组 h9 基本不变；但绿梁在重放组连续九帧未被颜色代理检出，逐帧画面也未恢复完整绿梁，因而不能选作教师。原 B 训练不受该小试改变。

[历史两窗口 Q-Former 的同相机静止控制](experiments/gaussiangpt_neoverse_qformer_static_control_20261102_v1/RESULTS.md)已完成：重复 S0 码对静止参考的误差为 0.159765，对运动参考为 0.029009；18 帧均更接近运动参考，画面也显示物体继续运动。该旧 Q＋decoder 系统没有通过同初态静止控制，但不足以将原因单独归咎 Q 或 decoder。

阶段 B 第 10 轮[两窗口预览与逐帧审查](experiments/gaussiangpt_neoverse_stage_b_anchor_preview_20261102_v1/sweep_10/VISUAL_REVIEW.md)的运动 MAE 为 0.082714，较第 5 轮稍有改善，但仍高于起点 0.029776；黄条细弱且位置偏移，绿梁多帧消失。两个窗口的动态验收仍未通过，完整评估和第 20 轮尚未完成。

为区分旧 decoder 是否能表达静止、以及旧 Q 是否能产生静止码，完成了[两窗口固定 decoder 静止自由码诊断 v3](experiments/gaussiangpt_neoverse_static_code_probe_20261102_v3/RESULTS.md)：只优化 540 次连续码，静止参考误差从 0.159765 降至 0.131698，但 15／18 帧仍更接近运动参考；逐帧画面中物体消失且出现黑色残影，没有得到可用静止码。这是单初始化、有限预算的两窗口结果，不能证明数学上不存在静止码。v1 在 GPU 申请前因 JSON 记录格式失败；v2 的 GPU 预检通过，但第 0 步评估读取配置错误，均在正式训练前退出并已保留。

阶段 C 新增[固定教师的源画面变化分组评估工具](experiments/gaussiangpt_neoverse_stage_c_motion_groups_20261102_v1/README.md)：逐条核对 1512 项教师指标，在四组中比较后验与教师的物理帧等权及 h1–h9 误差。本地与服务器测试使用合成 C／教师指标，尚未评估真实 C 模型；全画面变化分组不等于物体轨迹或动作成功。

[两窗口同初态运动／静止共享 decoder 诊断](experiments/gaussiangpt_neoverse_two_future_shared_decoder_20261102_v1/RESULTS.md)已完成 360 次成对更新：运动分支误差从 0.029776 至 0.031369，静止分支误差从 0.131698 降至 0.007583，18／18 帧更接近静止参考；逐帧画面保留了两物体的运动／静止差异，但黄条运动中心代理只有 8／9 帧在 10% 范围，未满足预登记 95% 比例。运动和静止分支各有自由码，共用初态与 decoder；这不能替代全 168 窗口教师、正式 Q-Former 或跨视频验收。

针对该小试的[三栏视频与逐帧审查素材](experiments/gaussiangpt_neoverse_two_future_visual_20261102_v2/result/MANIFEST.json)已生成并检查；使用 CPU 和进程退出事件，不影响训练。v1 来源哈希抄写错误在启动前发现并已留档，v2 改为读取原启动记录的完整哈希。[独立 GS 重放](experiments/gaussiangpt_neoverse_two_future_native_replay_20261102_v2/EXECUTION.md)的 36 份运动／静止 GS 包、72 视角全部通过；该一致性不证明动作语义正确。

在上述两窗口自由码教师上，[固定 decoder 的双未来 Q-Former 诊断](experiments/gaussiangpt_neoverse_two_future_qformer_20261102_v1/EXECUTION.md)已完成 600 次蒸馏并进入 594 次重建联合更新。蒸馏结束时 Q 运动误差 0.031389、教师 0.031369；静止参考误差分别为 0.007585 和 0.007583，两分支各 18／18 帧更接近对应参考。[独立 CPU 重算](experiments/gaussiangpt_neoverse_qformer_step600_cpu_replay_20261102_v1/RESULTS.md)验证了四组码及缓存前缀扰动检查；最终原始状态因果性、GS 冷重放和视觉审查尚待完成。该教师有单帧颜色代理偏差，此试验只检验两窗口提取，不是正式全片段 C 对照。

该双未来诊断的[第 900 步中期评估](experiments/gaussiangpt_neoverse_two_future_qformer_20261102_v1/result/step_00900/EVALUATION.json)中，Q 与教师相对原视频的运动区域 MAE 分别为 0.031060 和 0.031369，静止参考加权 MAE 分别为 0.007091 和 0.007583；两分支各 18／18 帧更接近对应参考。训练和最终视觉、因果及独立重放检查仍在进行，不能从两窗口中期数据推断全片段或跨视频表现。

[两窗口冻结前缀 memory 输入审计](experiments/gaussiangpt_neoverse_paired_prefix_memory_audit_20261102_v1/RESULTS.md)确认：真实运动前缀和重复初态前缀在相邻帧即有世界格点和 64 维特征差异；远期前缀差异更大。这不证明 Q 已学会读取动作，也不能把网格变化直接解释为物体位移。
