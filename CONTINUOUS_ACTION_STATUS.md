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
