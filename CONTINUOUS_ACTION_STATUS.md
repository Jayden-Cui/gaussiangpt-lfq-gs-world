# 连续动作码进展（2026-11-02）

目标是在给定初始 GS 状态时，从真实未来状态提取紧凑的连续变化码，再恢复主要物体的空间运动。当前数据是 NeoVerse 搭积木视频的 177 个相邻帧，形成 168 个重叠的 10 帧窗口；每个窗口的动作码为 `[9,16]`，没有 LFQ、FSQ 或 AR。这里的后验输入包含真实未来状态，不是在线未来预测。

## 正在执行：阶段 A

[阶段 A](experiments/gaussiangpt_neoverse_action_stage_a_20261102_v1/README.md) 冻结原 GaussianGPT 状态 codec 和 9560 步运动 decoder，只优化 168 组自由连续码。两个分片各覆盖 84 个窗口、20 轮，计划各 3780 次 B4 更新，固定在第 5、10、20 轮评估。训练和最终动态验收尚未完成。

两分片均已保存第 10 轮检查点，即各至少完成 1890 次更新。[两个关键窗口的 CPU 中间检查](experiments/gaussiangpt_neoverse_stage_a_interim_latent_20261102_v1/EXECUTION.md) 显示第 10 轮潜在特征误差与原 Q-Former 基本持平，占据 IoU 略低；[全部 168 个窗口的码均已变化](experiments/gaussiangpt_neoverse_stage_a_interim_latent_20261102_v1/CODE_DRIFT.md)。这不包含图像渲染，不能据此判断动作是否恢复。

[第 20 轮早期 CPU 汇总](experiments/gaussiangpt_neoverse_stage_a_final_early_20261102_v1/EXECUTION.md) 已完成真实第 10 轮的对照测试并在等待两个训练分片自然结束；它不替代渲染和动态验收。

[前 10 轮训练曲线](experiments/gaussiangpt_neoverse_stage_a_interim_latent_20261102_v1/TRAIN_PREFIX_ANALYSIS.md) 中，组合图像损失从第 1 轮的 0.132495 降至第 10 轮的 0.130169；潜在状态损失基本持平。该组合值不是单独的图像 MAE，也不是固定模型评估。

已完成的离线占据诊断覆盖 1512 个窗口预测：按 176 个物理未来帧等权，原预测支撑、使用真实目标格点选择预测特征、真实未来状态 codec 参考的运动 MAE 分别为 0.062531、0.057931、0.029464。真实目标格点只用于定位，不进入训练或实际预测。绿梁末帧的颜色区域代理 IoU 分别为 0、0.043132、0.904151；该代理不是物体追踪或动作成功率。[诊断记录与数据](experiments/gaussiangpt_neoverse_action_stage_a_eval_correction_20261102_v2/EXECUTION.md)。

原阶段 A 评估器发现重复初态对照误复用正确码输出；自由码优化及其他条件不受该分支影响。独立修正程序已安排在原任务结束后逐条修正对照并保留证据，最终画面仍须审查。[错误与修正说明](experiments/gaussiangpt_neoverse_action_stage_a_eval_correction_20261102_v2/README.md)。

[黄条与绿梁的诊断视频、首末帧](experiments/gaussiangpt_neoverse_action_stage_a_eval_correction_20261102_v2/diagnostic_media/) 左侧为原视频，右侧后九帧使用真实未来格点，仅供定位。视频以 5 fps 慢放，源视频约 29.97 fps。

## 后续模块和数据检查

- [阶段 B 准备](experiments/gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1/ENTRYPOINT_V3.md)：如果阶段 A 自由码仍不能恢复主要动态，从两窗口成功的 decoder 出发，联合训练全片段自由码和共享 decoder。168 个窗口的连续初始码已在 CPU 上准备并检查；正式训练与 GPU 预检尚未启动。
- [阶段 B 条件执行流程](experiments/gaussiangpt_neoverse_action_stage_b_orchestration_20261102_v1/EXECUTION.md)：训练完成后自动评估三个固定检查点并独立重放；服务器 CPU 模拟与草案拒绝测试通过，未启动训练。
- [阶段 C 准备](experiments/gaussiangpt_neoverse_stage_c_training_prepare_20261102_v1/ENTRYPOINT_V2.md)：对同一经审查的教师比较冻结 memory、可训练后验时空编码器以及显式时间／变化 memory。组件与 CPU 功能检查已完成，正式教师、GPU 训练和评估尚未完成。
- [运动／静止与后验评估准备](experiments/gaussiangpt_neoverse_stage_bc_evaluation_prepare_20261102_v1/DRIVER_V4_NOTES.md)：同初态、同相机比较两种未来，并检查因果前缀与教师相对误差；尚无真实模型验收结果。
- [多视频数据与候选范围](experiments/gaussiangpt_neoverse_multisource_action_data_20261102_v1/CROSS_SOURCE_NOTES.md)：三段视频各 177 帧。原半径 4 在第二段训练视频的 12 个窗口有目标格点超出范围；依据训练数据检查半径 6 后，三段全部 504 个初态候选集合均低于现有 25,000 节点上限。GPU 显存和跨视频动作效果尚未验证，改变半径后不能沿用旧教师指标。第三段视频曾被历史实验观察，不是未接触过的最终测试集。

仓库不含原始视频、GS 缓存、训练权重与服务器环境；代码中的绝对路径和哈希依赖原 134 服务器。详见根目录说明和各实验的 `EXECUTION.md`。
