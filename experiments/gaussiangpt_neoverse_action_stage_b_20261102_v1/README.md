# 阶段 B：共享 decoder 与连续自由码

阶段 A 的最终动态审查发现关键物体仍消失，按计划从曾在两个窗口成功的 decoder 出发，联合训练一个共享运动 decoder 和 168 组窗口自由码 `[168,9,16]`；原 GaussianGPT codec、初态编码器与 context 冻结，没有 LFQ、FSQ 或 AR。

固定计划为 20 轮、每轮覆盖 168×9 个窗口／未来步，逻辑 batch size 4，共 7,560 次更新；第 5、10、20 轮保存成对的 decoder 与码，并在训练完成后评估、独立重放。结果仍然只是单条训练视频上的自由码教师候选，不能当作已训练的 Q-Former 或跨视频泛化。

2026-11-02 服务器启动验收：真实 GPU 预检通过，图像损失向自由码与 decoder 的梯度非零；预检产生的更新已撤销，优化器和码已恢复，原 codec 与初态编码器仍冻结。启动记录中训练已达 25／7,560 步；请勿将此数当作当前实时步数。未完成训练、评估与视觉验收。

`CONFIG.json`、`BINDING.json`、`PREFLIGHT.json` 记录预算、阶段 A 审查和训练源哈希，不含模型权重或数据。训练代码见相邻 `gaussiangpt_neoverse_action_stage_b_prepare_20261102_v1`，自动评估与重放见 `gaussiangpt_neoverse_action_stage_b_orchestration_20261102_v1`。
