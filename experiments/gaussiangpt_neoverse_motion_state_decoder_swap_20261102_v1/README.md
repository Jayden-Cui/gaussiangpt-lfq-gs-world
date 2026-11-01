# 相同动作预测状态的 codec decoder 对照

读取阶段 A 最终保存的两个关键窗口、各九个未来步的预测稀疏状态；对完全相同的坐标、64 维特征和相机，分别使用原 codec 与已有 decoder-only codec 渲染。该操作不重新提取动作码、不更新任何模型，也不改变正在进行的阶段 B。

原 codec 的重新渲染必须逐帧复现阶段 A 原有图像和运动指标；再载入已核对编码器侧 155 个张量相同的 decoder-only checkpoint，检查同一预测状态被不同 native decoder 如何解释。视频每帧从左到右依次为源视频、原 codec、decoder-only codec，无顶部标记，以 5 fps 慢放。

结果仅覆盖一段训练视频的两个已知窗口和 18 个预测，不能证明完整 168 窗口、第二视频或未来 Q-Former 的兼容性。本次不对 decoder-only 输出执行新的独立 GS 包冷重放，也不会自动切换生产 codec。
