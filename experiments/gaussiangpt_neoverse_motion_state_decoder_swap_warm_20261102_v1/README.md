# 已成功两窗口动作状态的 codec decoder 对照

读取阶段 B 启动时使用的两窗口成功自由码教师保存的 18 个未来状态；对相同坐标、64 维特征与相机，比较原 codec 和已有 decoder-only codec。不更新任何模型，不改变运行中的阶段 B。

原 codec 必须逐帧复现历史成功结果后，才比较 decoder-only 对相同状态的解释。两段视频从左至右为源视频、原 codec、decoder-only codec，无顶部标记，5 fps 慢放。该检查只覆盖两段已知训练窗口，不能证明全片段或跨源动作兼容；decoder-only 输出未在本检查中执行独立 GS 冷重放。
