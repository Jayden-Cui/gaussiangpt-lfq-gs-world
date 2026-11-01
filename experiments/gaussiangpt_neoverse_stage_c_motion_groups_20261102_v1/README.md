# 阶段 C：按源画面变化分组比较固定教师

`report.py` 弥补阶段 C 原有全片段和 h9 汇总没有按动态程度分组的缺口。它复用 B 训练期间冻结的 168 窗口源画面变化分组；每组 42 个窗口，在相同分组下分别报告后验连续码、自由码教师、重复初态码、邻窗口码和真实状态 codec 参考的 MAE／运动区域 MAE。结果包括组内物理帧等权均值、窗口预测等权均值和 h1–h9 各自均值，以及后验／教师的运动误差比。

脚本只接受完整的 1512 个窗口预测，并逐条核对 C 评估里复制的 `teacher_free` 指标与绑定教师评估中的 `correct` 完全相同；还核对 decoder、codec、状态缓存、教师码和教师评估的哈希对应。它不进行模型训练，不使用 GPU，不修改原评估，不自动判定动作成功。

正式 C 完成后，在服务器上运行：

```bash
/data/CJH/envs/gaussiangpt/bin/python -B /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_stage_c_motion_groups_20261102_v1/report.py \
  --groups /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_motion_groups_20261102_v1/MOTION_GROUPS.json \
  --student /absolute/path/to/C/evaluation/EVALUATION.json \
  --teacher /absolute/path/to/the/bound/teacher/EVALUATION.json \
  --output /new/absolute/path/C_MOTION_GROUPS.json
```

三种 C 配置均应使用同一份冻结分组与同一份教师。分组来自源图像的全画面变化，可能包含手部和遮挡，并非物体追踪；它是在 B 训练期间建立的事后分层，不是 B 启动前预注册，也不是跨视频验证。只通过合成数据测试不代表真实 C 模型已经训练或通过验收。
