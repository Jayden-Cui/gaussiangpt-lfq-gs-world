# 阶段 C 评估入口：版本 4

正式评估与原生重放应使用 `c_evaluate_v4.py`。它保留版本 3 的逐文件代码校验和一次性 GPU 线程初始化，并要求教师通过版本 2 的完整预测、运动审查与原生 GS 重放校验。

`DRIVER_TEST_V4.json` 是 CPU 逻辑见证，不代表真实模型评估已经执行。旧入口保留供追溯。

V4 的全片段和 h9 汇总本身不按画面变化强度分层。正式 C 评估完成后，另用 `gaussiangpt_neoverse_stage_c_motion_groups_20261102_v1/report.py`，把 C 的 `evaluation/EVALUATION.json` 与同一收据绑定的教师 `EVALUATION.json` 交给该工具，生成高／低变化组及 h1–h9 报告；调用方式见该工具的 README。这一步不会更改 V4 已绑定的代码、模型或评估文件，也不能替代物体跟踪及视觉审查。
