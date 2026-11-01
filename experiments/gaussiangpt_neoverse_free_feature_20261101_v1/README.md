# 固定初态坐标上的自由未来特征诊断

目标是分开检查残差幅度约束和运动模块的拟合困难。对已失败的训练样例 `35→44`、`123→132`，直接优化完整初态坐标上的未来 64 维特征。此轮绕过动作 encoder、Q-Former、动作 decoder 和 LFQ。

## 实验契约

- 原始 baseline codec 全部冻结并保持 eval；使用原来的 native hard pruning、float64 decoder、float32 特征以及 448×252 全分辨率渲染。没有删点或空间支撑扩张。
- 每个样例分别运行有界/无界两组，每组 600 更新。两组从相同初态和零残差出发；Adam，学习率由 0.1 余弦衰减到 0.01；固定最终 600 步，不挑最佳关键帧。
- 设 `a=3*std(frame0)`，有界为 `F0+a*tanh(R/a)`，无界为 `F0+R`。有界与原式 `F0+a*tanh(U)` 等价（`U=R/a`）。两组优化变量均用特征单位表示，非零尺度通道的初始 Jacobian 都为 1，避免原始参数单位引入学习率差异。零尺度通道在有界组仍被锁定。
- 使用原正式实验真实视角终帧的三项 RGB 损失，整体乘以 10，保留相对权重：全图 MAE + 手部 ROI MAE + 10/9 运动加权 MAE。没有辅助正则、时间损失或教师训练损失。第二视角只评估，不参与优化。
- 与这次拟合使用同一个冻结 codec 重新生成 `held_initial` 和 `decoded_actual_target_state` 参考，不沿用各自适配过的正式模型参考图。
- 完整原视频仍为 177 帧、相邻源帧。此次有意选取两个终帧做能力诊断，并非把正式训练改为跳帧采样。

工程通过条件为四组均完成、真实输入特征梯度有限且非零、codec 参数与全部缓冲区保持不变、独立进程 native GS 重放通过（MAE≤1e-5 且 PSNR≥60）。重放只验证保存一致性，不能说明预测画质。

科学判断需要看第 44 帧黄色条是否竖起、第 132 帧绿色横梁是否横跨两柱。固定 HSV 颜色阈值、物体 ROI 的中心和主轴指标只是辅助，不能视为可靠物体追踪。拟合失败不证明数学上不存在解；终帧成功也不证明整段运动或动作压缩成功。

## 代码与运行

`core.py` 定义两种参数化、损失及诊断指标；`diagnostic.py` 加载有来源绑定的状态与冻结 codec、进行真实反向预检、拟合并保存图像与完整 GS；`replay.py` 在独立进程不加载 codec 地重放；`coordinator.py` 管理两个 GPU 工作者和汇总。

134 上新目录：`/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_free_feature_20261101_v1`。复用 `/data/CJH/envs/gaussiangpt/bin/python`，不修改环境、历史模块或历史实验。来源依赖和完整参数见 `protocol.json`。环境沿用既有 `gaussiangpt_movies_state_codec_20261029_v1/.aris/compute/ssh-134.md`，规格 SHA256 为 `5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f`。

部署时先执行一次 `python -B -m unittest -v test_core.py` 并保存带源码 SHA 的 `CPU_TESTS.json`，随后生成 `CODE.sha256`。由 `screen -dmS gg_free_feature_20261101 bash run.sh` 一次提交；两个端点分别在 GPU0/1 上运行，各端点内部按有界、无界顺序执行。每个 GPU 最少需要 24 GiB 空闲，allocator 上限 20 GiB；保留现有服务。每个工作者最多 5400 秒、重放 600 秒，超时只终止本轮自己的子进程，不重试、不自动改参。

`STATUS.json` 为紧凑总状态；`s035/STATUS.json`、`s123/STATUS.json` 为更新步数；完成后生成 `SUMMARY.json` 和 `SUMMARY.md`。每端点保存首/终帧真实图、各组第 0/100/300/600 步渲染、`comparison_v0.png`、`comparison_v1.png`、完整特征与 native GS 包、逐步日志及 codec 来源。不是运动视频；完整 10 帧实验需要根据本轮结果另作决定。

此目录复用仓库中历史模块，但权重、数据、状态缓存仍在服务器外部路径，因此单独下载源码不能离线复现。
