# 全片段物体监督：匹配预算 A/B 对照

第五轮回放模型仍在搬运中段丢失绿梁。数据检查确认源帧、GS、状态和训练缓存对齐；固定手部加权区域平均仅覆盖关键窗口中 24.15% 的绿色像素。本轮检验额外物体区域 RGB 监督是否改善这种退化，尚未认定该现象是唯一根因。

两组均从 `anchor_replay_pilot` 第五轮同一 checkpoint 恢复 decoder、全部 `[168,9,16]` 连续自由码和 Adam 动量／步计数。续用同一打乱序列的第 6、7 轮，各 846 次 B4 更新，累计 2961 次。每个窗口/未来步新增两次基础访问，两处关键窗口的每个未来步另有 20 次回放。初态编码、候选半径 4、原 GaussianGPT codec、原 RGB 与 latent 损失、学习率均不变。

`control` 的物体权重为 0；`object` 为 1。源 TRAIN RGB 按物理帧建立黄色条和绿色横梁的颜色区域，去掉小于 20 像素的连通分量，监督区域为初态与目标帧区域并集，再向外扩张 3 像素。黄色包括固定工作区内两根可见黄色条；绿色区域排除左侧绿色三角块。每种颜色分别求区域 RGB MAE，再对非空颜色组取平均。它们是颜色代理，不是人工实例真值；启用前检查接触图。

掩码只进入渲染后的损失，不进入 decoder 条件、候选坐标或状态选择。Q-Former、LFQ、FSQ、AR 均不参与本轮。历史 checkpoint 没保存随机数状态；两组采用同一明确的重启 seed。稀疏后端非完全确定，不能声称与历史进程逐位连续。

在 134 复用 `/data/CJH/envs/gaussiangpt/bin/python`，不安装或改变包。两组分别用 GPU0／GPU1，每进程要求至少 32 GiB 空闲并将 PyTorch allocator 限制为 24 GiB；既有常驻进程保持运行。每组训练上限 5400 秒；评估、独立重放有各自固定超时。结果不达标不自动加训。

执行顺序：生成 `CODE.sha256` → `prepare_masks.py` → `tests.py` → 检查掩码并保存 `MASK_REVIEW.json` → 两 GPU 各执行 `trial_worker.py preflight`，检验真实梯度和完整状态复位 → `launch.py`。预检对照组还与原训练函数比较一次更新。协调器会自动完成每组训练、1512 项评估、104 包／208 视角独立重放、八窗口与轴线报告。

最终须同时看物体持续存在、位置／方向、其他窗口退化、全片段与 h9 误差。原颜色代理门槛保留，但必须结合逐帧画面；颜色像素检出不等于完整物体保留。没有自动教师验收或自动 Q-Former 启动。

入口示例（先完成掩码检查和 CPU 测试）：

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 /data/CJH/envs/gaussiangpt/bin/python -B trial_worker.py preflight --arm control --out /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_object_supervision_ab_20261102_v1/preflight/control
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=1 /data/CJH/envs/gaussiangpt/bin/python -B trial_worker.py preflight --arm object --out /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_object_supervision_ab_20261102_v1/preflight/object
/data/CJH/envs/gaussiangpt/bin/python -B launch.py
```

预检和正式训练会拒绝覆盖已有目录。`run/SUMMARY.json` 是终态汇总；`run/control` 与 `run/object` 保存各自的训练、评估及失败证据。GPU 前置检查和恢复检验通过只说明工程路径可运行，不能提前宣布绿梁已经恢复。
