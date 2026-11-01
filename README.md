# GaussianGPT 空间状态与动作码实验

本仓库整理 GS World 实验的代码，供分析 GaussianGPT 空间状态、Q-Former 动作表示与 LFQ 量化的接入方式。`experiments/` 同时保留当前正式对照和历史版本；阅读时应优先从下列入口开始，避免把历史失败尝试当成最终实现。

## 当前主线：从 GS 状态到动作码

1. **空间状态与 GaussianGPT 编解码**：[codec.py](experiments/gaussiangpt_neoverse_state_motion_20261030_v1/state/code/codec.py) 将每一时刻 GS 编码为带整数坐标的连续 `[M_t, 64]` 状态，并从状态解码 GS；状态量化使用 `Identity`。GS 渲染和属性转换见同目录 `common.py`。
2. **Q-Former 与初态锚定解码**：[model_adapter.py](experiments/gaussiangpt_neoverse_z_paired_formal_20261031_v1/model_adapter.py) 接入9个按时间前缀读取状态的动作查询；骨干与初态锚定解码见 [anchored_ldm.py](experiments/gaussiangpt_neoverse_native_z_20261031_v6/code/anchored_ldm.py)、`aligned_ldm.py` 与 `multiscale64.py`。
3. **LFQ 与连续码对照**：[action_bottleneck.py](experiments/gaussiangpt_neoverse_z_paired_formal_20261031_v1/action_bottleneck.py) 实现共同的 `64→16→64` 投影。每个完整窗口的动作码形状为 `[B, 9, 16]`；LFQ 分支每个动作码有16个二值符号，ID形状为 `[B, 9]`，理论码本大小为65536。连续分支保留浮点码。两臂共享初始权重、尺度和辅助损失。
4. **因果归一化**：[initial_norm.py](experiments/gaussiangpt_neoverse_z_paired_formal_20261031_v1/initial_norm.py) 限制归一化与条件统计使用初态槽位；模型输出坐标锚定窗口初态。
5. **训练和评测**：[train.py](experiments/gaussiangpt_neoverse_z_paired_formal_20261031_v1/train.py)、[paired_step.py](experiments/gaussiangpt_neoverse_z_paired_formal_20261031_v1/paired_step.py)、[evaluate.py](experiments/gaussiangpt_neoverse_z_paired_formal_20261031_v1/evaluate.py) 和 [protocol.json](experiments/gaussiangpt_neoverse_z_paired_formal_20261031_v1/protocol.json) 分别给出训练流程、损失、动作码控制实验和固定配置。`coordinator.py` 串联两臂及后续评测。

实验以 NeoVerse 导出的搭积木 GS 为数据源，177帧，使用相邻帧构成10帧窗口，共168个重叠窗口；batch size 为4。Q-Former 的第 `h` 个查询读取该窗口的 `S_0…S_h`，并非从文本或已预测历史中自回归地产生未来动作码。

## 新增诊断：自由未来特征

[两端关键帧的有界／无界残差诊断](experiments/gaussiangpt_neoverse_free_feature_20261101_v1/README.md) 绕过动作编码和运动解码，在完整初态坐标上直接优化未来特征，固定同一份原始 codec。四组各 600 步已完成，独立 GS 重放通过；黄条仍破碎，绿梁没有恢复，未通过物体恢复验收。见[完整指标](experiments/gaussiangpt_neoverse_free_feature_20261101_v1/SUMMARY.md)、[人工视觉审查](experiments/gaussiangpt_neoverse_free_feature_20261101_v1/EXECUTION.md)、[第 44 帧对照](experiments/gaussiangpt_neoverse_free_feature_20261101_v1/s035/comparison_v0.png)及[第 132 帧对照](experiments/gaussiangpt_neoverse_free_feature_20261101_v1/s123/comparison_v0.png)。

[真实未来坐标＋自由特征对照](experiments/gaussiangpt_neoverse_future_support_20261101_v1/README.md) 的四组各600步已完成。换成真实未来坐标、仅从初态映射特征后，黄条竖直姿态与绿梁水平布局基本恢复，同模式运动误差下降36.5%–41.3%；仍有边缘和相邻视角缺陷。见[结果表](experiments/gaussiangpt_neoverse_future_support_20261101_v1/SUMMARY.md)、[审查记录](experiments/gaussiangpt_neoverse_future_support_20261101_v1/EXECUTION.md)、[44帧](experiments/gaussiangpt_neoverse_future_support_20261101_v1/s035/comparison_v0.png)和[132帧](experiments/gaussiangpt_neoverse_future_support_20261101_v1/s123/comparison_v0.png)。未来坐标是直接提供的参考信息，支撑数量和初始化同时变化；不能据此宣称模型已经预测未来几何。

[学习未来支撑的解码器测试](experiments/gaussiangpt_neoverse_learned_support_20261101_v1/README.md) 两臂各1200步已完成。仅从初态候选预测未来格点及64维特征，两个相邻10帧窗口的自由连续码形状为 [2,9,16]；codec冻结，推理不输入未来坐标或点数。18个未来状态平均运动MAE为0.029776，固定零码为0.039084（降低23.81%）；交换窗口码后为0.049276。黄条和绿梁主要运动过程能够重建，但仍有额外占据和手部伪影。详见[完整结果](experiments/gaussiangpt_neoverse_learned_support_20261101_v1/SUMMARY.md)、[逐帧审查与视频](experiments/gaussiangpt_neoverse_learned_support_20261101_v1/VISUAL_REVIEW.md)。本轮只支持两个已知窗口的decoder能力与条件依赖，尚未运行Q-Former/LFQ/AR。

[Q-Former连续码接入测试](experiments/gaussiangpt_neoverse_qformer_support_20261101_v1/README.md) 已完成1200步。冻结空间解码器和原codec后，Q-Former的18帧平均运动MAE为0.029765，自由码参考0.029776，重建保持门槛全部通过；30个GS包／60次视角重放通过。仅初态控制误差增加9.31%，未达到预设10%，后续状态依赖检查未全部通过。逐帧画面基本保持，但仍有手部、边缘和多余占据问题；结果范围为两个已知TRAIN窗口，未运行LFQ/FSQ/AR。见[结果与视觉审查](experiments/gaussiangpt_neoverse_qformer_support_20261101_v1/EXECUTION.md)、[指标汇总](experiments/gaussiangpt_neoverse_qformer_support_20261101_v1/SUMMARY.md)。

[全177帧训练](experiments/gaussiangpt_neoverse_qformer_full177_20261101_v1/README.md) 的posterior与受训initial_only均完成2000步、全部168窗口评估及每组104包/208视角原生GS重放。历史同名评估模块的导入错误已通过[独立恢复入口](experiments/gaussiangpt_neoverse_qformer_full177_eval_resume_20261101_v1/README.md) 修复，未追加训练更新。按176个物理未来帧等权，运动MAE分别为0.075142和0.076111，真实前缀组仅降低1.27%；真实未来状态codec参考为0.029464。[指标及目视记录](experiments/gaussiangpt_neoverse_qformer_full177_eval_resume_20261101_v1/EXECUTION.md) 显示静态结构较好，但黄色长条和绿色横梁的关键运动仍有消失、拖影，动作质量未达到目标。代码、原始评估JSON、两组全部16段诊断视频、32张视角对照关键帧及4张序列对照图均已收录。本轮全部动作阶段参与训练，无未见动作划分；仍为连续码 [B,9,16]，无LFQ/FSQ/AR。

[全177帧固定预算续训](experiments/gaussiangpt_neoverse_full177_fit_budget_20261101_v1/README.md) 已全部完成：两组均累计9560步，三个固定评估点及合计624份GS包/1248次视角重放通过，总耗时约5小时16分钟。最终运动MAE为0.062531（真实前缀）和0.063462（受训初态），较各自2000步改善16.78%/16.62%，真实前缀组相对优势仍仅1.47%。[完成指标与目视结论](experiments/gaussiangpt_neoverse_full177_fit_budget_20261101_v1/EXECUTION.md) 显示黄条和绿梁的关键运动仍有严重丢失/拖影，动作质量未达目标。已收录三阶段原始评估JSON、最终两组16段视频、32张视角关键帧，以及[无顶部标记的源视频／9560步Q-Former重建对照](experiments/gaussiangpt_neoverse_full177_fit_budget_20261101_v1/clean_comparisons)。本轮已结束，无新增训练或LFQ/FSQ/AR。

## 结果和关键帧

正式对照两臂各800步。按176个不同未来帧等权汇总，连续 Z／LFQ 的全图 MAE 分别为0.030408／0.032773；关键帧中的黄条和绿梁动作仍有明显错误，因此结果尚不支持动作重建成功或 AR 预测成功。

- [实验汇总与执行记录](experiments/gaussiangpt_neoverse_z_paired_formal_20261031_v1/EXECUTION.md)
- [44帧与132帧关键帧](keyframes/)
- [两段窗口对照视频](videos/)：5 fps慢放，源视频约29.97 fps。
- [完整结果表](experiments/gaussiangpt_neoverse_z_paired_formal_20261031_v1/SUMMARY.md)

## 仓库范围与复现限制

`experiments/gaussiangpt_neoverse_*` 收录本工程相关的各阶段源代码、测试和配置，包括数据准备、编解码、重放诊断、旧版动作模型、正式配对训练与评测。`experiments/gaussiangpt_state_vae_20261028_v1/vendor/` 保留实验使用的 GaussianGPT 上游源码及其许可证。

**仓库不含训练权重、原始视频、NeoVerse 输出、状态缓存、教师资产或GPU运行环境。** 实验代码中的 `/data/CJH/AGS-V6Direct/outputs` 是原134服务器目录，并硬校验原环境及文件哈希；这些代码可供分析，但直接下载仓库并不能独立复现运行。完整复现需要按代码与协议准备相同数据、权重、依赖版本及GPU扩展，并重新配置路径或在相同目录结构部署。

本次结果来自同一个已用于训练的片段和一个随机种子，是有真实未来状态输入的后验重建对照，不是未见场景泛化结果。历史目录中的失败或中断实验保留原有名称和记录，不应与当前正式结果混用。
