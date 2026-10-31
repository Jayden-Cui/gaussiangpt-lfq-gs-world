# 完整 codec 质量评测与重放记录

三臂各9027帧质量评测完整；质量评测完整、纯状态重放通过与总codec验收分别报告。codec尚未验收，等待另行视觉与重放审查。

|模型|TRAIN 50片段等权 PSNR|TRAIN 19来源组等权 PSNR|来源组等权 MAE|暴露14 PSNR|重放失败帧|
|---|---:|---:|---:|---:|---:|
|baseline|17.5167|18.0185|0.082541|17.7874|0|
|decoder_only|27.8823|27.6551|0.025958|22.0700|0|
|full_codec|26.3604|26.1082|0.036737|22.5533|0|

SUMMARY.json 保留每片段、每来源组以及三种汇总范围的全图/ROI MAE与PSNR；包含codec对真实图、teacher对真实图、codec对teacher与伪视角对teacher指标。
每个重放失败帧的clip、frame、PTS、失败视角与实测值保存在各模型 replay_failed_frame_details。原阈值为全图MAE≤1e-5且PSNR≥60，两个视角都必须满足。
差值统一为参考模型减候选模型：MAE为正表示候选改善，PSNR为负表示候选改善。仅作描述统计，不计算置信区间或自动质量判定。
50个片段存在重叠，19来源组等权与片段等权分别报告；已暴露14单列，不能视为未见验证集。
三臂每臂实际核验9027个state文件和1275个PNG的bytes及SHA。重放是双视角latent重放，不是独立native GS重放。动作Z未训练，不能据此判定LDM成功。
三臂统一使用v7固定float64 decoder/head/multiplier，encoder和保存state保持float32，world GS转回float32后交给原渲染器。ME始终False/SEPARATE/globalNone，两次manager对象必须不同。冷manager隔离证据与图像重放阈值结果独立报告。
原checkpoint模型摘要记录于精度转换前，运行模型摘要另列；castback_exact和encoder_exact明确验证原值可精确还原及encoder未变，运行state再作完整一致性检查。
准入v7是带剪枝观察钩子的两帧精度诊断；本完整评测不挂这些钩子，不能冒充同条件重复。它自己逐帧测量fresh→disk重放，float32对照与历史不同输入结果不替代float64 fresh-state准入。
