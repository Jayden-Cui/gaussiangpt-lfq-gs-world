"""Format previously computed values; does not re-run evaluation or statistics."""
import json
import pathlib

r = pathlib.Path(__file__).resolve().parent
p = r / 'PARTIAL_METRICS.json'
d = json.loads(p.read_text(encoding='utf-8'))
d['limitations'] = [
    '评测按固定顺序在第2239帧中止；当前结果存在来源选择偏差，不能代表完整50段。',
    '12个完整片段共2124帧，另1段仅114帧；全部为训练片段，不含tower14仅评测片段。',
    '不把重叠片段视为相互独立样本，不报告总体置信区间。',
    '只训练状态codec，未训练动作QFormer/LFQ；指标改善不能证明动作Z学会。',
    '仅解码器训练结束；全codec训练被队列停止，无最终评测，不能比较两臂优劣。',
    '列出的重放通过仅覆盖2238个已写入行；失败帧仍失败，不因此放宽门槛。',
]
p.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
lines = [
    '# 完整数据池codec的中断评测：同帧部分结果', '',
    '仅解码器训练已完成4426步；评测在 buildingblock_tower_15_176 第114帧（整个评测顺序第2239帧）未通过状态重放一致性检查。以下只分析失败前已写入的2238行，不把它当作完成的9027帧正式评测。', '',
    '分析时间（UTC）：' + d['analysis_time_utc'] + '。使用远端标准库Python只读CPU计算，没有GPU执行、重放重跑或修改远端文件。原始指标按clip+frame精确配对；未根据训练loss推断效果。', '',
    '## 12个完整片段的等权结果', '',
    '12段均为177帧，共2124帧，来自6个训练来源组。每段帧数相同，因此片段等权结果也等于这2124帧的逐帧等权结果。全部12段的全图PSNR、全图MAE和工作区ROI MAE均改善。', '',
    '| 比较对象 | 指标 | 原codec | 训练后decoder_only | 变化 |',
    '|---|---|---:|---:|---:|',
]
names = {'codec_real': '真实视频', 'codec_teacher': 'NeoVerse原视角', 'pseudo_view_teacher': 'NeoVerse附近伪视角'}
labels = {'mae': '全图MAE↓', 'psnr': '全图PSNR↑', 'workspace_roi_mae': 'ROI MAE↓', 'workspace_roi_psnr': 'ROI PSNR↑'}
for group, values in d['complete_12_clips_equal']['comparison'].items():
    for metric, v in values.items():
        change = f"降低{v['relative_error_reduction_percent']:.3f}%" if metric.endswith('mae') else f"{v['after_minus_before']:+.6f} dB"
        lines.append(f"| {names[group]} | {labels[metric]} | {v['baseline']:.6f} | {v['decoder_only']:.6f} | {change} |")
lines += ['', 'ROI为固定工作区矩形，包含手和背景，不等于物体语义分割。附近伪视角监督来自NeoVerse渲染，不是真实多视角几何真值。', '',
          '## 逐片段真实视频比较', '',
          '| 片段 | 帧数 | 原PSNR | 新PSNR | 原MAE | 新MAE | 原ROI MAE | 新ROI MAE |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
for row in d['per_clip']:
    v = row['comparison']['codec_real']
    lines.append(f"| {row['clip']} | {row['frames']} | {v['psnr']['baseline']:.4f} | {v['psnr']['decoder_only']:.4f} | {v['mae']['baseline']:.6f} | {v['mae']['decoder_only']:.6f} | {v['workspace_roi_mae']['baseline']:.6f} | {v['workspace_roi_mae']['decoder_only']:.6f} |")
v = d['all_2238_frames_descriptive']['comparison']['codec_real']
lines += ['', '## 全部2238帧的描述统计', '',
          f"包含上述12个完整片段以及 tower15_176 的0–113帧。全图PSNR {v['psnr']['baseline']:.6f} → {v['psnr']['decoder_only']:.6f} dB；全图MAE {v['mae']['baseline']:.9f} → {v['mae']['decoder_only']:.9f}（降低{v['mae']['relative_error_reduction_percent']:.4f}%）；ROI MAE {v['workspace_roi_mae']['baseline']:.9f} → {v['workspace_roi_mae']['decoder_only']:.9f}（降低{v['workspace_roi_mae']['relative_error_reduction_percent']:.4f}%）。另两类比较对象的全部指标保存在PARTIAL_METRICS.json。", '',
          '## 证据检查与失败边界', '',
          '- 2238行无重复clip/frame，全部匹配基线的时间戳、来源组、TRAIN角色、64维状态和112896输入GS。',
          '- 新结果全部绑定同一4426步checkpoint；编码器SHA与基线及TRAIN_DONE一致。没有比较不同帧集合。',
          '- 2238个已写入行的原视角与附近视角重放均满足MAE≤1e-5且PSNR≥60；失败帧不在这些已完成行中。',
          '- 基线和新结果的teacher_real数值最大差：MAE 1.86265e-9，ROI MAE 2.79397e-9，PSNR/ROI PSNR 7.62939e-6 dB，属于两条指标路径的微小差异；未宣称位级相同。', '',
          '| 已完成行的重放检查 | 最大MAE | 最小PSNR |', '|---|---:|---:|']
for name, v in d['checks']['replay'].items():
    lines.append(f"| {name} | {v['max_mae']:.10g} | {v['min_psnr']:.6f} |")
lines += ['', '失败帧报告的重放MAE为1.2637862710107584e-5，超过预设1e-5；PSNR为72.6033706665039 dB。该PSNR衡量同一状态保存前后渲染的一致性，不是对真实视频的重建画质。此处未诊断数值差异根因，也未放宽门槛。', '',
          '## 不能据此宣称的结论', '']
lines.extend('- ' + item for item in d['limitations'])
lines += ['', '## 原始文件SHA256', '']
lines.extend('- ' + k + ': `' + v + '`' for k, v in d['source_sha256'].items())
lines += ['', '分析脚本为analyze_partial_metrics.py；format_partial_metrics.py仅修复本地中文编码并生成此报告，保留远端统计的全部数值。原始结果路径和checkpoint/encoder SHA详见PARTIAL_METRICS.json。', '']
(r / 'PARTIAL_METRICS.md').write_text('\n'.join(lines), encoding='utf-8')
print('Wrote PARTIAL_METRICS.json and PARTIAL_METRICS.md; numerical values unchanged.')
