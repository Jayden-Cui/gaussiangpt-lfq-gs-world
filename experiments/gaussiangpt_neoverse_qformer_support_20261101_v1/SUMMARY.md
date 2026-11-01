# Q-Former连续码：冻结空间解码器测试

两个已知训练窗口，9个64维查询，连续码[2,9,16]；Q-Former按S0..Sh独立前缀读取，未运行LFQ/FSQ或AR。
600步自由码蒸馏 + 600步重建监督；解码器及codec完全冻结。18个未来状态等权平均。

| 条件 | 全图MAE | 运动MAE | PSNR | 支撑IoU |
|---|---:|---:|---:|---:|
| correct | 0.019617 | 0.029765 | 28.835 | 0.8180 |
| free_teacher | 0.019629 | 0.029776 | 28.838 | 0.8178 |
| initial_only | 0.021005 | 0.032537 | 28.025 | 0.8072 |
| zero_codes | 0.166510 | 0.220011 | 13.651 | 0.5251 |
| swapped_window_codes | 0.029770 | 0.048931 | 24.955 | 0.7526 |
| actual_target | 0.020141 | 0.030037 | 29.501 | reference |

数值门槛：{'mae_ratio_to_free_teacher_max': True, 'motion_mae_ratio_to_free_teacher_max': True, 'support_iou_drop_max': True, 'endpoint_color_iou_drop_max': True, 'initial_only_motion_error_ratio_min': False, 'swapped_motion_error_ratio_min': True}

未来输入依赖与重建保持分别报告；两个窗口、单次种子不能证明可迁移动作语义。
科学结果仍需看完整窗口与两端关键帧。没有自动进入LFQ或全量训练。
