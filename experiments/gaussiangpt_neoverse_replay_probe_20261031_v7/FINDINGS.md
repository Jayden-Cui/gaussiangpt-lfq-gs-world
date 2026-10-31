# float64解码在两帧短验证中通过

真实GPU诊断完成，RESULT SHA256为`15eae21a080a4cddec7e19ff660cd007f16e586fd4302c3a77104fc1f8242789`，终态工具chunk为`24db10`。每种精度均测试两帧、五条路径各三次、两个渲染视角；原检查点、所有输入、剪枝阈值与MAE/PSNR标准不变。

|精度|帧|重复输出GS数|失败记录/15|两视角最大重放MAE|对真实图主视角PSNR|
|---|---:|---|---:|---:|---:|
|float32|113|43836|0|1.390e-7|26.207748|
|float32|114|42920或42924|3|1.265e-5|25.806887|
|float64|113|43836|0|6.945e-8|26.207760|
|float64|114|42924|0|5.224e-8|25.806887|

float64的fresh_same_state、historical_state_comparison及原all_lanes汇总全部通过。原始模型参数由float32精确转换，castback逐值一致；encoder、保存的[N,64]状态不变，仍为float32；_world后GS转float32交原renderer。ME使用原False/SEPARATE，不启用之前失败的排序共享方案。原生CUDA后端实际完成了double计算；这不只是CPU推测。

114帧第二层最近阈值间距仍仅约2.199e-7，阈值没有移动；提高计算精度让该已观察边界保持在同一侧。此结果支持对已观察失败的数值精度修复，但不代表逐位完全确定性，也不是所有9027帧已验收。两个模式的encoder仍可能产生原float32微差，跨模式latent比较已保留，不把所有差异归因于单一算子。

下一步以同一float64解码设置评测baseline、decoder_only和完整full_codec，各9027帧；直接编码与磁盘重放必须使用独立manager，并且不再挂诊断pruning hooks。待该完整实测和视觉检查完成后才选择动作模型的codec。
