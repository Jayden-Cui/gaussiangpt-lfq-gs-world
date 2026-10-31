# v7：有界解码精度诊断

保持同checkpoint、两帧、五路三次、112896源GS、448×252、原MAE/PSNR与实际剪枝阈值。ME固定False和原SEPARATE，不再SHARE。两条件分别新建模型，名为decoder_precision=float32/float64；仅原decoder/feature_heads/gaussian_multiplier转目标精度，encoder及保存状态float32不变。每次decode通过同key/manager的F.double输入，world GS浮点属性转float32供原renderer。不能把精度条件假写为deterministic=True。

v6实际完成但True＋SHARE fresh仍失败；本轮只检验数值精度，不预先称已修复。precision.py记录checkpoint原模型、精确castback/encoder不变和条件内状态不变。真实GPU若不支持double则记录失败，不能静默回退。

保留raw先存、hook clone、真实重复多重集合、原始logit严格阈值、candidate/GS数量及quality诊断。clone有额外拷贝/调度成本。原all-lanes定义不变，fresh四路和旧historical比较分别汇总；旧历史features不同，仅historical失败不等同fresh存盘重放失败。

## 环境与冻结

134温复用 `/data/CJH/envs/gaussiangpt`，既有环境spec SHA `5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f`；沿用 `gaussiangpt_neoverse_vw1_alignment_20261030_v5/.aris/compute/ssh-134.md`。不安装或修改依赖；协议绑定四个已核ME源码SHA，原decoder源码另外固定。

八个源文件probe.py、precision.py、test_probe.py、test_precision.py、validate.py、protocol.json、PROBE_NOTES.md、EXECUTION.md冻结为CODE_MANIFEST。冻结后由未参与实现的新agent只执行一次：

```powershell
C:\Python314\python.exe -B E:\MoViesGS\experiments\gaussiangpt_neoverse_replay_probe_20261031_v7\validate.py --root E:\MoViesGS\experiments\gaussiangpt_neoverse_replay_probe_20261031_v7
```

validate发现test_*.py，实际CPU torch张量/模块测试必须执行，不能跳过或只做静态断言。要求tests>0、全通过、0skip、gpu_executed=false、manifest/protocol SHA匹配，保留输出/退出码。若返回session沿用原handle等待终态，不重新提交、不修补重试。CPU通过不证明ME/CUDA double支持。

## 一次GPU执行

只用134 GPU0，启动前一次确认可用显存≥16GiB、磁盘≥2GiB。CPU6线程，显存保护12GiB；初始化1200秒，两个精度条件合计1200秒，外部2460秒。超限停止，不缩输入或加预算。部署到同名全新目录：

```bash
env CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 timeout --signal=TERM --kill-after=20s 2460 /data/CJH/envs/gaussiangpt/bin/python -B -u /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_replay_probe_20261031_v7/probe.py --root /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_replay_probe_20261031_v7
```

独立screen -dmS记录launch.log，一次核PID/start_ticks，再由单个等待程序等RESULT/FAILED/进程结束；不循环查询或超时重提。旧v1–v6只读，启动目录不得已有STARTED/records/RESULT/FAILED。

## 输出解释

各精度独立PRECISION_CONVERSION与PRECISION_RESULT/PRECISION_FAILED；根RESULT使用decoder_precision_results、cross_decoder_precision。核对真实输入dtype/key/manager、原始logits、候选和GS数、质量、castback与encoder/模式内state验证、ME False/SEPARATE及原flag恢复。编码器仍是原float32路径，跨条件fresh latent若微差会显式记录，不能把所有跨条件差异仅归因于decoder精度。

阈值有限越界继续保存，非有限/执行/绑定/资源失败记录FAILED。completed只表示诊断完成，repair_validated=false。nearby无真实采集图，只对teacher计算质量并标real=null。即使两帧fresh通过也须后续全量9027帧评测，不预先授权放宽阈值。
