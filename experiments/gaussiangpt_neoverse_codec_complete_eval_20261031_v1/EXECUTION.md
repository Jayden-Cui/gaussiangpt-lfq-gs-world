# 三种空间 codec 的完整评测

推理设置已由v7真实GPU短验证确定，完整评测尚待源码冻结和提交。只评测，不训练动作模型，不自动选取或替换 LDM 的 codec。

原始 codec3000、完成4426次更新的 decoder_only、独立重训成功完成4426次更新的 full_codec，分别在相同51段、每段177相邻原帧上重新编码并评测，共9027帧/模型。50段TRAIN共8850帧，已暴露的tower14共177帧单列；后者不能用来选择检查点或称为未见验证集。保持每帧112896输入GS、448×252、64维连续状态、Identity状态量化、全部现有聚合输入，无抽帧、抽点、降分辨率。三个模型使用相同推理设置。

每帧保存状态并从磁盘重建，在真实相机及附近伪视角与直接解码进行重放比较。标准仍是两个视角分别MAE≤1e-5且PSNR≥60。有限数值超限保留逐帧失败后继续评测，任何失败仍使重放验收不通过。执行异常、NaN、结构错误、来源变化或资源/时间超限仍终止该臂。完整质量覆盖不等于重放通过，更不等于codec已验收或动作Z成功。

使用v7已完成两帧短验证的float64解码计算。只将ae.decoder、feature_heads、gaussian_multiplier的原float32权重数值精确转为float64；encoder、保存的[N,64]状态仍为float32。每次解码用同coordinate_map_key/manager把F.double作为输入，_world后将浮点GS属性转回float32交原renderer。转换记录须证明castback精确、encoder/未选模块参数与buffers不变；运行末尾验证所有运行精度状态未变。转换前检查点模型digest与转换后运行模型digest分开记录，不能把dtype差异伪装成同一字节摘要。

ME固定原False与SEPARATE模式，帧边界global manager保持None，不启用v6的SHARE方案。直接编码路径完成后，磁盘包必须在另一个全新manager中重建；保留第一个对象直到验证第二个确实不同，随后立即释放第一个引用，再进行重放解码。相同进程的独立manager不是跨进程原生GS包重放，分别报告。

v7短诊断启用pruning捕获hooks，本完整评测不挂这些hooks；不能把短诊断冒充完整验收。本轮自己验证9027帧的独立manager重放。v7的float64模式须实际完成且fresh四路通过才允许启动；float32对照、不同历史state比较保持原值，不要求它们通过来掩盖或混淆条件。原权重数值、原剪枝阈值、所有输入及target_key=None不变。

全量保存state与来源清单；每段0/44/88/132/176保存真实图、teacher、codec、附近视角及附近teacher。CPU比较检查实际9027个state与1275个PNG的字节数和SHA，报告每片段、50 TRAIN片段等权、19来源组等权、已暴露14独立指标。仅描述性比较，不计算置信区间；待后续视觉审查。

## 环境与CPU见证

温复用134服务器的`/data/CJH/envs/gaussiangpt`，不安装或升级包。环境spec SHA为`5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f`，沿用`gaussiangpt_neoverse_vw1_alignment_20261030_v5/.aris/compute/ssh-134.md`的既有环境证据。复用旧GPU预检必须标为历史证据，不能冒充本轮执行。

冻结所有根级.py、protocol.json与说明.md为CODE_MANIFEST后，由未参与实现的见证者执行一次：

```powershell
C:\Python314\python.exe -B E:\MoViesGS\experiments\gaussiangpt_neoverse_codec_complete_eval_20261031_v1\validate.py --root E:\MoViesGS\experiments\gaussiangpt_neoverse_codec_complete_eval_20261031_v1
```

要求实际测试数大于零、全部通过、零skip，精确绑定manifest与protocol；原始日志和退出码保留。CPU测试不替代完整GPU评测。验证失败保存证据，修订后另行冻结，不能覆盖失败见证。

## 一次启动与有界等待

GPU0顺序评测baseline、decoder_only。GPU1只读等待独立full_codec训练的PID479060/start_ticks40929788退出并具备成功DONE与TRAIN_DONE，再评测最终4426检查点。外部训练从不登记为本队列子进程，也不向它发送终止信号。各评测独立失败，其他臂继续。

每臂准入要求实际GPU可用显存至少16GiB、输出盘可用至少100GiB。CPU6线程；worker内部7200秒、12GiB显存保护；进程外层8500秒；外部训练等待8500秒；CPU比较1800秒；整队列20000秒。预算包含初始化，无自动重试或科学条件缩减。

部署到全新同名目录后用独立screen -dmS启动：

```bash
env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 /data/CJH/envs/gaussiangpt/bin/python -B -u /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_codec_complete_eval_20261031_v1/pipeline.py --root /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_codec_complete_eval_20261031_v1
```

一次启动确认记录准确PID/start_ticks及目录，再由单一程序等待queue/DONE、PARTIAL、FAILED或进程终止；模型不循环查询进度。完整比较只在三臂都完成9027帧时生成；缺任一臂不得写完整结果。DONE仅表示评测与比较完成，codec_accepted保持false等待实际证据审查。
