# 动作 adapter 实施与 CPU 证据

2026-10-31。此阶段实现四个 Python 文件；不接入 codec，不部署，不训练、不执行 optimizer.step、不初始化 CUDA。`DESIGN.md` 中 800 步只是草案，本阶段没有冻结训练协议、checkpoint、teacher 资产或资源预算。

## 已实现的接口

| 文件 | 内容 |
|---|---|
| `action_bottleneck.py` | `ActionBottleneck`：同一真实 LFQ 的 64→16→64 投影；固定公共 s_ref；continuous/LFQ 重建分支；两臂均调用原库的 per-h auxiliary；浮点 actual-code packet |
| `initial_norm.py` | 两种初态参考 norm，以及原 decoder 的注入函数；保留原 Parameters 对象、名称、shape 和 affine 权重 |
| `model_adapter.py` | hash 验证加载真实 v6 AnchoredLDM / v9 spatial API，注入 encode/decode/校准方法，克隆同初值两臂 |
| `test_adapters.py` | 使用真实依赖、非零输出头和非零空间支路的小图 CPU 前反向测试；不使用模型/LFQ mock |

`load_pinned` 对所有被执行旧模块和 vendored QFormer 文件先验证 SHA256，再从同一 verified bytes 编译；检查实际导入 LFQ 的路径与 SHA。旧代码不会被改写，也不写旧 pycache。若进程内已存在未经本 loader 标记的同名旧模块，会拒绝模块碰撞；后续 worker 应在干净进程先调用此 loader，不混用任意旧 `native_experiment` 导入缓存。

真实 LFQ 绑定：`lookup_free_quantization.py` SHA256 `9fb68bdcb3763ac580c2a0f931640a70a14a0fed65e0b29a8800ee015b4452eb`。检查配置包括 16 bits、bias-free projections、scale1、soft_clamp10、identity STE、无 spherical/rotation，以及既有 auxiliary 参数。

## 公共初始化接入顺序

下面是接口示意，不是本轮已执行的训练指令。`train_reference_features` 和 `fixed_train_window_batches` 必须来自随后通过验收并独立绑定的 TRAIN state，不应直接从未验收 fullpool 输出取用。

```python
api = load_pinned(experiments_or_outputs_parent)
common = api.AnchoredLDM(blocks=4, checkpoint_blocks=False).eval()
common.initialize_output_units(train_reference_features)
api.attach_spatial_action(common, True)
receipt = attach_adapters(common, mode='lfq16')  # s_ref 尚未绑定
calibration = common.calibrate_actions(fixed_train_window_batches)
pair = clone_pair(common)  # 两臂独立 Parameters，逐值相同
```

调用者在构建前设置共同 RNG，并记录 init、完整 calibration starts、数据/encoder/模型 SHA。`calibrate_actions` 临时 eval/no_grad，随后精确恢复原有混合 train/eval 状态。所有批次合并成一个 RMS 标量，不按批或 horizon 各算一个；只能绑定一次。示例不建 optimizer，因此不会偷偷继承旧矩估计。后续两个 optimizer 应在 clone 后各自新建。

已有公共初始化可先用已验证的同架构 state_dict 加载，再注入 adapter。若是已经保存的 adapter checkpoint，先构造相同结构、attach，再 strict load state_dict；不能直接 load 整个 pickled model 冒充来源绑定。`action_mode` 是外部实验 metadata，刻意不放进公共 state_dict，以使两臂公共初值完全相同；保存 checkpoint 时必须同时保存并核对 `action_mode`，测试已覆盖该往返。

两臂共用 `v=10*tanh(P_in(z)/10)/s_ref`。continuous 返回 v，LFQ train 返回 `v+(sign(v)-v).detach()`，LFQ eval 返回硬符号；最后共用 P_out。原库对未除 s_ref 的 u 计算 auxiliary，两臂完全相同。公共增益会改变原历史 LFQ 的 reconstruction 梯度尺度；不能把新模型与旧 v10b 的差异只归因于离散化。

`encode_windows` 返回：

```text
z_prequant: [B,9,64]
action_codes: float32 [B,9,16]       # 实际送往 P_out 的码
action_latents: float32 [B,9,64]
z_quantized: action_latents 的兼容别名
diagnostic_ids: int64 [B,9]          # 两臂均有，仅用于诊断
indices: LFQ 是实际 ID；continuous 必须是 None
action_mode / aux_loss / initial_contexts / posterior_used
```

`decode_action_codes` 消费实际 float codes。`decode_ids` 仅允许 LFQ；continuous 若拿诊断 IDs 解码会报错。`ActionBottleneck.packet / packet_to_latents` 支持无精度丢失的 float32 actual-code 往返，核对 mode/scale/schema，LFQ 另核对合法 IDs 与 signs 一致。该 packet 是低层格式，**还没有 checkpoint/source/hash 外层 provenance**；后续 runner 必须补齐，不能把现有 packet 当正式验收工件。

新的 `encode_windows` 独立调用 `encode_initial(S0)` 构建 decoder contexts；posterior 继续按原模型读取 `S0…Sh`，不改成只读初态。初态 norm 的统计和 adaptive affine 池化都只取前四槽；旧 spatial 支路保持原 API。输出仍为全量初态坐标上的 10 帧，没有增删行、降低分辨率或抽帧。

## 实际验证结果

本地 Python `C:\Python314\python.exe` 有 torch/einops，但没有 `vector_quantize_pytorch`。本地语法解析通过；本地 unittest 在 setUpClass 明确报缺真实 LFQ，实际运行 **0 tests，未通过**。没有安装替代包或伪造跳过成功。

使用 134 上已有 `/data/CJH/envs/gaussiangpt/bin/python`，新源码仅经 stdin 在内存编译，没有远端文件部署。设置 `CUDA_VISIBLE_DEVICES=''`、线程数 2、外层 timeout180、`-B`。运行时 Python3.10.21、torch2.8.0+cu129，结束 `CUDA_INITIALIZED=False`。

最终测试：**9/9 通过，0 failures/errors/skips，3.243 秒，exit0**。工具执行 chunk 为 `2df647`。此前首次 8 项通过为 `b6ddae`；新增 checkpoint 往返、加强 LFQ 配置与 ID 范围检查后完成本次 9 项验证。

| 文件 | 最终被测 SHA256 |
|---|---|
| action_bottleneck.py | `4111735d1575b5fd0e71046b8d38e2f4191797e898cb417911bd6a4da44dfb20` |
| initial_norm.py | `21e1cc366aebe48ae78fc5f1460c3ed765095157d655854d1c57f9a9b8e90b80` |
| model_adapter.py | `112bf3cc14ee7517f8052e06467c0a8674e00a6e65b108fbbb1017c6da163662` |
| test_adapters.py | `d896032ed357dd3da3892c467ede124e2b599846fb3522d64c6a64002354c03a` |

证据涵盖：

- s_ref=1 时与真实 LFQ 在 train/eval 的 forward、IDs、auxiliary、输入及全部有效投影梯度一致。
- 共同尺度只绑定一次，global RMS 不依赖分批；连续与硬码的 auxiliary 值/梯度一致，实际重建码不同。
- 非零 residual head、非零 spatial Wo 的真实小图：改变后续 Z 不影响 h=0/1/4/8 输出前缀，其对应未来 Z 梯度严格为零，当前 h Z 的梯度非零；同一输入能检测出旧 decoder 的 h0 未来依赖。
- posterior h3 对后续坐标/features 大幅变更保持前缀不变，同时对已观察 S3 存在非零梯度，对更晚 S7 梯度为零；decoder 初态 context 与单独 encode_initial 完全相同。
- B2/T10 小图的完整编码/解码保留所有初态行；每个样本每 h 都有重建梯度到 action；query 与 project_in 有非零梯度。
- 连续 actual float code 的 BytesIO torch.save/load 精确往返；连续 ID 解码、伪装 ID、LFQ ID/sign 冲突被拒绝。
- Parameter 对象/形状保持、共同克隆逐值相同、mode metadata 和 injected state_dict 往返、校准前后混合模式恢复。

这些是 CPU 小图机制证据，不是 full177/B4/native-GS GPU 预检、不验证 renderer、codec replay 或物体动作质量；没有正式 CPU_VALIDATION manifest 冻结或独立见证者背书。

## 无文件部署的复现命令

在 `E:\MoViesGS` 的 PowerShell 执行以下整块。它只把本地四个 Python 文件送入远端进程内存，旧依赖从已绑定 outputs 读取；不会创建远端文件。

```powershell
@'
from pathlib import Path
import base64, json, shlex, subprocess
root=Path('experiments/gaussiangpt_neoverse_z_ablation_20261031_v1')
names=('action_bottleneck','initial_norm','model_adapter','test_adapters')
payload={name:base64.b64encode((root/(name+'.py')).read_bytes()).decode() for name in names}
runner='''import sys,json,base64,types,hashlib,unittest,os
sys.dont_write_bytecode=True
payload=json.load(sys.stdin)
for name in ('action_bottleneck','initial_norm','model_adapter','test_adapters'):
    data=base64.b64decode(payload[name])
    print('SOURCE_SHA',name,hashlib.sha256(data).hexdigest(),flush=True)
    module=types.ModuleType(name)
    module.__file__='/__memory_only_z_adapter__/'+name+'.py'
    sys.modules[name]=module
    exec(compile(data,module.__file__,'exec'),module.__dict__)
import torch
print('RUNTIME',sys.version.split()[0],torch.__version__,'CUDA_VISIBLE_DEVICES='+os.environ.get('CUDA_VISIBLE_DEVICES',''),flush=True)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(sys.modules['test_adapters']))
print('CUDA_INITIALIZED',torch.cuda.is_initialized(),flush=True)
sys.exit(0 if result.wasSuccessful() else 1)
'''
remote="env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 Z_ADAPTER_SOURCE_PARENT=/data/CJH/AGS-V6Direct/outputs timeout 180 /data/CJH/envs/gaussiangpt/bin/python -B -c "+shlex.quote(runner)
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','172.19.58.134',remote],input=json.dumps(payload).encode())
raise SystemExit(result.returncode)
'@ | python -B -
```

## 后续边界

待 codec/replay 验收决定唯一来源后，才可以做实际 TRAIN calibration、init/plan/optimizer/teacher/资源绑定，编写新 worker 和正式 evaluation。旧 v10b loss 日志直接使用 `indices.tolist()`，因此不能不改 runner 就执行 continuous；`indices=None` 是刻意暴露语义差异，不能把诊断 ID 填回去伪装兼容。两臂 aux 的外部 .1 multiplier 仍由未来 runner 负责，本模块只返回库原有内层 .1 auxiliary。

后续还需要真实完整 B4T10 GPU 梯度/显存见证、分段反向等价、168窗控制映射与实际帧覆盖、codec/source provenance 和双视角 latent/native packet replay。实现没有做新 codec 选择，没有接入任一未通过验收权重，没有正式训练预算，不包含 AR。
