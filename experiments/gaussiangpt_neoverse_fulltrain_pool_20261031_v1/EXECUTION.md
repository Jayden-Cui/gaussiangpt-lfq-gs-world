# 全部 TRAIN 来源的完整 GS 数据池

固定包含权威索引中的 49 个原 TRAIN 片段，加上此前用户授权进入训练的 tower17_176，共 50 个片段、19 个来源组。tower14 不加入训练。每片段原视频 177 帧，逐帧 stride1，29.97fps；导出 448×252、112896 个 Gaussian/帧。总计 8850 个片段内帧，不声称跨片段去重后的物理帧数。

48 个片段完整新导出；tower10_176 与 tower17_176 复用已生成的完整序列，但必须先通过实际文件、时间戳、张量形状与哈希准入。全部点参与原聚合；不减帧、不减点、不降低分辨率、不替换归一化或科学算法。本队列只准备数据，既不训练 codec/LDM，也不代表动作 Z 成功。

## 环境与固定输入

继续使用 134 已验证的 GS Python `/data/CJH/envs/gaussiangpt/bin/python` 和原 NeoVerse runtime。GS 环境 spec SHA 为 `5813f7633c71422690253c97ae01036751f261478e289039755caa730f0f752f`，provider ledger 为 `experiments/gaussiangpt_neoverse_vw1_alignment_20261030_v5/.aris/compute/ssh-134.md`。不安装或修改包。原 tower10/tower17 的完整导出与原 helper 已有实际 GPU 运行证据；这次仅将相同调用应用到固定片段列表。

CODE_MANIFEST.json 冻结所有本目录 Python 源码、protocol.json、本文、INPUT_POOL.json 及两个指定历史依赖。独立进程组 helper 由 protocol 的绝对路径与 SHA 固定，并在运行前后校验。CPU_VALIDATION.json、queue/、clips/ 和各验收结果是输出，不能反过来推断为输入源码。

## 独立 CPU 文档见证

在 Windows 的 `E:\MoViesGS`，源码清单冻结后，由未参与编写的 fresh agent 逐字执行以下命令。它只进行 CPU 测试和冻结输入校验，不连接服务器、不导出、不训练、不安装包，也不替作者修复命令或代码。

```powershell
C:\Python314\python.exe -B E:\MoViesGS\experiments\gaussiangpt_neoverse_fulltrain_pool_20261031_v1\validate.py --root E:\MoViesGS\experiments\gaussiangpt_neoverse_fulltrain_pool_20261031_v1
```

必须实际运行四个测试模块并得到退出码 0、CPU_VALIDATION.json 的 `status=passed`、`gpu_executed=false` 和真实测试数大于零。已有见证文件不覆盖；如失败，保留失败证据，修复后以明确新修订验收，不能静默重试。

## 部署和单次提交

仅复制本次新 run 的冻结源码/文档/清单/CPU 见证；在远端检查压缩包 SHA 与清单，不能覆盖已有科学产物。以下是进入远端后的实际队列调用，外层 shell 上限由启动器设为 18620 秒、宽限 20 秒：

```bash
/data/CJH/envs/gaussiangpt/bin/python -B -u /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fulltrain_pool_20261031_v1/pipeline.py --root /data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fulltrain_pool_20261031_v1
```

队列核对 v9 的唯一 DONE、精确 PID/start_ticks 已退出及源码/结果 SHA，再进入数据准备。48 个新片段按数值来源组及 offset 排序后交替分配 GPU0、GPU1，每卡 24 个。只做一次 GPU 可用性准入，每卡至少 55GiB；初始磁盘至少 500GiB，每片段开始前保留 64GiB。CPU 数学库每进程 6 线程，纯 CPU 准入隐藏 CUDA。

每个片段依次执行静态输入绑定、完整 NeoVerse 导出、冻结 codec 准备，均有独立超时。prepare 使用原 helper 内部 1200 秒预算，外层 1230 秒。两卡导出阶段总上限 9000 秒，最终完整哈希验收上限 1800 秒；所有上限和依赖见 protocol.json。

首次失败停止本队列拥有的子进程组，保留已经写出的证据；不杀共享任务，不重启旧 run，不自动重试或降配。观测超时只代表未观察到终态，不能触发重启。

## 验收与边界

先生成 REUSE_ADMISSION.json，再对 48 个新片段生成各自 PREPARATION_RECEIPT.json，最终 POOL_COMPLETE.json 必须覆盖全部 50 片段。原 prepare helper 包含保存/重载张量逐项相等检查以及独立的 native/归一化渲染检查；本队列不把它们误称为“重载包再渲染”。

一次提交、一次启动确认、一次终态验收；程序等待训练依赖和任务终态。不要以模型反复 tail/ps/nvidia-smi 盯任务。遇到真实失败、用户要求或预定验收门槛时才读取对应证据。

获得完整数据池后仍需单独设计多来源 codec/LDM 训练与跨来源动作验证。当前 goal 保持 active，不能由导出完成或测试通过宣布动作码学成。
