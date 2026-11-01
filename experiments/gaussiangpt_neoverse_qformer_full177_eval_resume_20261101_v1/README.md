# 全177帧实验：从已完成权重接续评估

原训练目录 `gaussiangpt_neoverse_qformer_full177_20261101_v1` 的 posterior 与 initial_only 均已完成2000步，训练权重保存、重载和冻结边界检查通过。训练后的 `from evaluate import evaluate` 被模型加载器放到搜索路径首位的历史模块目录遮蔽，误选了旧版同名文件，导致两组在评估入口报错。

本目录只恢复评估，额外训练更新数为0。原训练源码、协议、缓存、checkpoint及失败记录保持原样。`resume_protocol.json` 绑定原源码清单、训练协议、两组末步checkpoint和训练完成记录；本目录的 `protocol.json` 是原协议的逐字副本。

`resume.py` 按绝对路径及独立模块名加载原版 `evaluate.py`、`replay.py`、`coordinator.py`，不修改原指标、采样、预测支撑或模型。状态缓存仍从原训练目录只读加载。所有输出写入本目录。每组复制的 `TRAIN_DONE.json` 是原始训练收据，`STARTED.json` 则记录本轮评估进程；结果另绑定实际末步模型摘要。

运行顺序：

```bash
/data/CJH/envs/gaussiangpt/bin/python -B resume.py regression
/data/CJH/envs/gaussiangpt/bin/python -B resume.py coordinate
```

CPU回归检查在真实模型加载之后复现旧版搜索路径遮蔽，验证显式路径选择了正确评估函数，并核对两组已训练权重的完整张量摘要。它不代表GPU渲染检查通过。评估分别使用GPU0、GPU1，每进程6线程，32GiB空闲显存门槛、24GiB分配上限；每组评估最多10800秒，原生GS重放最多1200秒，失败停止，无自动重试。

每组仍覆盖168窗口、1512项未来预测、176个物理未来帧，并导出8个诊断窗口的视频、104份原生GS包及208次视角重放。统计先对同一物理帧的不同上下文求平均，再对176帧等权。训练覆盖来自现有177帧TRAIN视频，无未见动作/新视频泛化结论；无LFQ/FSQ/AR。

状态以本目录 `STATUS.json`、各组 `EVAL_STATUS.json` 和最终 `SUMMARY.json` 为准。原目录 `FAILED.json` 继续保留评估导入失败证据，不能据其单独判定训练未完成。
