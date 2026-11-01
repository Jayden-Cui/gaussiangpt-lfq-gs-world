# 训练与评估入口：版本 2

训练使用 `b_worker_v2.py`，评估与原生重放使用 `b_evaluate_v2.py`。旧入口保留用于追溯，不应执行：旧入口在初始化后再次调用 PyTorch 线程设置，服务器环境会报 `cannot set number of interop threads`。

版本 2 的进程仅在开始时设置一次 inter-op 线程；后续 `gpu_runtime.py` 只检查可见 GPU、空闲显存和分配上限。`GPU_RUNTIME_TEST.json` 在模拟 GPU 上检查了重复资源检查不会再次设置 inter-op 线程，并拒绝显存不足或暴露多张 GPU 的情况。`WORKER_TEST_V2.json` 和 `EVAL_TEST_V2.json` 分别绑定新入口和测试文件的哈希。

目前仍未触发阶段 B，也没有完成真实 GPU 预检。该问题与已在运行的阶段 A 使用的入口无关。
