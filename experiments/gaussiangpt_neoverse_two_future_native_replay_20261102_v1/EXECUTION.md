# v1 导出失败记录

CPU 覆盖与来源哈希检查通过。GPU 包生产进程初始化 `BoundNativeBridge` 时，发现本目录的辅助模块 `common.py` 占用了原生 renderer 必须使用的模块名 `common`，触发 `ValueError: Conflicting module: common`。错误发生在首个 GS 包导出之前，未进行训练更新；原始错误见 `result/PRODUCER_FAILED.json`。

v2 使用独立目录将辅助文件改名为 `replay_utils.py`，并测试导入生产脚本不会抢占原生 renderer 的模块名；不得将本次失败记为重放通过。
