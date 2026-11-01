# 历史执行记录

第一次启动在 GPU 运行前拒绝了历史 codec 元数据：验证代码期待 `complete`，实际受绑定历史结果使用 `evaluated`；相关哈希一致。修正字段后第二次启动在独立进程导入 native bridge 时缺少原模型已绑定的模块路径，错误记录保存在 `result/FAILED.json`。这两次都没有更新训练模型。

独立 v2 新增真实历史元数据及 native bridge 源路径 CPU 检查后完成 GPU 参考渲染，保留本目录作为失败记录，不再运行 v1。
