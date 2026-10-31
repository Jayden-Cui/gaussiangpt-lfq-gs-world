# 完整训练池数据适配器

本目录的 `pool_index.py` 只负责数据准入、逐帧读取和确定性覆盖计划。它不创建模型、不训练、不修改旧实验、不初始化 CUDA，也不将旧 `states/` 作为新 encoder 的编码。本文件不冻结训练协议、步数预算或 GPU 资源。

## 精确数据来源

准入对象是 `gaussiangpt_neoverse_fulltrain_pool_20261031_v1` 的已完成池：49 个原 TRAIN 片段，加用户授权的历史 DEV `buildingblock_tower_17_176`。合计 50 个片段、19 个 source group、8,850 个片段内帧；48 个新导出、2 个复用。tower14 不进入此索引。片段之间的物理帧去重数量未知，不能将 8,850 称作去重后物理帧数。

固定依据均来自已读取的源码或终态证据：

| 对象 | SHA256 |
|---|---|
| `queue/DONE.json` | `920a7f7c030733b2ff6d1be62aafb243bf581ea0a9bcc53ba3fe775b3d0f694b` |
| `POOL_COMPLETE.json` | `8ffe7055cba065a49c233245b106631665dcd710b3ebee2fd08368ab543224df` |
| 池 `CODE_MANIFEST.json` | `3d889cabcd5e9ef58f91466c888020b60d81fd9e36d24e5c91abca39b62182b0` |
| `INPUT_POOL.json` | `438d3113a246c214222db14805088773356c8ac251da2946ab27b75ef111f96c` |
| 旧 multisource `sources.py` | `7e8eaa327d1c05e7d00fa07121832c1f66d617c6e76a016eb711f1453606b249` |

Windows 下的 `gaussiangpt_neoverse_fulltrain_pool_20261031_review` 只是终态收据镜像，不能当作实际 Linux 数据目录传给加载器。既有一次终态 stat 审计报告 8,850 个不同缓存路径、49,191,686,647 bytes，最大单帧 6,134,285 bytes；这不是本适配器已执行真实全池读取的证据。

## 调用接口

```python
from pool_index import load_pool_index, balanced_plan, assert_coverage, plan_summary, load_frame

index = load_pool_index('/data/CJH/AGS-V6Direct/outputs/gaussiangpt_neoverse_fulltrain_pool_20261031_v1')
plan = balanced_plan(index, seed=42, batch_size=4)
assert_coverage(index, plan)
metadata = plan_summary(index, plan)

for batch in plan:
    actual_batch_size = len(batch)
    for ref in batch:
        payload = load_frame(ref, device='cpu')
        # 调用方使用完整帧，随后释放；本模块不执行 loss/backward/step。
        del payload
```

`load_pool_index(pool_root) -> tuple[FrameRef, ...]` 返回元数据。每个 `FrameRef` 包含 clip、source_group、历史/当前角色、frame、PTS、cache 路径/bytes/SHA、native GS SHA、片段归一化 JSON、旧加载器路径和终态池 SHA。`ref.normalization` 返回独立字典；`ref.legacy_row()` 返回旧加载器使用的单帧记录。

该入口先要求精确 DONE 与 POOL_COMPLETE，再验证 146 个成功子任务、CPU 见证、源码清单、INPUT_POOL、reuse admission 和各片段准备收据。新导出的归一化、mapping、quality `frames` 取自新片段目录；复用 tower10 取旧 `quality/<clip>.json`，复用 tower17 取旧 `prepare/<clip>.json` 的 `rows`。每个使用到的元数据文件校验实际字节和 SHA。缓存 SHA 必须同时匹配全局 files、片段输出记录和 quality row，并按文件顺序用固定 1 MiB 读块重算实际 SHA。只按冻结名单和 `000..176` 构造路径，不扫描目录增补样本。

索引不反序列化缓存，也不读取历史状态张量；全部缓存校验会产生约 49.2 GB 的磁盘读取，应由后续运行入口给出明确 CPU/I/O 超时。索引结束前复核已读取文件的大小、inode、mtime/ctime，拒绝准入期间变化。该过程验证 native GS 的收据 SHA 与每帧 provenance 一致，不重新解包 native NPZ、不重做渲染或 teacher 推理；未使用的旧状态和模型权重不在此模块重新读取。

`load_frame(ref, device='cpu')` 每次只读取一帧：同一份 bytes 先验证大小/SHA，再交给 `torch.load(BytesIO(...), map_location='cpu', weights_only=True)`。它执行精确绑定旧 `sources.py` 中原有 shape、全点统计、PTS、归一化、有限值、整数类型、索引与体素坐标唯一性检查，之后才按原逻辑移动所需张量到指定 device。隔离模块副本只把 `ALL_CLIPS`、`TRAIN_CLIPS`、`ROLES` 参数化为这 50 个 TRAIN 片段；不改旧文件，不修改另一个实验已导入的模块。实际 Torch 解包失败时直接失败，不换版本、不降配。

每片段保留原 177 个相邻帧、29.97 FPS、448×252 teacher、每帧 112,896 个原 GS 的全部聚合统计。mapping/quality PTS 使用 `t/29.97`，容差 `1e-9`；与 ffprobe 小数测量比较用 `1e-6`。归一化保留各片段第一帧确定且绑定的中心零与正二次幂尺度，不用未来帧。真实池的49片scale为0.5，另1片为0.25，不能全池硬写0.5；原validate_normalization仍检查尺度合法性。加载器不产生或改变模型的 64 维连续状态，不新增量化。

## 完整轮转与尾批

`balanced_plan(index, seed, batch_size)` 返回一轮完整无放回计划。先按 clip/frame 排序，再由种子独立打乱各片段帧顺序；每轮从每个仍有帧的片段取一帧。对于完整池，每连续 50 个样本恰好覆盖 50 个不同片段，各片段最终曝光 177 次。输入索引顺序变化不会改变同一种子的计划。

函数支持一般的不等长片段集合，但完整池准入仍严格为 50×177。返回的 batch 不填充、不复制、不丢尾。B4 的 2,213 批中最后一批是 2 帧；调用方做串行梯度累积时必须除以 `len(batch)`。一次函数调用只生成一轮，第二轮应显式再次调用并记录种子和计划 SHA，不默默循环样本。

`assert_coverage(index, batches)` 拒绝遗漏、重复、外来样本和同身份但被篡改的 FrameRef。`plan_summary` 返回计划 SHA、实际批大小、clip/source-group 曝光、不同 cache 路径的字节总量和最大单帧大小。clip 等权不等于 source group 等权；两类曝光分别报告。

## CPU 测试与未覆盖条件

仅运行本地合成测试：

```powershell
C:/Python314/python.exe -B -m unittest discover -s experiments/gaussiangpt_neoverse_fullpool_codec_20261031_v1 -p test_pool_index.py -v
```

测试在系统临时目录创建 50×177 个小型合成缓存，复制本地固定源码/INPUT_POOL，并明确替换合成终态的测试 SHA。它覆盖完整准入链、50 片段角色与帧覆盖、确定性/均衡性、实际尾批、遗漏/重复/篡改拒绝、失败终态、源码变化、同大小缓存损坏，以及旧加载器的同 bytes→BytesIO 解包入口。合成缓存不是 Torch 张量，测试不会将其冒充真实训练数据或 kernel 见证。

真实全池索引、真实 Torch 张量载入、原生稀疏算子/渲染前反向、显存峰值、吞吐、训练与 fresh encode 尚需独立执行和见证；本模块没有执行这些工作。新 encoder 训练后必须对 8,850 帧重新编码并绑定 checkpoint，随后才能重建 LDM 数据与统计。额外 tower14 若用于评测，应由独立的旧绑定入口加载，保持其已暴露的 eval-only 身份，不能拼入本训练索引或训练统计。
