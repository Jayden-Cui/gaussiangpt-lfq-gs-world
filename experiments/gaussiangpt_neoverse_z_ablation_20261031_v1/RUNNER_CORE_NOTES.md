# CPU runner 核心

本轮仅新增 `runner_core.py`、`test_runner_core.py` 和本文件。不加载codec、模型、MinkowskiEngine或真实LFQ库，不执行CUDA、训练、部署或预算冻结，也不重复现有adapter的9项测试。

## 接口

- `candidate_plan()`：返回 `status=candidate, budget_frozen=false` 的800步B4计划及统计。与v7相同，seed20261031，采样器使用seed+17；每42步覆盖一次168窗。计划SHA为 `d80f965a9fbd7c4906327691e0a671488b68d5c8d32f62a8842107163c58c984`，19完整周期加2步；每臂3200窗口、32000帧及28800未来曝光。此函数没有启动/预算授权作用。
- `make_action_packet(codes, mode=..., starts=..., s_ref=..., source=...)`：只接受CPU float32实际码 `[N,9,16]`。明确记录start行序、1…9 horizon、mode、共同固定scale、来源、码张量摘要和metadata摘要。LFQ码须为±1；连续不转ID、不舍入、不重新归一化。调用GPU adapter后由上层显式将实际码detach到CPU再传入。
- `write_action_packet(path, packet)`：独占创建，返回文件SHA/字节数，不覆盖已有文件。`read_action_packet(path, expected_file_sha256=..., expected_mode=..., expected_source=..., expected_s_ref=..., expected_starts=...)` 从同一份已核SHA字节以 `weights_only=True/map_location=cpu` 读取，并重新验证张量和metadata。expected_starts可省略，其余期望值必须显式给出。
- `control_mapping` / `select_control_codes`：correct使用本start；fixed_bundle固定start0完整9码；fixed_single固定start0,h1码广播所有h；shuffle42/84/126按 `(recipient+offset)%168` 整束移交。输出包含接收方10帧身份、供体start/h，实际码保持逐位相同。显式按packet.starts查找，不能把存储行号当start。缺供体立即拒绝，不补齐、不用零码代替。
- `blind_window(window)`：接受CPU完整T10 state字典，输出每槽均为S0 geometry/features的独立副本，PTS/source_pts/frame_index仍保留接收方真实值；不重造时间。使用字段白名单去除潜藏未来几何的未知metadata，并标记 `blinded_posterior=true, state_semantics=synthetic_initial_repeated`。这是posterior输入控制，不能作为真实缓存重新署名或替代correct动作包。
- `evaluation_coverage()`：完整168窗×9h=1512未来曝光、176不同未来帧；h0另168曝光，整个时间轴0…176。保留物理帧重叠计数，指定后续统计先在同一物理帧内平均，不把1512曝光误称1512独立帧。

## 包的来源边界

source至少包含四个明确的小写SHA256：`model_checkpoint_sha256`、`codec_binding_sha256`、`state_manifest_sha256`、`adapter_sha256`，可加入JSON格式的common-init/其他来源字段。来源校验仅确认声明、期望值和文件内容一致；不会读取这些SHA指向的codec，也不会由SHA存在推断验收。所有包固定 `codec_admission_verified=false`，实际准入由未来codec_binding负责。

本层保存actual float codes，足以供 `decode_action_codes` 使用；不新增或解释LFQ的bit→ID顺序。实际LFQ IDs、diagnostic_ids及ID/sign一致性由现有action_bottleneck packet/quantizer接口另存另验，不能将continuous诊断ID当作本层输入。adapted_held和真实未来state参考需要最终codec，故不在此CPU核心生成；blinded需要后续真实posterior重新计算动作。

## CPU验证

本目录执行 `python -B -m unittest -v test_runner_core.py`。测试使用真实小CPU tensor和临时文件，覆盖：候选计划完整周期与不冻结标志；连续float32文件往返；错误mode/source/scale/fileSHA及码内容篡改拒绝；LFQ符号码和shape；供体整束映射、固定单码、乱序存储、缺供体拒绝；盲化不含未来坐标/features且接收方PTS不变；9h/176物理未来帧覆盖。

这些测试不证明codec或原生renderer通过，不运行新的训练预算。预留职责止于CPU接口；既有DESIGN、adapter和RUNNER_PLAN不做修改。

本轮实际结果：本地 `C:\Python314\python.exe` 执行上述测试，6项通过、0 failures/errors/skips，测试耗时0.136秒、exit0；工具chunk为 `a505a2`。这是本轮CPU实现验证，不是新的正式manifest冻结或GPU见证。
