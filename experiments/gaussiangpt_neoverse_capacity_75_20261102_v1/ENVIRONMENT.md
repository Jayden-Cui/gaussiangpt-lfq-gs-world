# 75 服务器环境与独立复核

所有目标机实验文件均位于 `/data/CJH/1`。解释器入口为 `/data/CJH/1/bin/python`：该脚本使用目录内的 glibc 2.35 加载 `/data/CJH/1/envs/gaussiangpt/bin/python`。系统自带 glibc 2.31、NVIDIA 550.78 驱动及其他已有环境未修改。直接调用环境内 ELF Python 会导致 MinkowskiEngine 缺少 `GLIBC_2.34`，应使用上述入口。

环境来自 134 的固定 Python 3.10.21、PyTorch 2.8.0+cu129 和原 MinkowskiEngine／gsplat 扩展。数据与参考源码为 `/data/CJH/1/assets`，绑定清单 SHA256 为 `3702a70b2f13dc9f214700f5f6601d873829bbc6ba4ee289933200c87fc14b5e`。运行脚本在导入 PyTorch 前设置线程数、缓存和临时目录。训练协调器的子进程也必须使用 `/data/CJH/1/bin/python`。

`ENV_NATIVE_01.json` 已记录一轮真实双视角 GS 解码／渲染／状态特征反向传播：输出 `[2,252,448,3]`，51,364 个 Gaussian，有限非零输入梯度，原状态 codec 保持冻结。它没有执行优化器更新，也不是动作质量结果。

独立复核者在 Windows PowerShell 执行以下命令，原样执行一次即可。134 上已认证的 SSH 复用连接负责连接 75；命令不包含密码。GPU3 专用于本次复核，在独立复核结束前不启动其他 GPU3 作业。输出文件必须不存在；如命令失败，不改入口、不安装包，报告错误。

```powershell
ssh -T -o BatchMode=yes -o ConnectTimeout=10 172.19.58.134 "ssh -T -S /data/CJH/AGS-V6Direct/outputs/.capacity75.sock -o BatchMode=yes amax@172.19.58.75 env CUDA_VISIBLE_DEVICES=3 PYTHONDONTWRITEBYTECODE=1 /data/CJH/1/bin/python -B /data/CJH/1/experiments/gaussiangpt_neoverse_capacity_75_20261102_v1/env_witness.py --output /data/CJH/1/experiments/gaussiangpt_neoverse_capacity_75_20261102_v1/ENV_AGENT_WITNESS.json"
```

验收：进程退出码 0，打印 `NATIVE_ENV_WITNESS` 且 `status=passed`；设备为 RTX 4090，`render_shape=[2,252,448,3]`，`feature_gradient_norm>0`，`native_codec_frozen=true`，`training_updates=0`。独立复核只确认该环境调用与文档一致；四种维度模型还要分别执行训练前检验。
