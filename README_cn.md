# 轮腿机器人强化学习 | [English](README.md)

基于 Isaac Lab 与仓库内定制的 RSL-RL，训练我们自己的 Motor43、Motor35 轮腿机器人。
原版 TRON1 的 PF/SF/WF 演示任务已移除；自己的 locomotion、recovery 和串联 play 按模型组织。

## 目录

```text
exts/bipedal_locomotion/
├── bipedal_locomotion_motor43/
│   ├── assets/          # urdf、usd、source、validated（旧策略配套模型）
│   ├── locomotion/     # 原有行走/站立设计
│   ├── recovery/       # 渐进起身及已有自定义起身实验
│   └── play/           # 串联场景、模型选择和坐标约定
├── bipedal_locomotion_motor35/
│   ├── assets/          # 自己的 urdf、usd、source 和电机参数
│   ├── locomotion/
│   ├── recovery/
│   └── play/
└── bipedal_locomotion_common/
    ├── wrappers/       # RSL-RL 环境与接口
    └── play/           # FK、切换状态机、动作混合
```

Motor35 保持既有继承关系：复用 Motor43 的基础任务/MDP，再覆盖新模型参数、轴向、力矩限制和域随机化；具体任务、注册及资产各归自己的包。此次整理没有重新设计奖励、观测、课程或控制参数。

`scripts/` 只放训练、播放和姿态生成等任务运行入口；检查、评估、分析、维护工具及测试统一放在 `tools/`，测试目录为 `tools/tests/`。保留的权重在 `checkpoints/`，视频在 `videos/`。`logs/` 和 `outputs/` 是运行产物，安装生成的 `*.egg-info`、Python 缓存和编辑器缓存也不提交。

## 模型在哪里

以下路径相对于 `exts/bipedal_locomotion/`：

| 内容 | 路径 |
|---|---|
| Motor43 原始 CAD 导出压缩包 | `bipedal_locomotion_motor43/assets/source/轮腿总装111.zip` |
| Motor43 当前适配后的 URDF | `bipedal_locomotion_motor43/assets/urdf/motor43.urdf` |
| Motor43 已验证串联权重配套的 URDF/USD | `bipedal_locomotion_motor43/assets/validated/` |
| Motor35 原始模型、电机说明、惯量 | `bipedal_locomotion_motor35/assets/source/Motor35_URDF/` |
| Motor35 训练用适配 URDF | `bipedal_locomotion_motor35/assets/urdf/robot.urdf` |
| 两种模型的 USD | 各包的 `assets/usd/` |

`WF_TRON1A.urdf` 是适配旧工程时沿用的文件名，内容已经是我们自己的机器人，原始文件名为 `轮腿总装111.urdf`。现在改名为 `motor43.urdf`，XML 内容及 mesh 不变。USD 内部仍保留 `WF_TRON1A` 命名，以保持已验证的二进制资产和引用不变；它不表示仍在使用原版 TRON1 模型。

本次提交把可复现的 Motor43 本地权重放在 `checkpoints/motor43/`，并把三种 Motor43 回放视频放在 `videos/motor43/`。Motor35 目录里只保留标明 `preflight_zero_policy` 的 smoke 权重和对应视频；它们用于验证任务入口，不代表已训练成功。权重来源和 SHA256 见 `checkpoints/manifest.json`。

Motor35 可以用自己的任务开始训练：运行的是经过名称映射、mesh 路径整理和轮碰撞处理的 URDF；源模型的质量、惯量、关节轴和角限位保留。原始 CAD 文件不要直接替换 Motor43 资产。Motor35 已通过工程预检，完整新策略仍需训练。[参数、域随机化与训练顺序](docs/motor35.md)。

## 安装

服务器 clone 后，在仓库根目录使用已安装 Isaac Sim / Isaac Lab 的 Python 执行下面命令。本次验证版本为 Isaac Sim 5.0 / Isaac Lab 2.2.1；将 `SIM_PY` 换成服务器上的路径即可。

```bash
SIM_PY=/home/myoukin/isaacsim/python.sh
"$SIM_PY" -m pip install -e exts/bipedal_locomotion
"$SIM_PY" -m pip install -e rsl_rl
"$SIM_PY" tools/prepare_motor35.py
```

必须使用仓库中的 RSL-RL（包含历史编码器和 `rsl_rl.runner`），训练/播放脚本已优先加载本仓库版本。

## 主要任务

| 功能 | Motor43 | Motor35 |
|---|---|---|
| 行走/站立训练 | `Isaac-Motor43-Locomotion-v0` | `Isaac-Motor35-Locomotion-v0` |
| 行走播放 | `Isaac-Motor43-Locomotion-Play-v0` | `Isaac-Motor35-Locomotion-Play-v0` |
| 渐进起身训练 | `Isaac-Motor43-Recovery-Progressive-v0` | `Isaac-Motor35-Recovery-Progressive-v0` |
| 170–180° 起身播放 | `Isaac-Motor43-Recovery-Inverted-Play-v0` | `Isaac-Motor35-Recovery-Inverted-Play-v0` |
| 起身→行走场景 | `Isaac-Motor43-Recovery-Locomotion-Play-v0` | `Isaac-Motor35-Recovery-Locomotion-Play-v0` |

旧 `Isaac-Limx-*` ID 不再提供。Motor43 的其他自定义 GetUp/Fallen 实验仍保留，见包内注册及 [Motor43 使用说明](docs/motor43.md)。本地旧日志和临时姿态库已经清理；选定权重已移入 `checkpoints/`。Fallen 实验按文档重新生成姿态库。

## 开始训练 Motor35

```bash
"$SIM_PY" scripts/rsl_rl/train.py \
  --task Isaac-Motor35-Locomotion-v0 --num_envs 512 --headless \
  --max_iterations 20000 --save_interval 100 --run_name motor35_dr
```

先验证新行走策略的零速站立、移动和转向，再把匹配的新模型权重填入下方，初始化起身课程：

```bash
MOTOR35_LOCO_CKPT="/absolute/path/to/validated_motor35_locomotion.pt"
"$SIM_PY" scripts/rsl_rl/train.py \
  --task Isaac-Motor35-Recovery-Progressive-v0 --num_envs 512 --headless \
  --resume True --checkpoint_path "$MOTOR35_LOCO_CKPT" \
  --reset_optimizer --getup_stage 0 \
  --max_iterations 30000 --save_interval 100 --run_name motor35_recovery_dr
```

环境数按显存调整。`max_iterations` 是本次新增迭代数；后续同任务断点续训不加 `--reset_optimizer --getup_stage 0`，以恢复优化器和课程状态。旧 Motor43 权重不能用于 Motor35。

## 串联 play

Motor43 默认使用包内 `assets/validated/` 模型、对应的默认姿态以及 `checkpoints/motor43/` 中的两份权重；clone 后可直接运行，不依赖旧日志或 Git 历史。三段视频的复现命令见 [Motor43 视频说明](videos/motor43/README.md)。

```bash
"$SIM_PY" scripts/rsl_rl/play_recovery_locomotion.py \
  --robot motor43 --duration 30 --forward_speed 0.3
```

若两份策略是在 Motor43 当前 `assets/urdf/` 对应模型上训练，传入它们并加 `--current_asset`。
Motor35 必须传入新模型训练的两份权重：

```bash
"$SIM_PY" scripts/rsl_rl/play_recovery_locomotion.py \
  --robot motor35 \
  --recovery_checkpoint /absolute/path/to/motor35_recovery.pt \
  --locomotion_checkpoint "$MOTOR35_LOCO_CKPT" \
  --duration 30 --forward_speed 0.3
```

`--forward_speed` 对 Motor43 是 body-y，对 Motor35 是 body-x。切换仍使用 FK、重力投影、腿长/轮轴方向和双轮接触判据，连续达标后平滑混合物理动作目标；失稳则平滑退回起身。Motor35 的坐标、轮速与动作缩放单独配置。两个模型各自的 locomotion/recovery 都是单帧 28 维、10 帧历史 280 维、actor 输入 34 维、动作 8 维；维数相同不代表不同模型的权重能混用。

Motor43 的逐回合力矩记录仍用 `scripts/rsl_rl/play_recovery_torques.py`，详见 [力矩记录说明](docs/motor43.md)。它是 Motor43 专用入口。

## 验证与 Git 提交

整理前后 23 个已有环境/runner 配置逐项对比，仅包名、路径及相应文件指纹变化；保留原有物理参数、奖励、课程和 PPO 参数。验证记录见 [目录迁移检查](docs/reorganization.md)。
源模型、适配 URDF、mesh、USD 和串联 play 所需快照均随包存放。`checkpoints/` 和 `videos/` 允许直接由 Git 跟踪；旧运行痕迹已清理，后续训练仍将新的运行记录写入被忽略的 `logs/`。训练成功后，将选定权重复制到对应的 `checkpoints/<robot>/` 再提交。Motor35 重建工具可从包内源文件生成训练资产。

## 许可证

保留原工程 [Apache 2.0](LICENCE) 及代码中的原始归属说明。
