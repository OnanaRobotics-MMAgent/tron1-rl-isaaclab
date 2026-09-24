# 中文 | [English](README.md)

# tron1-rl-isaaclab

基于 [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) 的 LimX **TRON1** 双足机器人强化学习训练栈，使用 PPO 训练 locomotion 策略。本仓库扩展了 Isaac Lab 模板，支持 TRON1 机器人变体的 Sim-to-Real 训练。

## 环境要求

- Isaac Sim + Isaac Lab，且 isaaclab / isaaclab_tasks / isaaclab_rl 可被 import
- 使用与 Isaac Sim 配套的 Python（本机为 Isaac Sim 5.0 / Python 3.11）
- GPU（推荐 >= 12 GB 显存，用于多环境训练）

## 安装

```bash
# 1. clone 仓库
git clone https://github.com/limxdynamics/tron1-rl-isaaclab.git
cd tron1-rl-isaaclab

# 2. editable install extension 与 vendored rsl_rl
pip install -e exts/bipedal_locomotion
pip install -e rsl_rl
```

> **说明：** 机器人模型资产（USD 文件）已内置在 `exts/bipedal_locomotion/bipedal_locomotion/assets/usd/` 中，覆盖 SF_TRON1A 和 WF_TRON1A 两种变体，无需额外下载模型文件。

## 训练

任务 ID 在 exts/bipedal_locomotion/ 中注册。

```bash
# Solefoot (SF)
python scripts/rsl_rl/train.py --task Isaac-Limx-SF-Blind-Flat-v0 --num_envs 4096 --headless

# Wheelfoot (WF)
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-Blind-Flat-v0 --num_envs 4096 --headless
```

常用选项：

- --resume True --checkpoint_path <path> -- 从指定的 .pt checkpoint 恢复训练
- --video -- 启用视频录制
- --max_iterations N -- 覆盖最大迭代次数

日志路径：logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/

## 翻倒起身（Recovery）

代码位于 `exts/bipedal_locomotion/bipedal_locomotion/tasks/recovery/`，与 `locomotion` 平级，包含环境配置、`mdp/` 和 `agents/`。复用 WF 机器人与基础配置；旧任务 ID、checkpoint 和 RSL-RL 兼容修复保留。腿关节使用有限角度及物理限位，轮子可连续转动。

### 当前轮腿模型：渐进式 recovery

新任务 `Isaac-Limx-WF-Recovery-Progressive-v0` 使用当前 WF USD，只在此任务开启自碰撞。沿用最新行走/站立策略的非零关节参考姿态、8 维动作缩放、观测与历史编码器、执行器和固定摩擦；速度指令设为零。腿目标受 USD 物理限位约束，轮子保留连续旋转。

课程共 36 级：0–5° 起步，然后 0–10°、5–15°，每次上限增加 5°，直至 170–180°。每级采样前后左右倾倒和随机 yaw；20% 回合复习较简单级别。每个统计窗口至少 4096 个当前级别的完成回合，总成功率 ≥80%、四个方向各 ≥70%（每方向至少 128 回合），连续 3 个窗口达标才升级。未达标就留在原级训练。每回合最多 12 秒，双轮支撑、身体无支撑接触、倾角 <15°、高度 0.18±0.04 m、线速度 <0.25 m/s、角速度 <0.5 rad/s 并连续维持 2 秒才成功。

奖励鼓励抬升、扶正、双轮支撑及稳定站立，允许恢复时身体接地；姿态和静止奖励主要在接近站立时生效。reset 根据适配 URDF 的碰撞凸包及站立关节姿态做正运动学，再按倾角计算离地 5 mm 的高度，不套用旧任务的 12 cm 高度下限。大角度阶段是近地面倾倒/翻倒初态，尚不覆盖任意折叠关节的自然静置姿态库。

当前仓库 logs 仅包含下面的 `model_10000.pt`，其 `task_state` 为空，是最新行走/站立 checkpoint；历史 recovery 的 `model_21000.pt` 未包含在当前 checkout 中。首次迁移用下面命令加载网络与编码器，重置优化器，并从第 0 级开始：

```bash
cd /home/myoukin/tron1-rl-isaaclab
export PYTHONPATH="$PWD/exts/bipedal_locomotion:$PWD/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
/home/myoukin/isaacsim/python.sh scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-Recovery-Progressive-v0 \
  --num_envs 2048 --headless \
  --resume True \
  --checkpoint_path "$PWD/logs/rsl_rl/wheel_leg_abad_kp8/2026-09-22_22-33-55_torque_soft_05/model_10000.pt" \
  --reset_optimizer --getup_stage 0 \
  --max_iterations 20000 --save_interval 100 --run_name self_collision_curriculum
```

`max_iterations` 是新增训练迭代数。输出到 `logs/rsl_rl/wheel_leg_recovery_progressive/`，TensorBoard 的 `Episode/Curriculum/recovery/level` 及 `GetUp/stage_XX/*` 显示课程和成功率。课程升级时 runner 也会自动保存 checkpoint。后续断点续训改用该目录内的新 checkpoint，并去掉 `--reset_optimizer --getup_stage 0`，即可恢复优化器和课程统计。播放使用 `Isaac-Limx-WF-Recovery-Progressive-Play-v0`，用 `--getup_stage 17` 检查 80–90°，`--getup_stage 35` 检查 170–180°。旧 recovery checkpoint 的动作缩放/关节参考姿态与新任务不同，不能仅因网络维度相同就直接混用。

新任务实现验证：28 项相关 CPU 测试通过；加载 `model_10000.pt` 后完成 64 环境、30 次 PPO 更新并保存课程状态。该短测仅验证训练链路，没有训练至恢复收敛。现有完整 `test_action_bounds.py` 中的原始模型重生成测试依赖仓库外 `/home/myoukin/轮腿总装111/轮腿总装111.urdf`，本机缺少该文件，未通过该项；本次修改涉及的动作边界测试均通过。

### 170–180° 翻倒 play 与逐回合力矩记录

新增 `Isaac-Limx-WF-Recovery-Inverted-Play-v0`：单机器人，固定最后一级 170–180°，开启自碰撞，禁用课程晋级与简单姿态回放。专用入口连续播放确定性策略并按完整回合记录（默认 20 回合）：

```bash
cd /home/myoukin/tron1-rl-isaaclab
/home/myoukin/isaacsim/python.sh scripts/rsl_rl/play_recovery_torques.py \
  --checkpoint_path "$PWD/logs/rsl_rl/wheel_leg_recovery_progressive/2026-09-23_19-56-33_self_collision_curriculum/model_30000.pt" \
  --episodes 20
```

默认打开仿真窗口并按实时速度播放；加 `--headless` 可无窗口快速导出。无需手动设置 PYTHONPATH。`--seed` 控制倾角/方向采样；`--output_dir` 可指定一个尚不存在的目录，否则自动保存到 `logs/recovery_torques/<时间戳>/`。Ctrl+C 保存当前部分回合，并标记不完整。

力矩包含两个分别命名的通道，单位 N·m：

- `pd_estimate`：`robot.data.applied_torque`，当前隐式 PD 的限幅后估算值，**不是 PhysX 实测电机驱动力矩**。
- `solver_joint_effort`：`root_physx_view.get_dof_projected_joint_forces()`，求解器返回的关节轴向力矩，可能含约束/接触响应，**不等于单独电机输出**。关节侧数据未折算减速器前电机轴力矩。

每个 0.005 s 物理步后、自动重置前读取这两路信号；PD 估算对应该物理区间开始时，求解器值对应区间结束。每 4 点取有符号算术平均，得到 0.02 s 控制周期曲线，不跨回合平均。峰值、均值、绝对值均值、RMS 全部从原始物理步样本计算，防止正负抵消或平滑掩盖峰值。

输出文件：

- `episode_0000/` 等每回合目录：`physics_torques.csv`（200 Hz 原始值）、`mean_torques.csv` / `.png`（50 Hz 平均曲线）、`summary.json`（每关节统计、初始角度/方向、成功/失败原因）。覆盖 abad/hip/knee/wheel 左右全部 8 个关节。
- `maxima.csv`：所有回合各关节的均值、绝对值均值、RMS、最小值、最大值、绝对峰值。
- `overall_maxima.csv`：全次播放各关节的最大/最小值、绝对峰值及峰值所在回合；包含已记录的部分回合。
- `episodes_mean.csv` / `.png`：所有完整回合（成功与失败均包含）按恢复开始时间对齐的有符号平均曲线。结束后的回合不补零，CSV 的 `contributing_episodes` 表示各时刻参与平均的回合数。不同倾倒方向的正负力矩可能抵消，因此电机负载评估请同时看逐回合绝对峰值、绝对值均值与 RMS。
- `metadata.json` / `summary.json`：权重路径与 SHA256、种子、信号定义、采样周期及回合信息。

已有 `scripts/rsl_rl/play.py --task Isaac-Limx-WF-Recovery-Inverted-Play-v0` 也可只播放该姿态；要保存力矩请用上述专用入口。

### 起身后平滑切换 locomotion

串联场景 `Isaac-Limx-WF-Recovery-Locomotion-Play-v0` 从 170–180° 翻倒开始，站立后不自动重置。必须使用双策略入口（普通 `play.py` 只加载一个策略）：

```bash
cd /home/myoukin/tron1-rl-isaaclab
/home/myoukin/isaacsim/python.sh scripts/rsl_rl/play_recovery_locomotion.py \
  --duration 30 --velocity_y 0.3 --yaw_rate 0
```

默认使用 recovery `2026-09-23_19-56-33_self_collision_curriculum/model_30000.pt` 和 locomotion `2026-09-22_22-33-55_torque_soft_05/model_10000.pt`；可用 `--recovery_checkpoint` / `--locomotion_checkpoint` 指定。加 `--headless` 无窗口运行，`--video` 保存视频，`--num_envs 16` 并行检查。`--velocity_y 0` 可检查切换后原地平衡。全程保持自碰撞开启，包括 locomotion 阶段。

注意：当前分支合入的 `bc5898b` 修改了髋关节限位、站立偏置和碰撞模型，与上述两份权重训练时不一致。因此这个专用入口默认从本地 Git 的 `8f9c9db` 导出独立模型快照到本次输出目录，并使用训练时的站立偏置；不修改当前仓库模型。若换用已针对当前新模型训练的权重，添加 `--current_asset`。仅观测维度一致不能保证模型或偏置变化后的兼容性。

判断使用适配 URDF 的正运动学：编码器关节角计算轮子中心、髋到轮心的有效腿长、轮轴方向；与 IMU 重力投影组合，计算沿重力方向的腿部伸展和推定支撑高度。连续 0.3 秒满足：机身倾角 <20°、两侧推定高度 0.14–0.23 m/差值 <3.5 cm、有效腿长为标称的 0.70–1.35 倍、腿部向下伸展 >7 cm、轮轴偏离水平 <35°、两轮轴夹角 <45°、两轮未交叉且横向间距 >8 cm、角速度 <0.8 rad/s，并且双轮各有 >3 N 垂直接触力，才开始切换。推定高度不是世界坐标真实高度；接触条件用于避免空中误判。此版本接触力来自仿真传感器，部署需要对应的接触检测。重力投影无法提供绝对 yaw，方向判据是相对重力及两轮之间的姿态。

切换用 `smoothstep` 在默认 1 秒内混合两策略的**物理关节目标**，腿部目标限制变化率 3 rad/s、轮速目标限制变化率 80 rad/s²，之后重新换算成动作。先切到 locomotion 零速站立，再用 1 秒平滑增加目标速度。纯 recovery 阶段保留原策略的原始动作反馈；两策略各自使用自己的历史编码器，共享实际提交动作形成的观测历史。持续 0.15 秒明显失稳（倾角 >40°、腿部塌缩或丢失双轮支撑）会以 0.3 秒反向混合回 recovery，避免直接硬切。`--ready_hold` 和 `--blend_time` 可调整防抖与过渡时间。

两份指定权重的推理接口一致：单帧 28 = 角速度3 + 重力投影3 + 腿关节相对角6 + 全关节速度8 + 上次动作8；10帧历史280，经各自 encoder 输出3；actor 输入34 = 历史特征3 + 当前观测28 + 速度指令3，输出8（腿位置6 + 轮速度2）。动作缩放相同：腿0.12、轮53.333，使用相同站立关节偏置。两份 checkpoint 的 critic 输入均214，但 play 不需要 critic；当前环境材质形状数变化可能改变特权观测长度，双策略入口只严格加载 actor/encoder，避免把 critic 维度变化误判成策略不兼容。

输出在 `logs/recovery_handoff/<时间戳>/`：`trace.csv` 保存状态、混合比例、几何指标、实际动作和机身速度；`transitions.json` 保存切换/退回/重置；`metadata.json` 保存参数与接口维度；`validation.json` 保存编码器 FK 与仿真轮心的最大误差（仿真真实位置只用于检查 FK，不参与切换）。状态编号为 0=recovery、1=混合到行走、2=locomotion、3=退回起身。该 play 不修改训练权重。 验证：种子20260924，16环境各15秒，16/16完成切换，无重置或退回；1.02–2.00秒开始过渡、2.02–3.00秒完成接管。目标体轴y速度0.3 m/s，最后5秒平均实测0.347 m/s、最大倾角1.23°；FK最大误差2.52微米。该结果是指定权重和训练模型快照上的短测，不代表当前新限位模型已经验证。

### 历史 recovery 实验（旧模型）

以下为旧模型记录与命令，不代表当前轮腿模型已经学会摔倒恢复。历史 `model_21000.pt` 曾通过原任务 33 阶段仿真评估（99.91%，16,896 回合）；该权重未包含在当前 checkout。

在仓库根目录执行（以下为本机路径）：

```bash
SIM_PY=/home/myoukin/isaacsim/python.sh
export PYTHONPATH="$PWD/exts/bipedal_locomotion:$PWD/rsl_rl${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINT="$PWD/logs/rsl_rl/wf_tron_1a_getup_bounded/2026-09-17_12-43-05_continuous_e10a7a122f48/model_21000.pt"

# 按需继续训练，另存新 checkpoint
"$SIM_PY" scripts/rsl_rl/train.py \
  --task Isaac-Limx-WF-GetUp-Bounded-v0 --num_envs 4096 \
  --resume True --checkpoint_path "$CHECKPOINT" \
  --max_iterations 2000 --save_interval 100 --headless

# 播放旧模型并录制约 15 秒视频
"$SIM_PY" scripts/rsl_rl/play.py \
  --task Isaac-Limx-WF-GetUp-Bounded-Play-v0 --checkpoint_path "$CHECKPOINT" \
  --num_envs 1 --getup_stage 32 --getup_fixed_stage \
  --headless --video --video_length 750
```

视频写入 checkpoint 同级的 `videos/play/`，重复录制前需保留已有同名文件。独立评估使用 `scripts/rsl_rl/evaluate_getup.py`（参数见 `--help`）。自动训练入口为 `manage_getup.py start/status/stop`：每 2000 次更新评估，自动晋级和记录；已有状态目录会恢复原进度，达标后自动停止。评估记录位于 `logs/getup_continuous/`。

### 固定倒置起身

`Isaac-Limx-WF-GetUp-Inverted-v0` 每回合以随机方向和偏航角从 170°–180° 倒置姿态重置，贴近地面后以中性动作等待连续 0.1 秒接触，再交由策略控制腿和轮。目标机身高度为 0.18 m；主要高度奖励乘以 `clamp((up_z + 1) / 2, 0, 1)^2`，其中 `up_z` 是机身上方向与世界向上方向的有符号点积。完全倒立时系数为 0，侧躺时为 0.25，完全朝上时为 1；靠近倒立时缓慢增加，靠近朝上时增加更快。其他奖励与原任务的稳定站立、接触和关节限位约束沿用。这个任务与原倾角课程使用独立的实验目录，不会修改原任务的 checkpoint。

```bash
python scripts/rsl_rl/train.py --task Isaac-Limx-WF-GetUp-Inverted-v0 \
  --num_envs 512 --max_iterations 100 --save_interval 25 --headless

python scripts/rsl_rl/check_getup.py \
  --task Isaac-Limx-WF-GetUp-Inverted-Play-v0 --num_envs 16 --headless

python scripts/rsl_rl/play.py --task Isaac-Limx-WF-GetUp-Inverted-Play-v0 \
  --checkpoint_path logs/rsl_rl/wf_tron_1a_inverted/<run>/model_100.pt \
  --num_envs 16 --eval_episodes 2 --headless
```

2026-09-23 的从零训练试跑（512 环境、100 次更新、1,228,800 环境步）保存于 `logs/rsl_rl/wf_tron_1a_inverted/2026-09-23_18-58-02_inverted_100/`。这次试跑使用的是改动前 `up_z > 0` 的高度奖励：触地与控制释放检查为 16/16，但高度奖励仍为零，独立评估成功率为 0/32；`model_100.pt` 只是旧奖励下的可复现实验 checkpoint，不是已学会起身的策略。新奖励尚未重新训练评估。

### 随机关节倒地（Fallen）

`recovery/fallen_pose_env_cfg.py` 定义被动姿态生成环境及新任务 `Isaac-Limx-WF-Recovery-Fallen-v0`（播放加 `-Play`）。生成时关闭腿的位置驱动，随机合法关节和身体朝向，自然落稳后记录状态；训练时恢复正常执行器，从姿态库 reset。数据文件位于 `data/recovery/fallen_v1.pt`，请与日志分开保留。

训练集 4,096 个、测试集 1,536 个状态独立生成并检查重复。按实际关节偏离默认姿态的程度分三档自动晋级，混合 30% 原任务姿态；这三档是关节姿态难度，不是原来的身体倾角阶段。每 2,000 次更新评估完整测试集及原任务，退化暂停，连续两次达标停止。测试集用于开发验收，不代表任意姿态或实机都已覆盖。

```bash
# 生成新姿态库（已有文件时拒绝覆盖）
"$SIM_PY" scripts/recovery/generate_fallen_poses.py --headless \
  --train_poses 4096 --test_poses 1536 --seed 20260920 --output data/recovery/fallen_v1.pt

# 首次启动后台训练；已有状态则恢复，重复启动不会另开进程
"$SIM_PY" scripts/recovery/manage_recovery.py start \
  --checkpoint "$CHECKPOINT" --pose_bank data/recovery/fallen_v1.pt \
  --num_envs 4096 --chunk_iterations 2000 --iterations 6000

# 检查数据完整性、实时迭代和最近评估
"$SIM_PY" scripts/recovery/check_recovery.py
# 在当前训练/评估段结束后暂停
"$SIM_PY" scripts/recovery/manage_recovery.py stop
```

控制器记录在 `logs/recovery_fallen/continuous/`，新 checkpoint 在 `logs/rsl_rl/wf_tron_1a_fallen/`。旧模型全测试集评估可用 `scripts/recovery/evaluate_fallen_poses.py --checkpoint "$CHECKPOINT" --pose_bank data/recovery/fallen_v1.pt --output logs/recovery_fallen/evaluation.json --headless`（用 `$SIM_PY` 运行）。播放新任务沿用 `play.py`，传入新 task ID、`--pose_bank data/recovery/fallen_v1.pt` 和所需 checkpoint。

## 机器人形态

| 形态 | 末端 | Task ID 前缀 |
|---|---|---|
| SF_TRON1A | sole foot (ankle pitch) | Isaac-Limx-SF-... |
| WF_TRON1A | wheel | Isaac-Limx-WF-... |

## 相关仓库

| 仓库 | 描述 |
|---|---|
| [tron1-robot-description](https://github.com/limxdynamics/tron1-robot-description) | TRON1 机器人模型文件 |
| [tron1-rl-isaacgym](https://github.com/limxdynamics/tron1-rl-isaacgym) | TRON1 Isaac Gym RL 训练 |
| [tron1-rl-deploy-ros](https://github.com/limxdynamics/tron1-rl-deploy-ros) | TRON1 RL 部署（ROS） |
| [tron1-rl-deploy-python](https://github.com/limxdynamics/tron1-rl-deploy-python) | TRON1 RL 部署（Python） |

## 许可证

[Apache 2.0](LICENCE)。
