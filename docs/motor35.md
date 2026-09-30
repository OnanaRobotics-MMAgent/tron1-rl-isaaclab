# Motor35 换型与训练前置检查

2026-09-24：已建立独立的 Motor35 模型准备、任务配置、域随机化和预检入口。
完成了短时物理检查及 locomotion → recovery 的各一次 PPO 更新；**没有进行完整训练，也没有新模型的成功率结论**。
`checkpoints/motor35/*/preflight_zero_policy.pt` 里的权重只是链路测试产物，不能当作站稳的初始化策略。

## 已核对的模型与电机参数

源目录为 `exts/bipedal_locomotion/bipedal_locomotion_motor35/assets/source/Motor35_URDF/`。文件为 `轮腿总装/urdf/轮腿总装（角度限位调整版本）.urdf` 和 `转动惯量.txt`。
该文件夹的 CSV 是旧导出：质量和关节限位与当前 URDF 不一致，不能用于覆盖当前模型。

| 项目 | 新模型使用值 | 依据/含义 |
|---|---:|---|
| 电机 | DM-J3507-2EC，24 V | 当前 URDF 注释指定的型号/电压 |
| 额定输出力矩 | 0.8 N·m | 全部 8 个电机共用的软惩罚阈值 |
| 峰值输出力矩 | 3 N·m | 执行器及求解器的电机驱动力矩上限 |
| 空载输出转速 | 460 rpm ≈ 48.171 rad/s | 不代表此速度下还能输出峰值力矩 |
| 额定输出转速 | 150 rpm ≈ 15.708 rad/s | 轮速动作的缩放基准 |
| 额定功率 | 12 W | 厂家额定工况；本次没有把它当瞬时硬上限 |
| 内部减速比 | 7:1 | 上述输出轴参数已经包含减速，禁止再乘 7 |
| 总质量 | 2.621 kg | 9 个刚体的 CAD 质量之和 |
| 轮半径/宽度 | 0.040 / 0.032 m | STL 测量，原有任务的 0.0375 m 不适用 |
| 前进方向 | body-x | 轮轴沿 body ±y；旧任务是 body-y |

额定/峰值力矩见[达妙官方产品页](https://www.dmbot.cn/index.php?c=show&id=118)，转速、电压、功率见厂家的[2025 产品选型手册第 6 页](https://www.worldrobotconference.com/profile/robot/download/2025/06/30/%E8%BE%BE%E5%A6%99%E7%A7%91%E6%8A%80DAMIAO%20-%202025%E5%B9%B4%E4%BA%A7%E5%93%81%E9%80%89%E5%9E%8B%E6%89%8B%E5%86%8C_20250630192546A441.pdf)。

CAD 的 mm 转 m、kg·mm² 转 kg·m² 后，URDF 的质量、质心和 **COM 处 Lxx 等惯性张量**逐项吻合；惯性矩阵正定且满足主惯量三角关系。
CAD 的 Ixx 等是关于输出坐标原点的张量，不能再次当作质心惯量使用。
这个文件描述各刚体的整体惯量，**不是电机转子的转动惯量**，不能拿来直接填关节 armature。

## 本次实现

- `tools/prepare_motor35.py`：核对原始 URDF/CAD，输出独立的 `exts/bipedal_locomotion/bipedal_locomotion_motor35/assets/urdf/robot.urdf`、便携的英文 mesh 路径和 `audit.json`。
  只映射语义名称、整理 mesh 路径、用测得尺寸的圆柱替换轮碰撞形状；保持质量、质心、惯量、关节轴和角度限位。
  不调用旧的 `prepare_wheel_leg_urdf.py`，因为它会改成旧机器人的质量、惯量、限位和碰撞外形。
- 非轮部件使用凸分解。检查发现单凸包会填平机身的凹口，导致机身与左右大腿的包络重叠。
  凸分解后的初始姿态已通过无重力悬空接触检查。**这不等于验证了所有关节姿态的碰撞精度**，后续仍应观察动作范围及倒地接触。
- 新的两个训练任务和两个 play 任务共用资产、控制参数、动作缩放和观测定义，**两类任务都开启自碰撞**。
  Motor43 的资产和动作参数保留；目录整理后，两个模型各自在自己的包注册，旧 `Isaac-Limx-*` ID 已移除。
- FK 求得站立参考：hip 左/右为 −0.08/+0.08 rad，knee 为 −0.65901/+0.65901 rad，abad 为 0。
  根部目标高度 0.187714 m，机体 x 方向总质心在轮轴上方。该姿态只是平衡策略的参考，并不保证无反馈时能站稳。
- 前进速度命令先设为 ±0.5 m/s，转向 ±1.5 rad/s；不继续使用旧的 ±2 m/s。
  新轮即使空载也只有约 1.927 m/s，且带载上限会更低。
- 使用显式 `DCMotor`，在 3 N·m 硬限幅内加入 Isaac Lab 的线性四象限力矩—速度近似，避免把峰值力矩和空载速度当成同时可用。
  这是仿真起点，**不是已拟合的实测电机曲线，也没有热保护模型**。
  初始腿 Kp/Kd=8/0.15，轮 Kp/Kd=0/0.02；这些增益需要小规模训练及实机辨识后调整。
  物理步长 2.5 ms，控制周期保持 20 ms。
- 两任务统一添加 `−0.25 × Σ max(|τ|−0.8, 0)`，作用于全部 8 个电机，并保留各自原有的力矩平方惩罚。
  `τ` 是显式执行器限幅后实际提交给仿真的驱动力矩；奖励最终还乘 0.02 s。
  该软惩罚允许起身瞬时超过额定值，不保证超载持续时间或电机温度。
- 修复公共 `joint_powers_l1` 忽略 `joint_ids` 的问题：专门的非轮功率项现在只统计腿部，轮功率不再被该项重复计入。
  这是先前 Motor35 前置工作中已有的修复，此次目录整理未再次修改该奖励函数；已有权重推理不受奖励函数变化影响。
- 新 critic 使用真实随机化后的质量、惯量、COM、执行器增益和固定维数的材质摘要。
  数据在启动随机化完成后缓存，避免每步 CPU/GPU 同步，也避免读到未随机化的 default mass/gain。
- 新权重保存模型/电机/动作接口指纹。恢复训练和通用 play 会拒绝旧 WF 权重或不同 Motor35 版本，防止“维数相同但物理含义不同”的误加载。

## 域随机化

这些是保守的初始训练区间，**不是测量所得的制造公差**。训练任务默认启用，play 使用标称参数并关闭观测噪声。

| 项目 | 范围/方式 |
|---|---|
| 每个刚体质量 | 标称的 0.9–1.1 倍，只随机化一次 |
| 每个刚体惯量 | 随同一刚体的质量同比缩放，不叠加第二次质量随机化 |
| 机身质心 | x/y/z 各 ±3 mm |
| 腿部/轮部 Kp、Kd | 标称的 0.9–1.1 倍；轮 Kp 始终为 0 |
| 非轮静/动摩擦 | 0.6–1.0 / 0.5–0.9 |
| 轮静/动摩擦 | 0.7–1.2 / 0.6–1.1 |
| 恢复系数 | 0–0.05；强制动摩擦不大于静摩擦 |
| 观测高斯噪声标准差 | 角速度 0.05 rad/s，重力投影 0.025，关节位置 0.01 rad，关节速度 0.01 rad/s（加噪后再按观测项缩放） |

物理参数在每个并行环境启动时采样；本轮不在每次倒地时改变模型参数，避免随机化累积及恢复课程难度突然变化。
命令与 last_action 不加噪声。未沿用旧配置的 −1 至 +2 kg 增重、±25 mm COM 和 500 N 推力。
训练初期不施加额外推力；后续稳态策略评估后再加入有间隔、强度受控的扰动。

## 观测与迁移

两种新策略接口相同：

| 项目 | 维数 |
|---|---:|
| policy：角速度 + 重力投影 + 六腿相对位置 + 八关节速度 + 上次动作 | 28 |
| 历史 | 10 × 28 = 280 |
| 历史编码器输出 | 3 |
| 命令 | 3 |
| Actor 输入 | 3 + 28 + 3 = 34 |
| Critic | 220 |
| 动作 | 8（6 位置 + 2 轮速） |

新 locomotion → 新 recovery 可以迁移权重，且已通过加载与 PPO 更新验证。
旧 WF 模型的重心、惯量、轴向、角限位、默认姿态、动作缩放及执行器均不同，必须从新模型重新训练，不能依据 28 维观测就复用旧权重。

## 可重复的前置检查

在仓库根目录执行；训练资产和 `audit.json` 保存在 Motor35 包的 `assets/urdf/`，USD 在 `assets/usd/Motor35/`。预检输出仍放在被 Git 忽略的 `outputs/motor35/`。

```bash
/home/myoukin/isaacsim/python.sh tools/prepare_motor35.py

/home/myoukin/isaacsim/python.sh tools/motor35/check_motor35.py \
  --task Isaac-Motor35-Locomotion-Play-v0 --collision_probe \
  --num_envs 2 --steps 50 --headless --output outputs/motor35/collision_probe.json

/home/myoukin/isaacsim/python.sh tools/motor35/check_motor35.py \
  --task Isaac-Motor35-Locomotion-v0 --num_envs 16 --steps 100 --headless \
  --output outputs/motor35/locomotion_check.json

/home/myoukin/isaacsim/python.sh tools/motor35/check_motor35.py \
  --task Isaac-Motor35-Recovery-Progressive-v0 --num_envs 16 --steps 100 --headless \
  --output outputs/motor35/recovery_check.json

/home/myoukin/isaacsim/python.sh -m unittest discover -s tools/tests -v
```

本次结果：标称自碰撞探针接触力/关节漂移为零；两个任务的 CAD 质量与惯量导入检查通过。
16 个 DR 环境的总质量为 2.5405–2.6835 kg；locomotion 短时测试输出峰值 2.3004 N·m，recovery 为 1.9312 N·m。
Recovery 检查了第 0、18、35 阶段的初始地面间隙及有限数值输出。
这是小幅随机动作的工程测试，**不计算策略恢复成功率**；无训练策略时摔倒和重置是预期现象。
回归测试中只有依赖仓库外旧 CAD 原件的精确再导出测试可跳过，仓库内的旧碰撞形状检查仍完整执行。

## 后续训练顺序和命令

先做少量环境的 pilot，观察显式 PD 的饱和比例、关节抖动、接触情况、零命令站立和训练速度，然后再扩大环境数。
下面的完整训练命令作为后续入口；本次没有启动它们。环境数要按凸分解模型的实际显存占用调整。

1. 新模型从零训练 locomotion（已含 20% 零速度站立命令和上述 DR）。

```bash
/home/myoukin/isaacsim/python.sh scripts/rsl_rl/train.py \
  --task Isaac-Motor35-Locomotion-v0 --num_envs 512 --headless \
  --max_iterations 20000 --save_interval 100 --run_name motor35_dr
```

2. 验证这份新权重能稳定站立、前后运动和转向，再将**实际通过验证的文件路径**填入 `MOTOR35_LOCO_CKPT`，用相同模型初始化 recovery。
   课程从 0–5° 开始，逐步推进到 170–180°；保持原有按方向统计、连续三窗口达标才升级和 20% 简单姿态回放机制。

```bash
MOTOR35_LOCO_CKPT="$PWD/checkpoints/motor35/locomotion/<已验证的model文件>.pt"
/home/myoukin/isaacsim/python.sh scripts/rsl_rl/train.py \
  --task Isaac-Motor35-Recovery-Progressive-v0 --num_envs 512 --headless \
  --resume True --checkpoint_path "$MOTOR35_LOCO_CKPT" \
  --reset_optimizer --getup_stage 0 \
  --max_iterations 30000 --save_interval 100 --run_name motor35_recovery_dr
```

3. 使用新 play ID 做标称回放；用训练配置及固定种子评估 DR 下的表现。通用 `scripts/rsl_rl/play.py` 接受下面的 ID 和 `--checkpoint_path`：

   - `Isaac-Motor35-Locomotion-Play-v0`
   - `Isaac-Motor35-Recovery-Inverted-Play-v0`（170–180°）

## 完整训练/实机迁移前仍要确认的内容

- 当前按文件声明的 **24 V、八关节同型号、无外部减速**适配。若实际供电、电机分组或传动改变，先改统一参数、重建资产并重跑预检。
- 厂家给出额定/峰值值，但这里还没有验证峰值可持续时间、温升/降额、母线电压下降、实际四象限力矩曲线、摩擦/死区、编码器偏差、通信延迟和转子反射惯量。
  目前没有添加未经测量的转子 armature 或声称已实现温度保护。实机数据到位后应更新执行器模型，并重新确认策略表现。
- 短时预检不能替代 pilot 的性能评估；随机关节倒地姿态覆盖也不等同于当前“标称腿姿 + 根部倾角”课程。
  需要更广倒地分布时，为 Motor35 重新生成训练/留出姿态库，旧 WF 的 fallen bank 不能复用。
- 串联入口现已支持 `scripts/rsl_rl/play_recovery_locomotion.py --robot motor35`，新模型的 FK 轮轴、横向间距、body-x 命令、动作缩放和轮速上限在 `bipedal_locomotion_motor35/play/profile.py` 单独配置。切换阈值及状态机保持原设计；尚无完整训练的新策略来验证起身后接管效果。
- `play_recovery_torques.py` 仍是 Motor43 专用入口。显式执行器的 `applied_torque` 是提交的驱动力矩，不能继续标为“隐式 PD 估算”；求解器轴向受力仍不能当作独立电机电磁转矩。新权重不要传给 Motor43 专用力矩脚本。
