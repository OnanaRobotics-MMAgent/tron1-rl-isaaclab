# 目录迁移检查

本次迁移把原有自定义轮腿任务拆成两个机器人包，并保留已经验证过的任务定义。

- Motor43：`exts/bipedal_locomotion/bipedal_locomotion_motor43/`
- Motor35：`exts/bipedal_locomotion/bipedal_locomotion_motor35/`
- 共享 wrapper、FK 和双策略切换：`exts/bipedal_locomotion/bipedal_locomotion_common/`

每个机器人包分别包含 `assets/`、`locomotion/`、`recovery/` 和 `play/`。Motor43 的 `assets/validated/` 是已有串联策略所对应的模型快照；它替代了旧脚本运行时从 Git 提取历史资产的做法。

迁移前后的 23 个已有环境/runner 配置（Motor43 共 19 个、Motor35 共 4 个）使用 Isaac Sim 启动并序列化比较。逐项排除模块名、任务 ID、资产路径和文件指纹后，环境参数、奖励、观测、动作、课程、事件、终止条件和 PPO 参数没有差异。本次另外增加 Motor35 串联 play 场景及坐标配置，复用原来的切换阈值、状态机和动作平滑规则。

已执行检查：

- 61 个单元测试中，60 个通过，1 个依赖仓库外 CAD 原件的测试跳过。
- editable install 成功，安装包发现 Motor43、Motor35 和 common 的全部 Python 子包及资产。
- Motor43 双策略串联 play：16 个环境、15 秒，全部进入 locomotion；FK 与仿真轮心最大误差约 2.52 μm。
- Motor43 当前资产与已验证快照各 17 个 URDF、mesh、USD 文件，分别与迁移前版本逐字节一致。20 个 MDP 源文件去掉导入路径/文件位置变化后，AST 一致。
- 迁移后 Motor35 locomotion 以 16 环境运行 100 步，recovery 在第 0、18、35 阶段各运行 100 步；质量/惯量、动作维度、自碰撞和 3 N·m 力矩限幅检查通过。
- Motor35 串联场景用零输出测试网络运行 4 环境、1 秒，验证入口、模型接口、坐标和 FK；这不是已训练策略，也没有证明成功起身和接管行走。

检查结果已在本次迁移过程中验证；运行日志和临时摘要已清理。Motor35 完整 locomotion/recovery 训练尚未完成。

`WF_TRON1A` 仍保留在 USD 内部命名及部分配置类名中，避免额外改写已验证资产与实现。公开使用的 Motor43 URDF 文件已经是 `assets/urdf/motor43.urdf`；其 XML 和 mesh 内容未因改名而改变。原始任务 ID `Isaac-Limx-*` 不再注册，也没有兼容别名。
