"""Motor35-only terms. Preserve the old policies' observation semantics."""
import torch


def read_physical_properties(env):
    """Fixed-width privileged data from actual randomized values, not defaults.

    Nine links: mass 9, inertia 81, COM 27; 8 motors: Kp 8, Kd 8;
    material min/mean/max 9, independent of convex decomposition shape count.
    Startup-only DR lets us capture this once rather than synchronize the CPU
    material/mass buffers on every policy step.
    """
    robot = env.scene["robot"]
    view = robot.root_physx_view
    materials = view.get_material_properties().to(env.device)
    stiffness = torch.zeros_like(robot.data.joint_pos)
    damping = torch.zeros_like(stiffness)
    for actuator in robot.actuators.values():
        stiffness[:, actuator.joint_indices] = actuator.stiffness
        damping[:, actuator.joint_indices] = actuator.damping
    return torch.cat([
        view.get_masses().to(env.device), view.get_inertias().to(env.device).flatten(1),
        view.get_coms().to(env.device)[..., :3].flatten(1), stiffness, damping,
        materials.amin(1), materials.mean(1), materials.amax(1),
    ], dim=1)


def capture_physical_properties(env, env_ids):
    env._motor35_physical_properties = read_physical_properties(env)


def physical_properties(env):
    value = getattr(env, "_motor35_physical_properties", None)
    return read_physical_properties(env) if value is None else value


def forward_velocity_error(env, command_name="base_velocity"):
    robot = env.scene["robot"]
    error = env.command_manager.get_command(command_name)[:, 0] - robot.data.root_lin_vel_b[:, 0]
    return torch.sqrt(1. + error.square()) - 1.


def reverse_motion(env, command_name="base_velocity"):
    target = env.command_manager.get_command(command_name)[:, 0]
    actual = env.scene["robot"].data.root_lin_vel_b[:, 0]
    return torch.relu(-target.sign() * actual) * (target.abs() > .1)
