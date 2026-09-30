"""Finite leg-joint limits; wheel rotation is deliberately excluded."""

import math

import torch


LEG_JOINT_LIMITS = {
    "abad_L_Joint": (math.radians(-22), math.radians(80)),
    "abad_R_Joint": (math.radians(-80), math.radians(22)),
    "hip_L_Joint": (math.radians(-58), 0.0),
    "hip_R_Joint": (0.0, math.radians(58)),
    "knee_L_Joint": (math.radians(-50), math.radians(78)),
    "knee_R_Joint": (math.radians(-78), math.radians(50)),
}


def validated_leg_limits(names, requested, asset_limits):
    """Intersect requested limits with USD limits, never enlarge the native range."""
    if set(names) != set(LEG_JOINT_LIMITS) or set(requested) != set(LEG_JOINT_LIMITS):
        raise ValueError("Specify exactly the six abad/hip/knee joints; wheels must not be limited here.")
    limits = torch.tensor([requested[name] for name in names], device=asset_limits.device,
                          dtype=asset_limits.dtype)
    if not bool(torch.isfinite(limits).all()) or bool((limits.abs() >= math.pi).any()):
        raise ValueError("Leg joint limits must be finite and strictly inside (-pi, pi); full turns are forbidden.")
    if bool(((limits[:, 0] > 0) | (limits[:, 1] < 0) | (limits[:, 0] >= limits[:, 1])).any()):
        raise ValueError("Leg joint ranges must be ordered and include the neutral zero pose.")
    result = asset_limits.clone()
    result[..., 0] = torch.maximum(result[..., 0], limits[:, 0])
    result[..., 1] = torch.minimum(result[..., 1], limits[:, 1])
    if not bool(torch.isfinite(result).all()) or bool(
        ((result[..., 0] > 0) | (result[..., 1] < 0) | (result[..., 0] >= result[..., 1])).any()
    ):
        raise ValueError("The USD limits do not admit the neutral reset inside the requested finite leg range.")
    return result


def outside_joint_limits(positions, limits, tolerance):
    """Use raw joint angles, never wrap modulo 2*pi (which could hide a full turn)."""
    return ((positions < limits[..., 0] - tolerance) | (positions > limits[..., 1] + tolerance)
            | ~torch.isfinite(positions)).any(dim=-1)
