"""Pure control laws used by both ROS 2 and simulation.

This module deliberately contains no ROS 2 dependencies.
"""


def bang_bang_control(
    current_position: float,
    target_position: float,
    max_torque: float,
    deadband: float = 0.0,
) -> float:
    """Return a fixed-magnitude torque according to the position error sign."""
    if max_torque <= 0.0:
        raise ValueError("max_torque must be positive")

    if deadband < 0.0:
        raise ValueError("deadband must be non-negative")

    position_error = target_position - current_position

    if abs(position_error) <= deadband:
        return 0.0

    return max_torque if position_error > 0.0 else -max_torque


def p_control(
    current_position: float,
    target_position: float,
    kp: float,
) -> float:
    """Calculate raw torque using proportional position feedback."""
    position_error = target_position - current_position
    return kp * position_error


def pd_control(
    current_position: float,
    current_velocity: float,
    target_position: float,
    target_velocity: float,
    kp: float,
    kd: float,
    feedforward_torque: float = 0.0,
) -> float:
    """Calculate raw torque using PD feedback and optional feedforward."""
    position_error = target_position - current_position
    velocity_error = target_velocity - current_velocity

    return (
        kp * position_error
        + kd * velocity_error
        + feedforward_torque
    )


def clip_torque(torque: float, max_torque: float) -> float:
    """Limit torque to the actuator's symmetric torque range."""
    if max_torque <= 0.0:
        raise ValueError("max_torque must be positive")

    return max(-max_torque, min(torque, max_torque))