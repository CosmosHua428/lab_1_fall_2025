"""Minimal single-joint simulation for CS123 Lab 1.

The simulation replaces the physical motor with a simple rigid-joint model:

    J * q_ddot + b * q_dot = torque

It is intentionally simple. Its purpose is to study controller behavior,
not to reproduce the real Pupper motor perfectly.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt

from controllers import clip_torque, pd_control


@dataclass
class JointState:
    """State of a revolute joint."""

    position: float = 0.0  # rad
    velocity: float = 0.0  # rad/s


@dataclass(frozen=True)
class JointModel:
    """Physical parameters of the simulated joint."""

    inertia: float = 0.05  # kg·m²
    damping: float = 0.05  # N·m·s/rad
    physics_dt: float = 0.001  # s


class SingleJointPlant:
    """A minimal rigid-joint dynamics model."""

    def __init__(
        self,
        model: JointModel,
        initial_state: JointState | None = None,
    ) -> None:
        if model.inertia <= 0.0:
            raise ValueError("inertia must be positive")

        if model.damping < 0.0:
            raise ValueError("damping must be non-negative")

        if model.physics_dt <= 0.0:
            raise ValueError("physics_dt must be positive")

        self.model = model
        self.state = initial_state or JointState()

    def step(self, applied_torque: float) -> JointState:
        """Advance the joint dynamics by one physics step."""

        acceleration = (
            applied_torque
            - self.model.damping * self.state.velocity
        ) / self.model.inertia

        # Semi-implicit Euler integration:
        # update velocity first, then update position.
        self.state.velocity += (
            acceleration * self.model.physics_dt
        )

        self.state.position += (
            self.state.velocity * self.model.physics_dt
        )

        return self.state


def simulate() -> None:
    """Run a PD-controlled step-response experiment."""

    model = JointModel(
        inertia=0.05,
        damping=0.05,
        physics_dt=0.001,
    )

    plant = SingleJointPlant(model)

    duration = 3.0
    control_frequency = 200.0
    control_period = 1.0 / control_frequency

    kp = 2.0
    kd = 0.6
    max_torque = 3.0

    physics_steps_per_control_step = round(
        control_period / model.physics_dt
    )

    if physics_steps_per_control_step < 1:
        raise ValueError(
            "Control period must not be shorter than physics timestep"
        )

    number_of_steps = int(duration / model.physics_dt)

    current_torque = 0.0

    times: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    target_positions: list[float] = []
    torques: list[float] = []

    for step_index in range(number_of_steps + 1):
        current_time = step_index * model.physics_dt

        # Start from zero and command a one-radian step at 0.25 s.
        target_position = 0.0 if current_time < 0.25 else 1.0
        target_velocity = 0.0

        # The physics model runs at 1000 Hz, while the controller runs at
        # 200 Hz. Between controller updates, the previous torque is held.
        if step_index % physics_steps_per_control_step == 0:
            raw_torque = pd_control(
                current_position=plant.state.position,
                current_velocity=plant.state.velocity,
                target_position=target_position,
                target_velocity=target_velocity,
                kp=kp,
                kd=kd,
            )

            current_torque = clip_torque(
                torque=raw_torque,
                max_torque=max_torque,
            )

        plant.step(current_torque)

        times.append(current_time)
        positions.append(plant.state.position)
        velocities.append(plant.state.velocity)
        target_positions.append(target_position)
        torques.append(current_torque)

    print_experiment_summary(
        positions=positions,
        velocities=velocities,
        target_positions=target_positions,
        torques=torques,
        max_torque=max_torque,
    )

    plot_results(
        times=times,
        positions=positions,
        velocities=velocities,
        target_positions=target_positions,
        torques=torques,
        max_torque=max_torque,
    )


def print_experiment_summary(
    positions: list[float],
    velocities: list[float],
    target_positions: list[float],
    torques: list[float],
    max_torque: float,
) -> None:
    """Print a few basic controller-performance measurements."""

    final_error = target_positions[-1] - positions[-1]

    maximum_position = max(positions)
    overshoot = max(0.0, maximum_position - 1.0)

    saturated_steps = sum(
        abs(torque) >= max_torque - 1e-9
        for torque in torques
    )

    saturation_ratio = saturated_steps / len(torques)

    maximum_speed = max(
        abs(velocity) for velocity in velocities
    )

    print(f"Final position: {positions[-1]:.4f} rad")
    print(f"Error at simulation end: {final_error:.4f} rad")
    print(f"Overshoot: {overshoot:.4f} rad")
    print(f"Torque saturation ratio: {saturation_ratio:.2%}")
    print(f"Maximum speed: {maximum_speed:.4f} rad/s")


def plot_results(
    times: list[float],
    positions: list[float],
    velocities: list[float],
    target_positions: list[float],
    torques: list[float],
    max_torque: float,
) -> None:
    """Plot position, velocity, and applied torque."""

    figure, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(9, 8),
        sharex=True,
    )

    axes[0].plot(times, target_positions, label="target position")
    axes[0].plot(times, positions, label="actual position")
    axes[0].set_ylabel("Position [rad]")
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(times, velocities)
    axes[1].set_ylabel("Velocity [rad/s]")
    axes[1].grid()

    axes[2].plot(times, torques, label="applied torque")
    axes[2].axhline(max_torque, linestyle="--", label="torque limit")
    axes[2].axhline(-max_torque, linestyle="--")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Torque [N·m]")
    axes[2].legend()
    axes[2].grid()

    figure.suptitle("Single-Joint PD Step Response")
    figure.tight_layout()

    plt.show()


if __name__ == "__main__":
    simulate()