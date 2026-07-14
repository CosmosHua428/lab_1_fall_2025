"""Study how delayed state observations affect PD control."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from controllers import clip_torque, pd_control
from sim.single_joint import JointModel, SingleJointPlant


@dataclass
class DelayResult:
    """Result from one observation-delay experiment."""

    delay_seconds: float
    times: list[float]
    positions: list[float]
    velocities: list[float]
    torques: list[float]
    target_positions: list[float]


def run_delay_experiment(delay_seconds: float) -> DelayResult:
    """Run a PD step response with delayed position and velocity."""

    model = JointModel(
        inertia=0.05,
        damping=0.05,
        physics_dt=0.001,
    )
    plant = SingleJointPlant(model)

    duration = 4.0
    step_time = 0.25

    control_frequency = 200.0
    control_period = 1.0 / control_frequency

    physics_steps_per_control_step = round(
        control_period / model.physics_dt
    )

    delay_steps = round(
        delay_seconds * control_frequency
    )

    kp = 2.0
    kd = 0.3
    max_torque = 3.0

    # 缓冲区长度为 delay_steps + 1。
    # buffer[0] 始终是控制器需要使用的旧状态。
    torque_buffer: deque[float] = deque(
        [0.0] * (delay_steps + 1),
        maxlen=delay_steps + 1,
    )

    total_steps = int(duration / model.physics_dt)
    current_torque = 0.0

    times: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    torques: list[float] = []
    target_positions: list[float] = []

    for step_index in range(total_steps + 1):
        current_time = step_index * model.physics_dt

        target_position = (
            0.0 if current_time < step_time else 1.0
        )
        target_velocity = 0.0

        if step_index % physics_steps_per_control_step == 0:
            raw_torque = pd_control(
                current_position=plant.state.position,
                current_velocity=plant.state.velocity,
                target_position=target_position,
                target_velocity=target_velocity,
                kp=kp,
                kd=kd,
            )

            calculated_torque = clip_torque(
                torque=raw_torque,
                max_torque=max_torque,
            )

        torque_buffer.append(calculated_torque)
        current_torque = torque_buffer[0]

        # 动力学始终用当前执行力矩更新真实状态。
        plant.step(current_torque)

        times.append(current_time)
        positions.append(plant.state.position)
        velocities.append(plant.state.velocity)
        torques.append(current_torque)
        target_positions.append(target_position)

    return DelayResult(
        delay_seconds=delay_seconds,
        times=times,
        positions=positions,
        velocities=velocities,
        torques=torques,
        target_positions=target_positions,
    )


def calculate_settling_time(
    result: DelayResult,
    step_time: float = 0.25,
    target_position: float = 1.0,
    tolerance_ratio: float = 0.02,
) -> float | None:
    """Return the first time after which error stays within ±2%."""

    tolerance = abs(target_position) * tolerance_ratio

    for index, current_time in enumerate(result.times):
        if current_time < step_time:
            continue

        remaining_positions = result.positions[index:]

        if all(
            abs(position - target_position) <= tolerance
            for position in remaining_positions
        ):
            return current_time - step_time

    return None


def print_summary(result: DelayResult) -> None:
    """Print metrics for one delay setting."""

    maximum_position = max(result.positions)
    overshoot = max(0.0, maximum_position - 1.0)

    maximum_speed = max(
        abs(velocity)
        for velocity in result.velocities
    )

    mean_absolute_error = sum(
        abs(target - position)
        for target, position in zip(
            result.target_positions,
            result.positions,
        )
    ) / len(result.positions)

    saturated_steps = sum(
        abs(torque) >= 3.0 - 1e-9
        for torque in result.torques
    )
    saturation_ratio = saturated_steps / len(result.torques)

    settling_time = calculate_settling_time(result)

    print(
        f"\nCommand delay: "
        f"{result.delay_seconds * 1000:.0f} ms"
    )
    print(
        f"Final position: "
        f"{result.positions[-1]:.4f} rad"
    )
    print(f"Overshoot: {overshoot:.4f} rad")
    print(
        f"Maximum speed: "
        f"{maximum_speed:.4f} rad/s"
    )
    print(
        f"Mean absolute error: "
        f"{mean_absolute_error:.4f} rad"
    )
    print(
        f"Torque saturation ratio: "
        f"{saturation_ratio:.2%}"
    )

    if settling_time is None:
        print("Settling time: not settled")
    else:
        print(
            f"Settling time: "
            f"{settling_time:.4f} s"
        )


def plot_position_results(
    results: list[DelayResult],
    output_directory: Path,
) -> None:
    """Plot position responses for all command delays."""

    plt.figure(figsize=(10, 6))

    reference = results[0]

    plt.plot(
        reference.times,
        reference.target_positions,
        linestyle="--",
        label="Target",
    )

    for result in results:
        delay_ms = result.delay_seconds * 1000

        plt.plot(
            result.times,
            result.positions,
            label=f"{delay_ms:.0f} ms",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Position [rad]")
    plt.title("Effect of Command Delay on PD Control")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory / "position_comparison.png",
        dpi=200,
    )
    plt.show()


def plot_torque_results(
    results: list[DelayResult],
    output_directory: Path,
) -> None:
    """Plot applied torque for all command delays."""

    plt.figure(figsize=(10, 6))

    for result in results:
        delay_ms = result.delay_seconds * 1000

        plt.plot(
            result.times,
            result.torques,
            label=f"{delay_ms:.0f} ms",
        )

    plt.axhline(
        3.0,
        linestyle="--",
        label="Torque limit",
    )
    plt.axhline(-3.0, linestyle="--")

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [N·m]")
    plt.title("Calculated and Applied Torque with Command Delay")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory / "torque_comparison.png",
        dpi=200,
    )
    plt.show()


def main() -> None:
    """Run the command-delay sweep."""

    output_directory = Path(
        "experiments/LAB-1-DELAY-02/figures"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    delay_values = [
        0.00,
        0.01,
        0.03,
        0.05,
        0.10,
    ]

    results = [
        run_delay_experiment(delay_seconds)
        for delay_seconds in delay_values
    ]

    for result in results:
        print_summary(result)

    plot_position_results(
        results=results,
        output_directory=output_directory,
    )

    plot_torque_results(
        results=results,
        output_directory=output_directory,
    )


if __name__ == "__main__":
    main()