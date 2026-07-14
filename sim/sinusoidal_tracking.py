"""Sinusoidal trajectory tracking experiments for CS123 Lab 1.

Run from the repository root with::

    python -m sim.sinusoidal_tracking
"""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from controllers import clip_torque, pd_control
from sim.single_joint import JointModel, SingleJointPlant


@dataclass
class SineResult:
    """Time-series result from one sine tracking experiment."""

    frequency: float
    times: list[float]
    target_positions: list[float]
    target_velocities: list[float]
    positions: list[float]
    velocities: list[float]
    torques: list[float]


def run_sine_experiment(frequency: float) -> SineResult:
    """Track one sinusoidal trajectory using PD control."""

    if frequency <= 0.0:
        raise ValueError("frequency must be positive")

    model = JointModel(
        inertia=0.05,
        damping=0.05,
        physics_dt=0.001,
    )
    plant = SingleJointPlant(model)

    duration = 8.0

    control_frequency = 200.0
    control_period = 1.0 / control_frequency

    physics_steps_per_control_step = round(
        control_period / model.physics_dt
    )

    kp = 2.0
    kd = 0.3
    max_torque = 3.0

    amplitude = 0.5
    center = 0.0
    angular_frequency = 2.0 * math.pi * frequency

    total_steps = int(duration / model.physics_dt)
    current_torque = 0.0

    times: list[float] = []
    target_positions: list[float] = []
    target_velocities: list[float] = []
    positions: list[float] = []
    velocities: list[float] = []
    torques: list[float] = []

    for step_index in range(total_steps + 1):
        current_time = step_index * model.physics_dt

        target_position = (
            center
            + amplitude
            * math.sin(angular_frequency * current_time)
        )

        target_velocity = (
            amplitude
            * angular_frequency
            * math.cos(angular_frequency * current_time)
        )

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
        target_positions.append(target_position)
        target_velocities.append(target_velocity)
        positions.append(plant.state.position)
        velocities.append(plant.state.velocity)
        torques.append(current_torque)

    return SineResult(
        frequency=frequency,
        times=times,
        target_positions=target_positions,
        target_velocities=target_velocities,
        positions=positions,
        velocities=velocities,
        torques=torques,
    )


def calculate_metrics(result: SineResult) -> dict[str, float]:
    """Calculate steady tracking metrics after the first cycle."""

    # 跳过第一个周期，避免初始瞬态过度影响指标。
    analysis_start_time = 1.0 / result.frequency

    start_index = next(
        index
        for index, current_time in enumerate(result.times)
        if current_time >= analysis_start_time
    )

    times = np.asarray(result.times[start_index:])
    targets = np.asarray(
        result.target_positions[start_index:]
    )
    positions = np.asarray(result.positions[start_index:])
    torques = np.asarray(result.torques[start_index:])

    errors = targets - positions

    rmse = float(np.sqrt(np.mean(errors**2)))
    maximum_error = float(np.max(np.abs(errors)))

    saturation_ratio = float(
        np.mean(np.abs(torques) >= 3.0 - 1e-9)
    )

    # 用正弦、余弦基函数拟合实际位置：
    # q ≈ a sin(wt) + b cos(wt) + c
    # 由此估计振幅与相位。
    omega = 2.0 * math.pi * result.frequency

    design_matrix = np.column_stack(
        [
            np.sin(omega * times),
            np.cos(omega * times),
            np.ones_like(times),
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        design_matrix,
        positions,
        rcond=None,
    )

    sine_coefficient = coefficients[0]
    cosine_coefficient = coefficients[1]

    fitted_amplitude = float(
        math.hypot(
            sine_coefficient,
            cosine_coefficient,
        )
    )

    commanded_amplitude = 0.5
    amplitude_ratio = (
        fitted_amplitude / commanded_amplitude
    )

    fitted_phase = math.atan2(
        cosine_coefficient,
        sine_coefficient,
    )

    # 正值表示实际响应落后于目标。
    phase_lag_degrees = -math.degrees(fitted_phase)

    phase_lag_degrees = (
        (phase_lag_degrees + 180.0) % 360.0
    ) - 180.0

    return {
        "rmse_rad": rmse,
        "maximum_error_rad": maximum_error,
        "amplitude_ratio": amplitude_ratio,
        "fitted_amplitude_rad": fitted_amplitude,
        "phase_lag_degrees": phase_lag_degrees,
        "maximum_torque_nm": float(
            np.max(np.abs(torques))
        ),
        "saturation_ratio": saturation_ratio,
    }


def print_summary(
    result: SineResult,
    metrics: dict[str, float],
) -> None:
    """Print the tracking metrics for one frequency."""

    print(f"\nFrequency: {result.frequency:.2f} Hz")
    print(f"RMSE: {metrics['rmse_rad']:.4f} rad")
    print(
        f"Maximum error: "
        f"{metrics['maximum_error_rad']:.4f} rad"
    )
    print(
        f"Amplitude ratio: "
        f"{metrics['amplitude_ratio']:.4f}"
    )
    print(
        f"Phase lag: "
        f"{metrics['phase_lag_degrees']:.2f} deg"
    )
    print(
        f"Maximum torque: "
        f"{metrics['maximum_torque_nm']:.4f} N·m"
    )
    print(
        f"Torque saturation ratio: "
        f"{metrics['saturation_ratio']:.2%}"
    )


def save_raw_data(
    result: SineResult,
    output_directory: Path,
) -> None:
    """Save one frequency run to CSV."""

    raw_directory = output_directory / "raw"
    raw_directory.mkdir(parents=True, exist_ok=True)

    filename = (
        f"frequency_{result.frequency:.2f}hz.csv"
    )

    with (raw_directory / filename).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "time_s",
                "target_position_rad",
                "target_velocity_rad_s",
                "position_rad",
                "velocity_rad_s",
                "torque_nm",
            ]
        )

        writer.writerows(
            zip(
                result.times,
                result.target_positions,
                result.target_velocities,
                result.positions,
                result.velocities,
                result.torques,
            )
        )


def save_summary(
    results_and_metrics: list[
        tuple[SineResult, dict[str, float]]
    ],
    output_directory: Path,
) -> None:
    """Save all frequency metrics to one CSV."""

    with (output_directory / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "frequency_hz",
                "rmse_rad",
                "maximum_error_rad",
                "fitted_amplitude_rad",
                "amplitude_ratio",
                "phase_lag_degrees",
                "maximum_torque_nm",
                "saturation_ratio",
            ]
        )

        for result, metrics in results_and_metrics:
            writer.writerow(
                [
                    result.frequency,
                    metrics["rmse_rad"],
                    metrics["maximum_error_rad"],
                    metrics["fitted_amplitude_rad"],
                    metrics["amplitude_ratio"],
                    metrics["phase_lag_degrees"],
                    metrics["maximum_torque_nm"],
                    metrics["saturation_ratio"],
                ]
            )


def plot_tracking_results(
    results: list[SineResult],
    output_directory: Path,
    show_plots: bool = False,
) -> None:
    """Plot target and actual position for every frequency."""

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13, 9),
        sharex=True,
        sharey=True,
    )

    for axis, result in zip(axes.flat, results):
        axis.plot(
            result.times,
            result.target_positions,
            linestyle="--",
            label="Target",
        )

        axis.plot(
            result.times,
            result.positions,
            label="Actual",
        )

        axis.set_title(
            f"{result.frequency:.2f} Hz"
        )
        axis.set_xlabel("Time [s]")
        axis.set_ylabel("Position [rad]")
        axis.grid()
        axis.legend()

    figure.suptitle(
        "Sinusoidal Position Tracking at Different Frequencies"
    )
    figure.tight_layout()

    figure.savefig(
        output_directory / "tracking_comparison.png",
        dpi=200,
    )
    if show_plots:
        plt.show()
    else:
        plt.close(figure)


def plot_error_results(
    results: list[SineResult],
    output_directory: Path,
    show_plots: bool = False,
) -> None:
    """Plot tracking error for every target frequency."""

    figure = plt.figure(figsize=(11, 6))

    for result in results:
        errors = [
            target - actual
            for target, actual in zip(
                result.target_positions,
                result.positions,
            )
        ]

        plt.plot(
            result.times,
            errors,
            label=f"{result.frequency:.2f} Hz",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Position error [rad]")
    plt.title("Tracking Error at Different Frequencies")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory / "error_comparison.png",
        dpi=200,
    )
    if show_plots:
        plt.show()
    else:
        plt.close(figure)


def plot_torque_results(
    results: list[SineResult],
    output_directory: Path,
    show_plots: bool = False,
) -> None:
    """Plot applied torque for every target frequency."""

    figure = plt.figure(figsize=(11, 6))

    for result in results:
        plt.plot(
            result.times,
            result.torques,
            label=f"{result.frequency:.2f} Hz",
        )

    plt.axhline(
        3.0,
        linestyle="--",
        label="Torque limit",
    )
    plt.axhline(-3.0, linestyle="--")

    plt.xlabel("Time [s]")
    plt.ylabel("Torque [N·m]")
    plt.title("Applied Torque at Different Frequencies")
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_directory / "torque_comparison.png",
        dpi=200,
    )
    if show_plots:
        plt.show()
    else:
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the experiment."""

    parser = argparse.ArgumentParser(
        description="Run the sinusoidal tracking frequency sweep."
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display plots after saving them",
    )
    return parser.parse_args()


def main(show_plots: bool = False) -> None:
    """Run the sinusoidal frequency sweep."""

    output_directory = Path(
        "experiments/LAB-1-PERIODIC-03"
    )
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    frequencies = [
        0.25,
        0.50,
        1.00,
        2.00,
    ]

    results = [
        run_sine_experiment(frequency)
        for frequency in frequencies
    ]

    results_and_metrics = []

    for result in results:
        metrics = calculate_metrics(result)
        results_and_metrics.append(
            (result, metrics)
        )

        print_summary(result, metrics)
        save_raw_data(result, output_directory)

    save_summary(
        results_and_metrics,
        output_directory,
    )

    plot_tracking_results(
        results,
        output_directory,
        show_plots=show_plots,
    )

    plot_error_results(
        results,
        output_directory,
        show_plots=show_plots,
    )

    plot_torque_results(
        results,
        output_directory,
        show_plots=show_plots,
    )


if __name__ == "__main__":
    arguments = parse_args()
    main(show_plots=arguments.show)
