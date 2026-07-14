# Simulation

This directory contains hardware-free substitutes for the Lab 1 motor experiments.

Planned experiments:

- Single-joint dynamics
- Bang-bang control
- P control
- PD control
- Torque saturation
- Observation and command delay
- Sinusoidal trajectory tracking

Run the sinusoidal frequency sweep from the repository root:

```powershell
python -m sim.sinusoidal_tracking
```

Add `--show` to display the plots after saving them. Without this flag the
script is suitable for headless and automated runs.
