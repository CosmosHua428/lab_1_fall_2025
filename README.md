# CS123 Fall 2025 — Lab 1

This repository is my reproduction of Stanford CS123 Lab 1.

## Goals

- Understand the ROS 2 joint control pipeline
- Implement and analyze P and PD control
- Replace unavailable hardware experiments with simulation
- Preserve compatibility with the original Lab 1 interfaces

## Branches

- `master`: synchronized with the official upstream repository
- `dev/lab1-simulation`: personal implementation and simulation work

## Repository Structure

- `lab_1.py`: original course control node
- `lab_1.launch.py`: ROS 2 launch configuration
- `lab_1.yaml`: controller configuration
- `sim/`: hardware-free simulations
- `experiments/`: experiment records and results