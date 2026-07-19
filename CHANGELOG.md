# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-07-19

First public open-source release.

NeuCoDe was developed as a Master's thesis toolkit at Technische Hochschule
Nuernberg Georg Simon Ohm between September 2025 and July 2026 (462 commits,
single author). The public repository begins from a clean, squashed history at
this release; the full development history is retained privately. This changelog
summarizes the capabilities present at first release rather than the incremental
thesis history.

### Added

- **Simulation core** — a Cython-wrapped C library for fast FOPDT/FOIPDT plant
  simulation with PID control, disturbances, and Coulomb friction.
- **Controllers** — `PIDController`, `ANNController`, `SNNController`,
  `KerasController`, and `AkidaController` behind a common interface.
- **Architectures** — MLP replacements (`DefaultMLPArchitecture`,
  `NoContextMLPArchitecture`); spiking networks `HybridControlSNN` and
  `PopulationControlSNN` (both Akida-exportable) plus an experimental all-spike
  `SpikeControlSNN`; with rate, delta, and hybrid input encoders.
- **PID tuning** — Differential-Evolution global search for optimal gains, a
  Ziegler-Nichols reaction-curve tuner (`ZieglerNicholsReactionCurveTuner`), a
  supervised MLP gain predictor (`SupervisedTuner`), and an optional
  reinforcement-learning tuner (`RLTuner`, extra `rl`).
- **Dataset generation** — `TuningDatasetGenerator` (plant to optimal PID gains
  via differential evolution) and `ReplacementDatasetGenerator` (closed-loop
  expert trajectories for imitation learning).
- **Replacement training** — imitation learning (behavioral cloning) that trains
  ANN and SNN controllers to replace a PID expert (`ANNReplacementTrainer`,
  `SNNReplacementTrainer`); an optional `DAggerTrainer` adds an iterative
  dataset-aggregation loop to reduce distribution shift; quantization-aware
  calibration collectors feed the export step.
- **Exporters** — quantization-aware fixed-point (`int8`) C-header export for
  Cortex-M MCUs (`ANNExporter`, `SNNExporter`), BrainChip Akida `.fbz` export
  (`AKD1000Exporter`, 4-bit, optional `akida`), and experimental NIR interchange
  export (`NIRExporter`, optional `nir`).
- **Harness & benchmarks** — simulation and hardware harnesses with
  step/ramp/noisy/disturbed/robust benchmark scenarios, standard control metrics
  (ITAE, overshoot, settling time), and plot/table reporting.
- **Firmware** — STM32 Nucleo G431RB firmware with architecture-agnostic SNN/ANN
  input encoding via channel descriptors, runtime controller mode switching
  (PID/ANN/SNN/open-loop/sysid), and a serial telemetry protocol; FOC motor
  control via SimpleFOC + STSPIN830.
- **Packaging & docs** — installable package with optional dependency extras
  (`rl`, `nir`, `akida`, `docs`, `dev`) and a MkDocs documentation site.

### Notes

- The optional `[akida]` extra pulls proprietary BrainChip software under its own
  license (not Apache-2.0) and requires BrainChip hardware. See `NOTICE`.

[Unreleased]: https://github.com/reipan/neucode/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/reipan/neucode/releases/tag/v0.1.0
