# NeuCoDe — Toolkit for Neuromorphic Control Design & Deployment on Resource-Constrained Devices

[![CI](https://github.com/reipan/neucode/actions/workflows/ci.yml/badge.svg)](https://github.com/reipan/neucode/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

NeuCoDe is a modular, end-to-end toolkit for the design, evaluation, and firmware deployment of neural network-based feedback controllers on resource-constrained embedded systems.
It provides a complete pipeline from plant characterisation to on-device control: PID tuning dataset generation, ANN/SNN replacement training, fixed-point export, and closed-loop validation — in both simulation and on real hardware.
The pipeline is decomposed into stages with explicit artifact contracts, allowing entry at any stage rather than requiring an end-to-end run.

> **Status: work in progress (alpha).** NeuCoDe began as a Master's thesis
> project and is currently being prepared for its first open-source release —
> some parts (documentation, packaging, and examples) are still being cleaned up
> and generalized. It is under active development and released pre-1.0: public
> APIs, artifact formats, and firmware interfaces may change without notice
> before a `v1.0` release. Bug reports and feedback are very welcome — see
> [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Features

- **Fast simulation core** — FOPDT / FOIPDT plant models with dead time,
  disturbances, and Coulomb friction, implemented in C and wrapped with Cython.
- **Unified controllers** — PID, ANN (MLP), and spiking (SNN) controllers, plus
  Keras and BrainChip Akida controllers, behind a single interface; drop any of
  them into the same harness without changing surrounding code.
- **Spiking architectures** — hybrid-encoder and population-coded SNNs (both
  Akida-exportable) and an experimental all-spike variant, with rate, delta, and
  hybrid input encoders.
- **PID tuning** — Differential-Evolution global search for optimal gains, a
  Ziegler-Nichols reaction-curve tuner, a supervised MLP gain predictor, and an
  optional reinforcement-learning tuner.
- **Imitation-learning replacement** — train ANN/SNN controllers to replace a PID
  expert by behavioral cloning on generated closed-loop datasets; an optional
  DAgger (dataset-aggregation) trainer reduces distribution shift.
- **Multi-target export** — quantization-aware `int8` C headers for Cortex-M MCUs
  (ANN/SNN), BrainChip Akida `.fbz` (AKD1000, 4-bit), and experimental NIR
  interchange for other neuromorphic backends (Lava, Sinabs, Rockpool, ...).
- **Benchmarking harness** — simulation and hardware harnesses with
  step / ramp / noisy / disturbed / robust scenarios, standard control metrics
  (ITAE, overshoot, settling time), and plot/table reporting.
- **Embedded firmware** — STM32 Nucleo G431RB firmware with architecture-agnostic
  SNN/ANN input encoding via channel descriptors, runtime mode switching
  (PID / ANN / SNN / open-loop / sysid), and a serial telemetry protocol; FOC
  motor control via SimpleFOC + STSPIN830.
- **Staged, resumable pipeline** — every stage has explicit artifact contracts,
  so you can enter at any point instead of running end-to-end.

---

## Prerequisites

### System packages

```sh
sudo apt update && sudo apt install build-essential bear python3 python3-venv python3-dev python3-tk python3-pip
```

### Install

```sh
python3 -m venv venv
source venv/bin/activate

# Core toolkit + Cython simulation core (numpy, scipy, torch, snntorch, ...)
pip install -e .
```

Optional extras enable additional capabilities:

```sh
pip install -e ".[rl]"      # reinforcement-learning PID tuner
pip install -e ".[nir]"     # NIR interchange export
pip install -e ".[docs]"    # documentation site (MkDocs)
pip install -e ".[dev]"     # test + development tooling (includes rl, nir)
```

> **Optional — Akida/BrainChip neuromorphic export** (`AKD1000Exporter`):
> ```sh
> pip install -e ".[akida]"
> ```
> This pulls **proprietary** BrainChip software under its own license (not
> Apache-2.0) and requires BrainChip hardware. The core toolkit does not import
> it unless the Akida export path is used. See [NOTICE](NOTICE). For aarch64
> production `.fbz` export, use `Dockerfile.akida-export` instead.

### Verify

```sh
pytest
```

---

## Quick start

Use the toolkit directly from Python — the same harness and benchmark code work
for every controller type:

```python
from neucode import (
    FOPDTPlant, FOPDTPlantGenerator,
    SimulationHarness, StepBenchmark,
    ANNController, PIDController,
)
from neucode.experiment import Experiment

plant = FOPDTPlant(K=1.0, tau=2.0, theta=0.4)
controller = PIDController(kp=2.0, ki=0.5, kd=10.0)  # kd is industrial form: kd_raw / dt
benchmark = StepBenchmark(dt=0.01, total_time=20.0, step_value=1.0)

harness = SimulationHarness(
    plant=plant, controller=controller,
    setpoint=None, actuator_limits={'u_min': -10.0, 'u_max': 10.0},
)
result = benchmark.run(harness, get_time_series=True)
print(f"ITAE: {result.itae:.4f}  Overshoot: {result.overshoot_percent:.1f}%")
```

Swap `PIDController` for a trained `ANNController` or `SNNController` — the
harness and benchmark code are unchanged. The same pattern extends across the
toolkit's plants, tuners, trainers, and exporters; see the
[API documentation](#documentation) for the full surface. Task-oriented guides
(e.g. DAgger replacement training and Akida export) will be added over time.

---

## Firmware Deployment

The firmware targets an **STM32 Nucleo G431RB** with the X-NUCLEO-IHM16M1
(STSPIN830) driver board and an MT6701 encoder. It is built with
[PlatformIO](https://platformio.org/) (STM32Duino + SimpleFOC).

### Build and flash

```sh
cd firmware/boards/nucleo-g431rb-simplefoc
pio run              # build
pio run -t upload    # flash to a connected board
```

Generated model headers are stubbed automatically on a clean clone (see
`pre_build.py`), so the firmware builds before you run the ML pipeline. Run
pipeline step 6 to write the real `model_data.h` headers, then rebuild.

---

## Documentation

The API documentation is built with [MkDocs](https://www.mkdocs.org/) + [mkdocstrings](https://mkdocstrings.github.io/) (Python) and [mkdoxy](https://github.com/JakubAndrysek/MkDoxy) (C/firmware via Doxygen).

### Prerequisites

```sh
pip install mkdocs mkdocs-material mkdocstrings[python] mkdoxy
sudo apt install doxygen   # required by mkdoxy
```

### Serve locally

```sh
mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

### Build static site

```sh
mkdocs build
```

Output is written to `site/`.

---

## Simulation Core Development

To work on the C library in isolation, use the `Makefile` at the project root:

| Command | Description |
|---|---|
| `make all` | Build simulation core, tests, example binary, and static/shared libraries |
| `make test` | Build the C test suite |
| `make run-tests` | Build and run the C test suite |
| `make example` | Build a standalone example binary |
| `make lib` | Build static and shared library variants |
| `make clean` | Remove all build artefacts |

To regenerate `compile_commands.json` for IDE autocompletion:

```sh
make clean && bear -- make all
```

Or run the combined helper (simulation core + firmware):

```sh
./setup_vcs_compile_db.sh
```

---

## Optional & proprietary components

NeuCoDe's core installs and runs entirely on permissively licensed dependencies.
Some capabilities are gated behind optional install extras:

| Extra | Enables | Notes |
|---|---|---|
| `neucode[rl]` | RL-based PID tuner | Open-source (`gymnasium`, `stable-baselines3`) |
| `neucode[nir]` | NIR interchange export | Open-source (`nir`) |
| `neucode[akida]` | BrainChip Akida (AKD1000) export | **Proprietary.** Pulls BrainChip software under its own license (not Apache-2.0) and requires BrainChip hardware. See [NOTICE](NOTICE). |
| `neucode[docs]` | Documentation site | MkDocs toolchain |

## License & citation

NeuCoDe is released under the [Apache License 2.0](LICENSE). Third-party
attributions are listed in [NOTICE](NOTICE).

If you use NeuCoDe in academic work, please cite it. Metadata is provided in
[CITATION.cff](CITATION.cff); a BibTeX entry:

```bibtex
@software{fischer_neucode,
  author  = {Fischer, Benedikt},
  title   = {{NeuCoDe: Toolkit for Neuromorphic Control Design and Deployment on Resource-Constrained Devices}},
  year    = {2026},
  version = {0.1.0},
  license = {Apache-2.0},
  url     = {https://github.com/reipan/neucode}
}
```
