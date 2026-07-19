# NeuCoDe API Documentation

Reference documentation for the NeuCoDe toolkit — an end-to-end platform for the design, training, and embedded deployment of neural network-based feedback controllers.

For installation, quick-start, and pipeline usage see the `README.md` in the project root.

---

## Sections

- **[Python API](neucode/controllers.md)** — all public Python modules (controllers, plants, harness, trainers, tuners, exporters, …)
- **[SimCore Cython Wrapper](neucode/simcore.md)** — `Simulation` and `StandaloneMetrics` classes exposed by the Cython extension
- **[SimCore C API](simcore/files.md)** — Doxygen reference for the pure-C simulation engine (`neucode/c_src/`)
- **[Firmware C API](firmware/files.md)** — Doxygen reference for the embedded firmware stack
