# Contributing to NeuCoDe

Thanks for your interest in improving NeuCoDe! This project began as a Master's
thesis toolkit and is now developed in the open. Contributions of all kinds are
welcome: bug reports, documentation fixes, new controllers or plant models,
firmware improvements, and additional export backends.

By contributing, you agree that your contributions will be licensed under the
project's [Apache License 2.0](LICENSE).

## Ways to contribute

- **Report a bug** — open an issue with a minimal reproduction (see the issue
  template). Include your OS, Python version, and whether optional extras
  (`torch`, `akida`, ...) are installed.
- **Suggest a feature** — open an issue describing the use case before writing
  code, so we can agree on scope and design.
- **Send a pull request** — for anything beyond a trivial fix, please open (or
  comment on) an issue first.

## Development setup

```sh
# System build dependencies (Debian/Ubuntu)
sudo apt update && sudo apt install build-essential bear python3 python3-venv python3-dev python3-tk python3-pip

# Python environment
python3 -m venv venv
source venv/bin/activate

# Editable install with dev tooling (pytest, build, cython, RL + NIR extras)
pip install -e ".[dev]"
```

Optional extras, installed only if you work on those paths:

```sh
pip install -e ".[akida]"   # BrainChip Akida export (proprietary SDK; see NOTICE)
pip install -e ".[docs]"    # MkDocs documentation site
```

## Running the tests

```sh
pytest                 # default suite (RL tests excluded)
pytest -m rl           # RL tuner tests (memory-heavy, run separately)
```

The C simulation core has its own test suite:

```sh
make run-tests
```

## Coding conventions

- **Python**: follow the style of the surrounding code (PEP 8, 4-space indent).
  Public classes and functions should have docstrings; the docs site is
  generated from them via `mkdocstrings`.
- **C / firmware**: formatted with `clang-format` (see `.clang-format`).
- **ASCII only** in source files (`.py`, `.c`, `.h`) — no Unicode arrows, degree
  signs, or em-dashes. Use `deg`, `->`, etc.
- Keep changes focused; unrelated refactors belong in separate PRs.

## Commit messages

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```
type(optional-scope): short imperative summary
```

Common types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`,
`ci`, `chore`, `style`. The scope in parentheses is an optional module hint.
Examples:

```
feat(exporters): add ONNX export backend
fix(harness): correct settling-time metric
docs(readme): add citation section
ci: cache pip and apt dependencies
```

Keep the summary in the imperative mood and roughly under 72 characters; add a
body after a blank line when more context helps.

## Pull request checklist

- [ ] Tests pass (`pytest` and, if you touched the C core, `make run-tests`).
- [ ] New/changed public APIs have docstrings.
- [ ] `import neucode` still works **without** optional extras installed
      (heavy/proprietary deps must stay lazily imported).
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).

## Reporting security issues

Please do **not** open a public issue for security-sensitive reports. See
[SECURITY.md](SECURITY.md) for how to report privately.
