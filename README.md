<div align="center">

# CAST-FV

### Compact neural state optimization under finite-volume constraints

**A solution-label-free neural framework for constructing finite-volume flow and transport states**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-12%20passed-2E8B57)](#testing)
[![Research code](https://img.shields.io/badge/status-research%20code-6F42C1)](#scope)

[中文说明](README_zh-CN.md) · [Algorithm](docs/ALGORITHM.md) · [Release checklist](docs/RELEASE_CHECKLIST.md)

</div>

CAST-FV represents the full cell-centered physical state with a compact neural map and optimizes that state directly against one finite-volume residual system. It requires no CFD solution labels, stored target trajectories, or pretrained operator weights. For unsteady problems, a fresh neural map and optimizer are created at every requested physical-time level; only the previously retained physical state enters the backward-Euler residual.

<p align="center">
  <img src="assets/castfv_workflow.png" width="92%" alt="CAST-FV workflow from prescribed physics to a retained finite-volume state">
</p>

## Highlights

- **Solution-label-free state construction:** equations, mesh metrics, and boundary data define the optimization objective.
- **A common numerical authority:** all candidate architectures are judged by the same cell-centered finite-volume residual.
- **Compact parameterizations:** CNN, U-Net, Transformer, grid GNN, and residual-corrector variants are available in 2D; the compact CNN is available in 3D.
- **Controlled physical-time advancement:** every time level starts a new neural optimization and retains its lowest-objective candidate.
- **Algorithm-only release:** no CFD reference fields, training trajectories, model checkpoints, or out-of-scope solver extensions are included.

## Installation

```bash
git clone GITHUB_REPOSITORY_URL
cd CAST-FV
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test]"
```

PyTorch installation can depend on the local CUDA configuration. For a GPU-specific build, install the appropriate PyTorch wheel first and then run `pip install -e ".[test]"`.

## Quick start

Run a small CPU example:

```bash
castfv steady \
  --dimension 2 \
  --cells 16 \
  --architecture cnn \
  --budget 25 \
  --device cpu \
  --output outputs/steady
```

The command writes the retained state, objective history, metadata, and a diagnostic field plot to `outputs/steady/`.

Run two physical-time levels with a fresh neural map at each level:

```bash
castfv unsteady \
  --dimension 2 \
  --cells 16 \
  --architecture cnn \
  --budget 20 \
  --time-step 1.0 \
  --levels 2 \
  --device cpu \
  --output outputs/unsteady
```

Compare the five 2D state parameterizations under the same numerical protocol:

```bash
castfv compare \
  --cells 16 \
  --budget 20 \
  --device cpu \
  --output outputs/architecture_comparison
```

Equivalent Python entry points are provided in [`examples/quick_start.py`](examples/quick_start.py) and [`examples/unsteady.py`](examples/unsteady.py).

## Core algorithm

For one steady state or one requested physical-time level:

1. create a neural map from coordinates and boundary-condition channels to the full cell-centered state;
2. construct boundary face values algebraically;
3. assemble continuity, momentum, and passive-scalar finite-volume residuals;
4. minimize their component-wise mean absolute residual with AdamW for a fixed update budget;
5. retain the finite candidate with the lowest objective encountered during optimization.

For unsteady advancement, the retained state becomes the sole temporal history in the next backward-Euler residual. Network weights and optimizer states are not carried across physical-time levels.

<p align="center">
  <img src="assets/fv_residual_ledger.png" width="92%" alt="Finite-volume residual assembly used as the common numerical authority">
</p>

The equations and their mapping to the implementation are documented in [`docs/ALGORITHM.md`](docs/ALGORITHM.md).

## Architectures

<p align="center">
  <img src="assets/architecture_design.png" width="94%" alt="Controlled comparison of five neural state parameterizations">
</p>

| Capability | 2D | 3D |
|---|:---:|:---:|
| Compact CNN | ✓ | ✓ |
| U-Net | ✓ | — |
| Transformer | ✓ | — |
| Grid GNN | ✓ | — |
| Residual corrector | ✓ | — |
| Steady construction | ✓ | ✓ |
| Backward-Euler advancement | ✓ | ✓ |

## Python API

```python
from castfv import CaseConfig, OptimizationConfig, optimize_state

case = CaseConfig(dimension=2, cells=16, reynolds=100.0, peclet=30.0)
optimization = OptimizationConfig(
    architecture="cnn",
    budget=25,
    learning_rate=1.0e-3,
    width=16,
)

result = optimize_state(case, optimization, device="cpu")
print(result.best_objective, result.best_iteration, result.parameters)
```

## Scope

This repository is the minimal public implementation of the CAST-FV algorithm in the associated *Physics of Fluids* manuscript. It includes Cartesian 2D/3D finite-volume residuals for incompressible flow with passive-scalar transport, steady construction, backward-Euler physical-time advancement, five 2D state parameterizations, examples, and tests.

To keep the public boundary unambiguous, it intentionally excludes:

- CFD reference solutions, snapshots, target trajectories, and paper result archives;
- pretrained models, learned operators, FNO/PINO implementations, and comparison timing data;
- pressure-projection, persistent face-flux, accepted-state, and other solver extensions outside the present method;
- private machine paths, experiment logs, and unpublished checkpoints.

The paper reports its controlled experiments on a $48\times48$ grid. The smaller settings in the quick-start commands are smoke tests; use `--cells 48` and the study-specific update budget for full-scale runs.

## Repository layout

```text
CAST-FV/
├── assets/                 # Selected conceptual figures from the manuscript
├── docs/                   # Algorithm notes and release guidance
├── examples/               # Minimal Python entry points
├── src/castfv/             # Public CAST-FV implementation
├── tests/                  # CPU unit and smoke tests
├── CITATION.cff            # GitHub citation metadata
├── pyproject.toml          # Installable package and CLI
└── README.md
```

## Testing

```bash
pytest -q
```

The test suite checks feature construction, 2D/3D zero-state residuals, all public architecture output shapes, steady optimization, and multi-level physical-time advancement.

## Citation

The associated manuscript is under preparation/submission. Until its bibliographic record is available, use the repository citation exposed by [`CITATION.cff`](CITATION.cff). Add the manuscript DOI to that file after publication.

## Code availability

Once this repository is public, the manuscript statement can read:

> The source code implementing CAST-FV is publicly available on GitHub at **GITHUB_REPOSITORY_URL**.

Replace the placeholder with the public repository URL before submission. See [`docs/CODE_AVAILABILITY.md`](docs/CODE_AVAILABILITY.md).

## License

A software license has not yet been selected. Before making the repository public, add a `LICENSE` file; the MIT License is a suitable permissive default if it matches the authors' intended reuse terms. Until a license is added, no reuse license is granted.

## Contact

- Software: Demin Liu — liudemin@stu.xjtu.edu.cn
- Corresponding author: Tieyu Gao — sunmoon@mail.xjtu.edu.cn

## Acknowledgement

The work is Supported by HPC Platform, Xi’an Jiaotong University. Interdisciplinary Doctoral Training Support Project at Xi'an Jiaotong University(ID: IDT2025)”
