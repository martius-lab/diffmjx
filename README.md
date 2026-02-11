# diffmjx: Differentiable MuJoCo Simulations with Well-Defined Gradients

MuJoCo's MJX backend enables GPU-accelerated, differentiable physics simulation in JAX.
However, the gradients produced by MJX are often ill-defined: non-smooth contact dynamics
introduce discontinuities, fixed-step integrators miss collision events, and several JAX
primitives used internally are not differentiable.

**diffmjx** is an umbrella repository that integrates three libraries addressing each of
these issues, enabling gradient-based optimization through rigid-body contact simulations.

## Components

### softjax

Smooth, differentiable drop-in replacements for non-differentiable JAX operations
(`abs`, `min`, `max`, `sort`, `where`, and others). Supports three modes: **soft**
(smooth approximation), **hard** (original non-smooth op), and **straight-through
estimation** (forward pass uses the hard op, backward pass uses the soft surrogate).

[Documentation](https://a-paulus.github.io/softjax/)

### mjx_diffrax

Replaces MJX's built-in Euler and RK4 integrators with adaptive ODE solvers (Tsit5,
Dopri5, and others) from [diffrax](https://github.com/patrick-kidger/diffrax). Adaptive
stepping lets the solver detect and resolve collision events that fixed-step integrators
miss. Provides `step()` and `multistep()` entry points.

### mujoco-mjx (fork)

A fork of [MuJoCo](https://github.com/google-deepmind/mujoco) with three additions to
the MJX backend:

- **Contact Force Distribution (CFD)** — provides non-zero gradient signal even when no
  contact is active, guiding the optimizer toward contact configurations.
- **`col_soft_enable`** — enables soft collision geometry via softjax, smoothing the
  collision detection pipeline.
- **`scan_loop`** — replaces `jax.lax.while_loop` in the constraint solver with a
  scan-based loop, making the solver compatible with reverse-mode automatic
  differentiation.

## Installation

### Prerequisites

- Python 3.12+
- CUDA 12 (for GPU acceleration)
- SSH access to the private component repositories

### Setup

```bash
bash setup.sh
uv sync
```

`setup.sh` clones the three component repositories into `external/` and creates a
virtual environment. `uv sync` installs all dependencies with the local packages in
editable mode.

Run an experiment with:

```bash
uv run experiments/<experiment>/run.py
```

## Experiments

| Directory | Description |
|---|---|
| `exp_01_diffrax_toyexample` | 1-D bouncing ball comparing Euler, ideal, and adaptive integration |
| `exp_02_diffrax_tossobjects` | Object tossing with adaptive diffrax integration in MJX |
| `exp_03_cfd_billiard` | Billiard optimization using Contact Force Distribution gradients |
| `exp_04_optimize-then-discretize` | WIP |
| `exp_05_diffrax_time-toss` | WIP |
| `exp_06_diffrax-vmap` | WIP |

See `experiments/` for full working examples.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation
If this library helped your academic work, please consider citing:

```bibtex
@misc{diffmjx2025,
  author = {Paulus, Anselm and Geist, {A. Ren\'e} and Martius, Georg},
  title = {Softjax},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/a-paulus/softjax}}
}
```

Also consider starring the project [on GitHub](https://github.com/a-paulus/softjax)!

Special thanks and credit go to [Patrick Kidger](https://kidger.site) for the awesome [JAX repositories](https://github.com/patrick-kidger) that served as the basis for the documentation of this project.

