# diffmjx: MuJoCo XLA with informative contact gradients

[📄 Paper](www.link.com)  |  [🧑‍💻 Code](www.link.com)  |  [📚 Docs](www.link.com)

MuJoCo's MJX backend enables GPU-accelerated, differentiable physics simulation in JAX.
However, the gradients produced by MJX are often ill-defined as: 

- **collision detection** resorts to non-differentiable operations, 

- **contact solver** yield non-zero gradients only for colliding objects, 

- **numerical integrators** using fixed stepsizes cause gradient oscillations due to discretization errors.

**diffmjx** is an umbrella repository that integrates three libraries addressing each of
these issues, enabling gradient-based optimization through rigid-body contact simulations.


## Components 

### mujoco-mjx (fork)
A fork of [MuJoCo XLA](https://github.com/google-deepmind/mujoco) implementing the following fixes:

- **Contact Force from a Distance (CFD)**: Contact constraints apply miniscule forces even when no
  contact is active, guiding the optimizer toward contact configurations. Straight-through estimation allows to use CFD forces only for gradient computation keeping the forward simulation untouched.

- **`col_soft_enable`** — smoothly differentiable collision detection via [Softjax](#softjax), smoothing the collision detection pipeline.

- **`scan_loop`** — replaces `jax.lax.while_loop` in the constraint solver with a scan-based loop.

### mjx_diffrax
Replaces MJX's built-in Euler and RK4 integrators with adaptive ODE solvers (Tsit5,
Dopri5, and others) from [Diffrax](https://github.com/patrick-kidger/diffrax). Adaptive
stepsize control improves gradient quality by reducing integration errors. Provides `step()` and `multistep()` entry points.


> [!CAUTION]  
> For long simulations, applying `jax.jit` (or alternatively `eqx.filter_jit`) on `multistep()` compiles the full simulation loop causing extreme compile times. 
If you need gradients for long trajectories, only jit `step`.

### softjax

Smooth, differentiable drop-in replacements for non-differentiable JAX operations
(`abs`, `min`, `max`, `sort`, `where`, and others). Supports three modes: **soft**
(smooth approximation), **hard** (original non-smooth op), and **straight-through
estimation** (forward pass uses the hard op, backward pass uses the soft surrogate).

[📄 Paper](www.link.com)  |  [🧑‍💻 Code](www.link.com)  |  [📚 Docs](www.link.com)

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

