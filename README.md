<h2 align='center'>diffmjx: MuJoCo XLA with informative contact gradients</h2>


[![arXiv paper](https://img.shields.io/badge/arXiv-paper-salmon)](https://arxiv.org/abs/2506.14186)

MuJoCo's MJX backend enables GPU-accelerated, differentiable physics simulation in JAX.
However, the gradients produced by MJX are often erroneous, as: 

- **collision detection** may use non-differentiable operations, 

- **contact solver** yield non-zero gradients only for colliding objects, 

- **numerical integrators** with fixed stepsizes cause gradient oscillations due to discretization errors.

**diffmjx** is an umbrella repository that integrates three libraries each of which addresses one of the above issues, enabling gradient-based optimization through rigid-body contact simulations.

## Libraries

<p align="center">
  <img alt="diffmjx overview" src="imgs/diffmjx_overview.png">
</p>

### mujoco-mjx (fork)

[![Github repo](https://img.shields.io/badge/Github-repo-blue)](https://github.com/a-paulus/mujoco)

A fork of [MuJoCo XLA](https://github.com/google-deepmind/mujoco) implementing the following features:

- **Contact Force from a Distance (CFD)**: Contact constraints apply miniscule forces even when no
  contact is active, guiding the optimizer toward contact configurations. Straight-through estimation allows to use CFD forces only for gradient computation keeping the forward simulation untouched.

- **`col_soft_enable`** — smoothly differentiable collision detection via [Softjax](#softjax), smoothing the collision detection pipeline.

- **`scan_loop`** — replaces `jax.lax.while_loop` in the constraint solver with a scan-based loop.

### mjx_diffrax
[![Github repo](https://img.shields.io/badge/Github-repo-blue)](https://github.com/a-paulus/mjx_diffrax)

Replaces MJX's built-in Euler and RK4 integrators with adaptive ODE solvers (Tsit5,
Dopri5, and others) from [Diffrax](https://github.com/patrick-kidger/diffrax). Adaptive
stepsize control improves gradient quality by reducing integration errors. Provides `step()` and `multistep()` entry points.


> [!NOTE]  
> For long simulations, applying `jax.jit` (or alternatively `eqx.filter_jit`) on `multistep()` compiles the full simulation loop causing extreme compile times.

### softjax
[![Github repo](https://img.shields.io/badge/Github-repo-blue)](https://github.com/a-paulus/softjax)
[![arXiv paper](https://img.shields.io/badge/arXiv-paper-salmon)](https://arxiv.org/abs/2603.08824) 

Smooth, differentiable drop-in replacements for non-differentiable JAX operations
(`abs`, `min`, `max`, `sort`, `where`, and others). This repo is used to soften the discrete / discontinuous operators in MJX's collision detection. 
Some collision functions such as `plane-{sphere, ellipsoid, capsule}` and `sphere-sphere` are already smooth in MJX, whereas SoftJax is used to soften the `plane-{cylinder, cube}` and `cube-cube` collisions. More involved collision detection functions such as `mesh-mesh` collisions are currently not supported.


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

`setup.sh` clones the required repositories into `external/` and creates a
virtual environment. `uv sync` installs all dependencies with the local packages in
editable mode.

## Experiments
Run an experiment with:

```bash
uv run experiments/<experiment>/run.py
```

| Directory | Description |
|---|---|
| `exp_01_diffrax_toyexample` | 1-D bouncing ball comparing Euler, ideal, and adaptive integration |
| `exp_02_diffrax_tossobjects` | Object tossing with adaptive diffrax integration in MJX |
| `exp_03_cfd_billiard` | Billiard optimization using Contact Force Distribution gradients |
| `exp_04_optimize-then-discretize` | WIP |
| `exp_05_diffrax_time-toss` | WIP |

See `experiments/` for full working examples.

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation
If this library helped your academic work, please consider citing:

```bibtex
@inproceedings{
paulus2026differentiable,
title={Differentiable Simulation of Hard Contacts with Soft Gradients for Learning and Control},
author={Anselm Paulus and Andreas Ren{\'e} Geist and Pierre Schumacher and V{\'\i}t Musil and Simon Rappenecker and Georg Martius},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
}
```

(Also consider starring the project [on GitHub](https://github.com/a-paulus/softjax))
