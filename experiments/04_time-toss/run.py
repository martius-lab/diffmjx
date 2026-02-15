import os
import sys
from itertools import product
from os.path import join
from pathlib import Path
import hydra
import jax
import jax.numpy as jnp
import omegaconf
import pandas as pd
import yaml
import equinox as eqx
import mujoco
from mujoco import mjx

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_disable_jit", False)

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import cfg_to_flat_dct, profile, render_trajectory, upscale

import mjx_diffrax

print(jax.default_backend())

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def loss_fn(u0, m, d, Nlength, r_cost_weight, cfg_diffrax):
    """Sum of running costs."""
    #val = d[f'{cfg.x0.name}'].at[cfg.x0.idx].set(cfg.x0.val)
    #d = d.tree_replace({f'{cfg.x0.name}': val})
    # cost_idx
    d = d.replace(qvel=d.qvel.at[2].set(u0))
    d, ds = mjx_diffrax.multistep(m, d, nsteps=Nlength, cfg=cfg_diffrax)
    
    def cost(d):
        return d.qpos[1]
    
    return r_cost_weight * jnp.sum(jax.vmap(cost)(ds)) + cost(d)


def render_u0(u0, m, d, mj_model, Nlength, cfg_diffrax, path, name):
    """Visualisation for a single force."""
    d_i = d.replace(qvel=d.qvel.at[2].set(u0))
    _, traj_data = mjx_diffrax.multistep(m, d_i, nsteps=Nlength, cfg=cfg_diffrax)
    render_trajectory(
        traj=traj_data,
        m=mj_model,
        path=path,
        name=name,
    )


def fd_gradient(fn, x, *args, eps=1e-2):
    return (fn(x + eps, *args) - fn(x - eps, *args)) / (2 * eps)


def analyze(
    name,
    m,
    mj_model,
    u0,
    dt,
    r_cost_weight,
    qz0,
    vy0,
    cfg_diffrax,
    path,
    render_video=False,
):
    d = mjx.make_data(m)
    d = jax.tree.map(upscale, d)
    d = d.replace(qpos=d.qpos.at[2].set(qz0), qvel=d.qvel.at[1].set(vy0))

    Nlength = int(dt / m.opt.timestep)

    u0 = jnp.array(u0)
    args = (m, d, Nlength, r_cost_weight, cfg_diffrax)

    jit_time_fw, runtime_fw, (_, loss_compiled) = profile(loss_fn, u0, *args)
    print(f"JIT time Forward: {jit_time_fw}, runtime Forward: {runtime_fw}")

    loss_and_grad_fn = eqx.filter_value_and_grad(loss_fn)
    jit_time_bw, runtime_bw, ((loss_val, grad), _) = profile(loss_and_grad_fn, u0, *args)
    print(f"JIT time Backward: {jit_time_bw}, runtime Backward: {runtime_bw}")

    grad_fd = fd_gradient(loss_compiled, u0, *args)
    print(f"Loss: {loss_val}, Gradient: {grad}, Gradient FD: {grad_fd}")

    if render_video:
        render_u0(u0, m, d, mj_model, Nlength, cfg_diffrax, path=path, name=f"{name}_u0={u0}")

    return loss_val, grad, grad_fd, jit_time_fw, runtime_fw, jit_time_bw, runtime_bw


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg):
    if cfg.quick:
        omegaconf.OmegaConf.update(cfg, "analyze.dt", 0.1)
    print(f"Starting run with parameters: \n{omegaconf.OmegaConf.to_yaml(cfg)}")

    columns = ["loss", "grad", "grad_fd", "jit_time_fw", "runtime_fw", "jit_time_bw", "runtime_bw"]

    for u0, system in product(cfg.u0s, cfg.systems):
        local_path = join(cfg.xml.path, system + ".xml")
        mj_model = mujoco.MjModel.from_xml_path(local_path)

        csv_path = str(RESULTS_DIR / f"{system}_results.csv")
        configs_path = str(RESULTS_DIR / f"{system}_configs.txt")

        # Load existing results if present
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path, index_col=0)
        else:
            df = pd.DataFrame(columns=columns)

        settings = list(cfg.xml.overwrite_settings.items())
        if cfg.quick:
            settings = settings[:1]

        for setting_name, setting in settings:
            if setting_name in df.index and not cfg.force_overwrite:
                print(f"Skipping {setting_name}: already in results")
                continue
            
            # Setup MJX model
            m = mjx.put_model(mj_model)
            if m.mesh_convex == ():
                m = m.replace(mesh_convex=None)
            m = m.tree_replace(cfg_to_flat_dct(cfg.xml.overwrite))
            
            if 'mjx' in setting_name:
                integrator_map = {
                    "Euler": mujoco.mjtIntegrator.mjINT_EULER,
                    "RK4": mujoco.mjtIntegrator.mjINT_RK4,
                    "ImplicitFast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
                    }
                cfg_diffrax = None
                m = m.tree_replace({'opt.integrator': integrator_map[setting.integrator],
                                    'opt.timestep': setting.timestep})
            else:
                cfg_diffrax = mjx_diffrax.DiffraxConfig(**setting.diffrax)
                
                # CosntantStepsize controller uses m.timestep as stepsize
                if cfg_diffrax.stepsize_controller == 'Constant':
                    m = m.tree_replace({'opt.timestep': setting.timestep})

            print(f"Analyzing {system} (u0={u0}) with {setting_name}")

            render_video = cfg.render and "1e-10" in setting_name

            (
                loss_val,
                grad_val,
                grad_fd,
                jit_time_fw,
                runtime_fw,
                jit_time_bw,
                runtime_bw,
            ) = analyze(
                setting_name,
                m,
                mj_model,
                u0,
                **cfg.analyze,
                cfg_diffrax=cfg_diffrax,
                path=str(RESULTS_DIR),
                render_video=render_video,
            )

            df.loc[setting_name] = [
                loss_val, grad_val, grad_fd,
                jit_time_fw, runtime_fw, jit_time_bw, runtime_bw,
            ]
            print(df)
            df.to_csv(csv_path)

            # Append setting config to single txt file
            setting_dict = omegaconf.OmegaConf.to_container(setting, resolve=True)
            with open(configs_path, "a") as f:
                f.write(f"--- {setting_name} ---\n")
                yaml.dump(setting_dict, f)
                f.write("\n")


if __name__ == "__main__":
    main()
