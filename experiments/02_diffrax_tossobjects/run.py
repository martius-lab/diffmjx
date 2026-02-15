import sys
from os.path import join
from pathlib import Path
import hydra
import omegaconf
import jax
import jax.numpy as jnp
import pandas as pd
import equinox as eqx
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_disable_jit", False)

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import cfg_to_flat_dct, plot_loss_grads, render_trajectory, upscale

import mujoco
from mujoco import mjx
import mjx_diffrax

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg):
    if cfg.quick:
        omegaconf.OmegaConf.update(cfg, "u0_resolution", 5)
        omegaconf.OmegaConf.update(cfg, "Nlength", 5)
    print(f"Starting run with parameters: \n{omegaconf.OmegaConf.to_yaml(cfg)}")

    local_path = join(cfg.xml.path, cfg.xml.system + ".xml")

    mj_model = mujoco.MjModel.from_xml_path(local_path)
    m = mjx.put_model(mj_model)
    if m.mesh_convex == ():
        m = m.replace(mesh_convex=None)  # empty tuple breaks adjoint gradients in diffrax
    m = m.tree_replace(cfg_to_flat_dct(cfg.xml.overwrite))

    d = mjx.make_data(m)
    d = jax.tree.map(upscale, d)  # Some versions of MJX do not upscale to float64
    d = d.replace(qpos=d.qpos.at[2].set(cfg.qz0), qvel=d.qvel.at[1].set(cfg.vy0))
    
    u0s = jnp.linspace(cfg.u0_start, cfg.u0_end, cfg.u0_resolution)

    diffrax_cfg = mjx_diffrax.DiffraxConfig(**cfg.diffrax)  # Set integration parameters for Diffrax

    # Render initial condition examples
    if cfg.render:
        for ui in [u0s[0], u0s[len(u0s) // 2], u0s[-1]]:
            d_i = d.replace(qvel=d.qvel.at[2].set(ui))
            _, traj_data = mjx_diffrax.multistep(m, d_i, nsteps=cfg.Nlength, cfg=diffrax_cfg, ctrls=None)
            render_trajectory(traj=traj_data, m=mj_model, path=str(RESULTS_DIR), name=f"{cfg.xml.system}_toss_u0={float(ui):.2f}")


    @eqx.filter_jit
    @eqx.filter_value_and_grad
    def loss(u0, m, d, r_cost_weight, Nlength, cfg_diffrax):
        d = d.replace(qvel=d.qvel.at[2].set(u0))
        d, ds = mjx_diffrax.multistep(m, d, nsteps=Nlength, cfg=cfg_diffrax, ctrls=None)

        def cost(d):
            # Ball position in y-direction (aka throwing distance)
            return d.qpos[1]
        
        return r_cost_weight * jnp.sum(jax.vmap(cost)(ds), axis=0) + cost(d)


    # Compute loss and gradient for all initial conditions
    loss_vmap = eqx.filter_vmap(loss, in_axes=(0, None, None, None, None, None))
    losses, grads = loss_vmap(u0s, m, d, cfg.r_cost_weight, cfg.Nlength, diffrax_cfg)

    pd.DataFrame({
        "u0s": u0s,
        "losses": losses,
        "grads": grads,
    }).to_csv(RESULTS_DIR / f"{cfg.xml.system}.csv", index=False)
    print(f"Data saved to {RESULTS_DIR / f'{cfg.xml.system}.csv'}")
    plot_loss_grads(u0s, losses, grads, figpath=str(RESULTS_DIR / f"{cfg.xml.system}_loss"))

if __name__ == "__main__":
    main()