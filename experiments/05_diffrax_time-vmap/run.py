import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import hydra
import jax
import jax.numpy as jnp
import omegaconf
import pandas as pd
import yaml

import equinox as eqx
import mujoco
from mujoco import mjx

jax.config.update("jax_debug_nans", False)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "high")
jax.config.update("jax_disable_jit", False)

# Add parent directory for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import cfg_to_flat_dct, profile, render_trajectory

import mjx_diffrax

print(jax.default_backend())

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def build_sphere_chain_model(
    num_spheres: int,
    zmin: float,
    zmax: float,
    radius: float = 0.01,
    gap: float = 0.01,
):
    rng = np.random.default_rng(42)

    xml_lines = [
        '<mujoco model="sphere_chain">',
        '  <option gravity="0. 0. -2." timestep="0.01" solver="Newton" iterations="4" ls_iterations="10" tolerance="1e-8" cone="pyramidal"/>,'
        '',
        '<option>',
        '<flag warmstart="enable" refsafe="disable" eulerdamp="disable"/>',
        '</option>',
        '',
        '<default>',
        '<geom solimp="0.0 0.95 0.001 0.5 2" solref="0.005 1.0"/>',
        '</default>',
        '',
        '<visual>',
        '<map fogend="5" fogstart="3"/>',
        '</visual>',
        '',
        '<asset>',
        '<texture type="skybox" builtin="gradient" rgb1="1.0 1.0 1.0" rgb2="1.0 1.0 1.0" width="32" height="512"/>',
        '<texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.9 0.9 0.9" rgb2="0.95 0.95 0.95"/>',
        '<material name="grid" texture="grid" texrepeat="5 5" texuniform="true" reflectance=".0"/>',
        '<texture name="ball" type="cube" builtin="flat" mark="cross" width="64" height="64" rgb1="0.3 0.5 0.9" rgb2="0.3 0.5 0.9" markrgb="1 1 1"/>',
        '<material name="ball" texture="ball" texuniform="true" rgba="1 1 1 1."/>',
        '</asset>',
        '',
        '  <worldbody>',
        '<light cutoff="100" diffuse="1 1 1" dir="-2. 0 -1.3" directional="true" exponent="1" pos="1 0 1.3" specular=".1 .1 .1"/>',
        '<camera name="track" pos="0.597 0.050 0.085" xyaxes="-0.436 0.900 -0.000 0.101 0.049 0.994"/>',
        '    <!-- Ground plane at z = 0 -->',
        '    <geom name="ground" type="plane" condim="3" pos="0 0 0" size="20 20 0.125" material="grid"/>',
    ]

    dx = 2 * radius + gap  # spacing between sphere centers

    for i in range(num_spheres):
        x = -dx * (i // 10)
        y = -i * dx + (i // 10) * 10 * dx
        z = rng.uniform(zmin, zmax)
        xml_lines += [
            f'    <body name="sphere_{i}" pos="{x:.5f} {y:.5f} {z:.5f}">',
            f'      <joint name="free_{i}" type="free"/>',
            f'      <geom name="sphere_geom_{i}" type="sphere" '
            f'size="{radius:.5f}" density="1" material="ball"/>',
            '    </body>',
        ]

    xml_lines += [
        '  </worldbody>',
        '</mujoco>',
    ]

    xml = "\n".join(xml_lines)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    return model, data, xml


def loss_fn(u0, m, d, Nlength, r_cost_weight, cfg_diffrax):
    """Sum of running costs."""
    d = d.replace(qvel=d.qvel.at[2].set(u0))
    d, ds = mjx_diffrax.multistep(m, d, nsteps=Nlength, cfg=cfg_diffrax)
    r_cost = jnp.sum(jax.vmap(lambda di: di.qpos[2])(ds))
    return r_cost_weight * r_cost + d.qpos[2]


def create_random_batch(d_mjx, num_batch, num_spheres, zmin, zmax):
    """Create num_batch copies of the MJX data object with random height in range [zmin,zmax]."""
    rng = jax.random.split(jax.random.PRNGKey(0), num_batch)

    def replace_heights(rng):
        random_heights = jax.random.uniform(rng, (num_spheres,), minval=zmin, maxval=zmax)
        return d_mjx.replace(qpos=d_mjx.qpos.at[2::7].set(random_heights))

    d_mjx_batch = jax.vmap(lambda rng: replace_heights(rng))(rng)
    return d_mjx_batch


def fd_gradient(fn, x, *args, eps=1e-2):
    return (fn(x + eps, *args) - fn(x - eps, *args)) / (2 * eps)


def render_u0(u0, m, d, mj_model, Nlength, cfg_diffrax, path, name, camera):
    """Visualisation for a single force."""
    d_i = d.replace(qvel=d.qvel.at[2].set(u0))
    _, traj_data = mjx_diffrax.multistep(m, d_i, nsteps=Nlength, cfg=cfg_diffrax)
    render_trajectory(
        traj=traj_data,
        m=mj_model,
        path=path,
        name=name,
        camera=camera,
    )


def analyze(
    name,
    m,
    mj_model,
    mj_data,
    num_spheres_tmp,
    num_batch_tmp,
    zmin,
    zmax,
    u0,
    dt,
    r_cost_weight,
    camera,
    cfg_diffrax,
    path,
    render_video=False,
    **kwargs,
):
    Nlength = int(dt / m.opt.timestep)
    d_init = mjx.put_data(mj_model, mj_data)
    d = create_random_batch(d_init, num_batch_tmp, num_spheres_tmp, zmin, zmax)

    def loss_fn_vmap(u0, m, d_batch, Nlength, r_cost_weight, cfg_diffrax):
        return jax.vmap(loss_fn, in_axes=(None, None, 0, None, None, None))(
            u0, m, d_batch, Nlength, r_cost_weight, cfg_diffrax
        )

    u0 = jnp.array(u0, dtype=jnp.float64)
    args = (m, d, Nlength, r_cost_weight, cfg_diffrax)

    jit_time_fw, runtime_fw, (_, loss_fn_compiled) = profile(loss_fn_vmap, u0, *args)
    print(f"JIT time Forward: {jit_time_fw}, runtime Forward: {runtime_fw}")

    def loss_and_grad_fn_vmap(u0, m, d_batch, Nlength, r_cost_weight, cfg_diffrax):
        return jax.vmap(eqx.filter_value_and_grad(loss_fn), in_axes=(None, None, 0, None, None, None))(
            u0, m, d_batch, Nlength, r_cost_weight, cfg_diffrax
        )

    jit_time_bw, runtime_bw, ((loss, grad), _) = profile(
        loss_and_grad_fn_vmap, u0, *args
    )
    print(f"JIT time Backward: {jit_time_bw}, runtime Backward: {runtime_bw}")

    grad_fd = fd_gradient(loss_fn_compiled, u0, *args)
    print(f"Loss: {loss}, Gradient: {grad}, Gradient FD: {grad_fd}")

    if render_video:
        render_u0(
            u0, m, d_init, mj_model, Nlength, cfg_diffrax,
            path=path, name=f"{name}", camera=camera,
        )

    return loss, grad, grad_fd, jit_time_fw, runtime_fw, jit_time_bw, runtime_bw


@hydra.main(config_path=".", config_name="config", version_base="1.3")
def main(cfg):
    print(f"Starting run with parameters: \n{omegaconf.OmegaConf.to_yaml(cfg)}")

    cfg_diffrax = mjx_diffrax.DiffraxConfig(**cfg.diffrax)

    columns = ["loss", "grad", "grad_fd", "jit_time_fw", "runtime_fw", "jit_time_bw", "runtime_bw"]
    csv_path = str(RESULTS_DIR / "sphere_chain_results.csv")
    configs_path = str(RESULTS_DIR / "sphere_chain_configs.txt")

    # Load existing results if present
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame(columns=columns)

    for num_spheres, num_batch in product(cfg.analyze.num_spheres, cfg.analyze.num_batch):
        setting_name = f"num_spheres={num_spheres}-num_batch={num_batch}"

        if setting_name in df.index and not cfg.force_overwrite:
            print(f"Skipping {setting_name}: already in results")
            continue

        mj_model, mj_data, _ = build_sphere_chain_model(
            num_spheres, zmin=cfg.analyze.zmin, zmax=cfg.analyze.zmax
        )
        m = mjx.put_model(mj_model)
        if m.mesh_convex == ():
            m = m.replace(mesh_convex=None)
        overwrite_config = cfg_to_flat_dct(cfg.xml.overwrite)
        m = m.tree_replace(overwrite_config)

        print(f"Analyzing {setting_name}")
        (
            loss,
            grad,
            grad_fd,
            jit_time_fw,
            runtime_fw,
            jit_time_bw,
            runtime_bw,
        ) = analyze(
            name=setting_name,
            m=m,
            mj_model=mj_model,
            mj_data=mj_data,
            num_spheres_tmp=num_spheres,
            num_batch_tmp=num_batch,
            **cfg.analyze,
            cfg_diffrax=cfg_diffrax,
            path=str(RESULTS_DIR),
            render_video=False,
        )

        df.loc[setting_name] = [
            loss, grad, grad_fd,
            jit_time_fw, runtime_fw, jit_time_bw, runtime_bw,
        ]
        print(df)
        df.to_csv(csv_path)

        # Append setting config to single txt file
        setting_dict = omegaconf.OmegaConf.to_container(cfg.xml.overwrite, resolve=True)
        with open(configs_path, "a") as f:
            f.write(f"--- {setting_name} ---\n")
            yaml.dump(setting_dict, f)
            f.write("\n")


if __name__ == "__main__":
    main()
