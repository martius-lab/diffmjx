import time
from collections.abc import MutableMapping
from os.path import join
from omegaconf import OmegaConf
from jax import Array
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import mediapy
import mujoco
from tqdm import tqdm
    
sns.set_style("ticks")

plt.rcParams.update(
    {
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "mathtext.fontset": "cm",
    }
)

colors = [
    (0.368, 0.507, 0.71),  # Blue
    (0.881, 0.611, 0.142),  # Orange
    (0.923, 0.386, 0.209),  # Red
    (0.56, 0.692, 0.195),  # Green
    (0.528, 0.471, 0.701),  # Purple
    (0.772, 0.432, 0.102),  # Orange
    (0.364, 0.619, 0.782),  # Light blue
    (0.572, 0.586, 0.0),  # Moss green
    (102 / 255.0, 88 / 255.0, 84 / 255.0),  # Olive green
    (0.46, 0.193, 0.104),  # Dark red
    (0.973, 0.436, 0.229),  # Light red
]


def profile(func, *args, warmups=1, runs=1, **jit_kwargs):
    """
    Measure the JIT compilation time and mean runtime of a JAX function.

    Parameters:
        func       : Callable – the function to JIT and measure
        *args      : arguments to pass to the function for compilation & timing
        warmups    : int – how many warm-up calls before timing (default: 1)
        runs       : int – number of timed calls to average (default: 10)
        **jit_kwargs : additional keyword args passed to `jax.jit`

    Returns:
        (compile_time_in_seconds, mean_runtime_in_seconds)
    """
    import equinox as eqx

    # JIT compile and measure compile time
    t0 = time.perf_counter()
    compiled_fn = eqx.filter_jit(func, **jit_kwargs).lower(*args).compile()
    compile_time = time.perf_counter() - t0

    # Warm up
    for _ in range(warmups):
        out = compiled_fn(*args)
        out = block_until_ready(out)

    # Measure runtime
    t0 = time.perf_counter()
    for _ in range(runs):
        out = compiled_fn(*args)
        out = block_until_ready(out)
    mean_runtime = (time.perf_counter() - t0) / runs

    return compile_time, mean_runtime, (out, compiled_fn)


def block_until_ready(x):
    import jax

    return jax.tree.map(lambda _x: _x.block_until_ready(), x)


def init_mjxmodel(mj_model, overwrite):
    import mujoco.mjx as mjx

    m = mjx.put_model(mj_model)
    if m.mesh_convex == ():
        m = m.replace(
            mesh_convex=None
        )  # empty tuple breaks adjoint gradients in diffrax
    m = m.tree_replace(flatten(overwrite["opt"]))
    return m


def flatten(dictionary, parent_key="", separator="."):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


def cfg_to_flat_dct(cfg, resolve: bool = True):
    cfg_dct = OmegaConf.to_container(cfg, resolve=resolve)
    flat_cfg_dct = flatten(cfg_dct)
    return flat_cfg_dct


def update_mjstate(model, data, pvc):
    nq, nv, nu = model.nq, model.nv, model.nu
    data.qpos = pvc[:nq]
    data.qvel = pvc[nq : nq + nv]
    data.ctrl = pvc[nq + nv : nq + nv + nu]
    return data


def render_trajectory(
    traj,
    m,
    path="./",
    name="video",
    height=480,
    width=640,
    vis_contacts=False,
    reduce_video_size=False,
    slowmotion=False,
    camera="track",
):
    if isinstance(traj, Array):
        traj = np.array(traj)
    else:
        traj = np.concatenate([traj.qpos, traj.qvel, traj.ctrl], axis=-1)


    print("Rendering trajectory...")

    # Calculate FPS based on model timestep
    orig_fps = 1 / m.opt.timestep
    print(f"Original FPS based on timestep: {orig_fps}")

    # Adjust trajectory if FPS is too high
    if orig_fps > 50:
        # Calculate the sampling interval to get 100 FPS
        interval = int(orig_fps / 50)
        traj = traj[0::interval]
        fps = 50
        print(f"Adjusted to 50 FPS, sampling every {interval} frames")
    else:
        fps = orig_fps

    if reduce_video_size:
        # Render trajectory at FPS/5 to reduce video size
        fps = int(fps / 5)
        traj = traj[0::5]

    data = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height, width)
    frames = []

    options = mujoco.MjvOption()

    if vis_contacts:
        # visualize contact frames and forces, make body transparent
        mujoco.mjv_defaultOption(options)
        # options.frame = mujoco.mjtFrame.mjFRAME_GEOM  # Show body frames
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

        # tweak scales of contact visualization elements
        m.vis.scale.contactwidth = 0.05
        m.vis.scale.contactheight = 0.015
        m.vis.scale.forcewidth = 0.03
        m.vis.map.force = 0.2

    for i in tqdm(range(traj.shape[0])):
        data = update_mjstate(m, data, traj[i])
        mujoco.mj_forward(m, data)
        renderer.update_scene(data, camera=camera, scene_option=options)
        frame = renderer.render()
        frames.append(frame)

    if slowmotion:
        fps = fps / 5

    print(f"Writing video with FPS: {fps}")
    mediapy.write_video(join(path, name + ".mp4"), frames, fps=fps)
    try:
        import wandb
        if wandb.run is not None:
            wandb.log({f"video_{name}": wandb.Video(join(path, name + ".mp4"), format="mp4")})
    except ImportError:
        pass


def upscale(x):
    """Convert data to 64bit as make_data gives data in 32bit
    Code source: https://github.com/google-deepmind/mujoco/issues/2237"""
    import jax.numpy as jnp

    if "dtype" in dir(x):
        if x.dtype == jnp.int32:
            return jnp.int64(x)
        elif x.dtype == jnp.float32:
            return jnp.float64(x)
    return x


def unroll(m, d, n_steps: int):
    """Unroll the dynamics using scan."""
    import jax
    import mujoco.mjx as mjx

    def body(_d, _):
        _d = mjx.step(m, _d)
        return _d, _d

    d1, ds = jax.lax.scan(body, d, length=n_steps)
    return d1, ds


# def to_systemstate(d: Data):
#     return SystemState(d.qpos, d.qvel, d.qacc, d.time)


def plot_loss_grads(
    u0s, losses, grads, figpath="./loss_gradients_last", xlabel="Velocity in x"
):
    # fig, ax1 = plt.subplots(figsize=(5, 3.09), dpi=300)
    fig, ax1 = plt.subplots(figsize=(4, 2.472), dpi=300)

    color1 = "black"  # colors[3] # "black" #
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Loss", color=color1)
    ax1.plot(u0s, losses, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color2 = colors[0]
    ax2.set_ylabel(r"Gradient", color=color2)
    ax2.plot(u0s, grads, color=color2)
    ax2.axhline(y=0, linestyle="--", color="grey", linewidth=1.0)
    ax2.tick_params(axis=r"y", labelcolor=color2)

    axcolor = "gray"
    ax2.spines["left"].set_color(axcolor)  # setting up Y-axis tick color to red
    ax2.spines["right"].set_color(axcolor)
    ax2.spines["bottom"].set_color(axcolor)
    ax1.tick_params(axis="x", colors=axcolor)
    ax1.tick_params(axis="y", colors=axcolor)
    ax2.tick_params(axis="x", colors=axcolor)
    ax2.tick_params(axis="y", colors=axcolor)
    # ax1.tick_params(colors="black", which='both')
    # ax2.tick_params(colors="black", which='both')
    sns.despine(right=False)
    ax1.margins(x=0)
    ax2.margins(x=0)

    fig.tight_layout()
    print(f"Saving figure to {figpath}")
    fig.savefig(figpath + ".pdf")
    fig.savefig(figpath + ".png")
    return fig


def plot_loss_grads_list(
    u0s_list,
    losses_list,
    grads_list,
    factors=None,
    figpath="./loss_gradients_last",
    xlabel="Velocity in x",
    ylabel1="Loss",
    ylabel2="Gradient",
    title_list=None,
    numgrad=False,
):
    if factors is None:
        factors = [1.0] * len(u0s_list)

    fig, axs = plt.subplots(
        1, len(u0s_list), figsize=(10, 2.472), dpi=300, sharey=True, squeeze=False
    )

    ylims = (
        np.min(np.array(grads_list)) * 1.05,
        np.max(np.array(grads_list)) * 1.05,
    )

    for i in range(len(u0s_list)):
        u0s = u0s_list[i]
        losses = losses_list[i]
        grads = grads_list[i]

        if numgrad:
            numerical_grad = np.gradient(np.array(losses[0]), np.array(u0s[0]))
            grads.append(factors[i] * numerical_grad)  # Don't reassign, just append
            u0s.append(u0s[0])

        if len(grads) == 3:
            color2_list = [colors[0], colors[10], colors[3]]
            alphas = [1.0, 0.5, 1.0]
            linestyles = ["-", "-", ":"]
        elif len(grads) == 2:
            color2_list = [colors[0], colors[3]]
            alphas = [1.0, 1.0]
            linestyles = ["-", ":"]
        else:
            raise ValueError(f"Number of gradients is not supported: {len(grads)}")

        color1 = "black"  # colors[3] # "black" #
        ax1 = axs[0, i]
        ax1.set_xlabel(xlabel)
        ax1.plot(u0s[0], losses[0], color=color1)
        ax1.margins(x=0)

        if np.isclose(factors[i], 1.0):
            ax1.set_title(title_list[i])  # loc='left')
        else:
            ax1.set_title(title_list[i] + f"\n Gradscale: {factors[i]}")  # loc='left')

        if i == 0:
            ax1.set_ylabel(ylabel1, color=color1)
            ax1.tick_params(axis="y", labelcolor=color1)
            ax1.spines[["right", "top"]].set_visible(False)
        else:
            ax1.tick_params(axis="y", colors="white")
            ax1.spines[["left", "right", "top"]].set_visible(False)

        ax2 = axs[0, i].twinx()  # Create a second y-axis sharing the same x-axis
        ax2.set_ylim(ylims)

        for j in range(0, len(grads)):
            color2 = "grey"
            ax2.plot(
                u0s[j],
                grads[j],
                color=color2_list[j],
                alpha=alphas[j],
                linestyle=linestyles[j],
            )
            ax2.axhline(y=0, linestyle=linestyles[j], color="grey", linewidth=1.0)
            ax2.margins(x=0)

        if i == len(u0s_list) - 1:
            ax2.set_ylabel(ylabel2, color=color2)
            ax2.tick_params(axis="y", labelcolor=color2)
            ax2.spines["right"].set_color(color2)
            ax2.spines[["left", "top"]].set_visible(False)
        else:
            ax2.axes.get_yaxis().set_ticks([])
            ax2.spines[["left", "right", "top"]].set_visible(False)

    fig.tight_layout()
    print(f"Saving figure to {figpath}")
    fig.savefig(figpath + ".pdf")
    fig.savefig(figpath + ".png")
    return fig