import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from scipy.integrate import solve_ivp

import src.attractors3d as a3d
from src.position import Vector

plt.style.use("dark_background")


def make_animation(
    name, func, t_max=60, n_points=10000, midpoint=(0, 0, 0), variance=1
):
    dt = 1 / 60  # 60 fps animation
    t_eval = np.arange(0, t_max, dt, dtype=float) + dt
    n_frames = len(t_eval) + 1

    positions = Vector.zero(shape=(n_frames, n_points))
    positions[0] += Vector.random_gauss(shape=(n_points,), scale=variance) + midpoint

    # Generate random colors for each point (HSV space for better visual spread)
    colors = np.random.rand(n_points)

    initial_conditions = positions[0, :].flatten()

    sol = solve_ivp(
        func, (0, t_max), initial_conditions, "RK45", t_eval=t_eval, vectorized=True
    )

    positions[1:, :] = sol.y.reshape(3, -1, n_frames - 1).transpose(2, 1, 0)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    # Remove figure padding
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    points = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        positions[0, :, 2],
        alpha=0.7,
        marker=",",
        s=1,
        c=colors,
        cmap=np.random.choice(
            (
                "Purples",
                "Blues",
                "Greens",
                "Oranges",
                "Reds",
                "YlOrBr",
                "YlOrRd",
                "OrRd",
                "PuRd",
                "RdPu",
                "BuPu",
                "GnBu",
                "PuBu",
            )
        ),
        edgecolors="none",
    )

    ax.set_xlim(positions[:, :, 0].min(), positions[:, :, 0].max())
    ax.set_ylim(positions[:, :, 1].min(), positions[:, :, 1].max())
    ax.set_zlim(positions[:, :, 2].min(), positions[:, :, 2].max())

    def update(frame):
        points._offsets3d = (
            positions[frame, :, 0],
            positions[frame, :, 1],
            positions[frame, :, 2],
        )

        # Rotate view
        ax.view_init(elev=25, azim=frame / 8)

        return [points]

    ani = FuncAnimation(
        fig, update, interval=1000 * dt, frames=n_frames, repeat=False, blit=True
    )

    def progress(frames, max_frames):
        bar.update(1)

    bar = tqdm(total=n_frames, desc=f"Saving animation for {name}", colour="green")
    ani.save(
        f"videos/Animation_{name}.mp4", writer="ffmpeg", progress_callback=progress
    )
    bar.close()


def make_figure(name, func, t_max=300, start_point=(0, 0, 0), ax=None):
    dt = 1 / 180  # high precision points
    t_eval = np.arange(0, t_max, dt, dtype=float) + dt
    n_frames = len(t_eval) + 1

    positions = Vector.zero(shape=(n_frames,))
    positions[0] += start_point

    sol = solve_ivp(func, (0, t_max), positions[0], "RK45", t_eval=t_eval)

    positions[1:, :] = sol.y.T

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_axis_off()
        # Remove figure padding
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    ax.plot(positions.x, positions.y, positions.z, color="white")
    plt.title(f"{name} Attractor")

    return ax


if __name__ == "__main__":
    # make_animation("Lorentz", a3d.lorentz, midpoint=(1, 1, 21), variance=3)
    # make_animation("Thomas", a3d.thomas)
    # make_animation("Langford", a3d.langford)
    # make_animation("Dadras", a3d.dadras, midpoint=(1,1,0))
    # make_animation("Chen_Lee", a3d.chen_lee)
    # make_animation("Lorenz83", a3d.lorentz83)
    # make_animation("Rössler", a3d.rossler)
    # make_animation("Halvorsen", a3d.halvorsen, variance=0.1)
    # # ! rabinovich_fabrikant does not work
    # make_animation("Three Scroll", a3d.three_scroll)
    # make_animation("Sprott", a3d.sprott)
    # make_animation("Sprott Linz", a3d.sprott_linz, variance=0.1,
    #                midpoint=(0.1, 0.1, 0.1))
    # make_animation("Four Wing", a3d.four_wing, variance=0.1)

    # make_figure("Lorentz", a3d.lorentz, start_point=(0.1, 0, 20))
    # plt.savefig("images/Lorentz.png")
    # make_figure("Thomas", a3d.thomas, start_point=(0.1, 0, 0))
    # plt.savefig("images/Thomas.png")
    # make_figure("Langford", a3d.langford, start_point=(0.1, 0, 0))
    # plt.savefig("images/Langford.png")
    # make_figure("Dadras", a3d.dadras, start_point=(1.1, 2.1, -2))
    # plt.savefig("images/Dadras.png")
    # make_figure("Lorentz83", a3d.lorentz83, t_max=60)
    # plt.savefig("images/Lorenz83.png")
    # make_figure("Rössler", a3d.rossler, start_point=(1, 0, 0))
    # plt.savefig("images/Rössler.png")
    # make_figure("Halvorsen", a3d.halvorsen, start_point=(-1.48, -1.51, 2.04))
    # plt.savefig("images/Halvorsen.png")
    # make_figure("Rabinovich Fabrikant", a3d.rabinovich_fabrikant,
    #             start_point=(-1, 0, 0.5))
    # plt.savefig("images/Rabinovich Fabrikant.png")
    # make_figure("Three scroll", a3d.three_scroll,
    #             start_point=(-0.1, -0.1, -0.1), t_max=6)
    # plt.savefig("images/three_scroll.png")
    # make_figure("Sprott", a3d.sprott, start_point=(0.1, 0.1, -0.1), t_max=300)
    # plt.savefig("images/Sprott.png")
    # make_figure("Sprott Linz", a3d.sprott_linz, start_point=(0.1, 0.1, 0.1))
    # plt.savefig("images/Sprott_Linz.png")
    # make_figure("Four_wing", a3d.four_wing, start_point=(0.1, 0.1, 0.1))
    # plt.savefig("images/Four_wing.png")

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.set_axis_off()
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # make_figure("Chen Lee", a3d.chen_lee, start_point=(5, 10, 10), ax=ax)
    # make_figure("Chen Lee", a3d.chen_lee, start_point=(-7, -5, -10), ax=ax)
    # plt.tight_layout()
    # plt.savefig("images/Chen Lee.png")
    print("Process completed")
