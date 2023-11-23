"""analyze benchmarks"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np


def load_results() -> dict:
    """load reults from ./results/
    :return: [flags, times]"""

    res = {}
    for fname in pathlib.Path("results").iterdir():
        fname_short = fname.name[6:-4]
        flags, env = fname_short.split("]")
        flags, env = flags[1:], env
        # noinspection PyTypeChecker
        data = np.load(fname)
        if env not in res:
            res[env] = {}
        res[env][flags] = data

    for env in res:
        res[env] = dict(sorted(res[env].items(), key=lambda i: np.mean(i[1])))
        # res[env].sort(key=lambda i: np.mean(i[1]))

    return res


def plot_results():
    """plot the results"""
    results = load_results()
    default_comp = "clang-15"
    N_ax = len(results) + (1 if default_comp in results else 0)
    fig, axs = plt.subplots(N_ax, 1)
    if N_ax == 1:
        axs = [axs]
    colors = ["blue", "red", "green", "orange", "teal", "yellow"]
    for env_i, (env, res_env) in enumerate(results.items()):
        ax = axs[env_i]
        for i, (flags, data) in enumerate(res_env.items()):
            ax.plot(np.ones_like(data) * i, data, alpha=0.005, linestyle="", marker="o", markeredgewidth=0, color=colors[env_i])
            ax.semilogy(i, np.mean(data), linestyle="", marker="o", color="black")
            ax.text(i - 0.3, np.mean(data), flags, rotation="vertical")
        ax.plot([], marker="o", color=colors[env_i], label=env or "default")
        if env_i == 0:
            ax.set_ylabel("execution time")
        ax.set_xticks([])
        ax.set_xlim(-0.5, None)
        ax.legend()

    if default_comp in results:
        ax = axs[-1]
        default_i = None
        for i, flags in enumerate(results[default_comp].keys()):
            is_default = flags.startswith("'DEFAULT")
            if is_default:
                if default_i is None:
                    default_i = i
                i = default_i
            for env_i, (env, res_env) in enumerate(results.items()):
                if flags in res_env:
                    data = res_env[flags]
                    ax.semilogy(i, np.mean(data), linestyle="", marker="o", color=colors[env_i])
            text_pos = np.mean(results[default_comp][flags])
            if is_default:
                if flags.endswith("0'"):
                    flags = "DEFAULT"
                else:
                    continue
            ax.text(i - 0.3, text_pos, flags, rotation="vertical")
        ax.set_xticks([])
        ax.set_title("Aggregated averages")
        ax.set_xlim(-0.5, None)

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_results()
