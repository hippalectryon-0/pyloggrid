"""Utilities for fast plotting"""
from __future__ import annotations

from pyloggrid.Libs.singlethread_numpy import np

np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring
import typing

import imageio.v2 as imageio
import matplotlib
import scienceplots  # needs to be imported, even if not called directly
from matplotlib import cycler
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Slider

# noinspection PyStatementEffect
scienceplots.stylesheets  # prevent removal of import

if typing.TYPE_CHECKING:
    from typing import Any, Callable

    # noinspection PyPackageRequirements
    from cycler import Cycler
    from matplotlib.colors import LinearSegmentedColormap

    from pyloggrid.LogGrid.Grid import Grid


def rightpad_array_2D(a: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    """Pad a 2D+ array with zeros.

    The padding is done at the large coordinate edge (it is not centered).

    Args:
        a: array to pad
        Nx: amount of pixels to pad in first dimension
        Ny: amount of pixels to pad in second dimension
    """
    b = np.zeros((a.shape[0] + Nx, a.shape[1] + Ny) + a.shape[2:]).astype(a.dtype)
    b[: a.shape[0], : a.shape[1]] = a
    return b


def plot2axes() -> tuple[Figure, Any, Any]:
    """Create a plot with left and right axis.

    Axes are set to `Any` since the real class is hidden in ``mpl.axes._subplots``.

    Returns:
        ``figure, ax1, ax2``
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    return fig, ax1, ax2


def plot2axesX() -> tuple[Figure, Any, Any]:
    """plot with bottom and top axis.

    Axes are set to `Any` since the real class is hidden in ``mpl.axes._subplots``.
    """
    fig, ax1 = plt.subplots()
    ax2 = ax1.twiny()
    return fig, ax1, ax2


def labels(xlabel: str = "", ylabel: str = "", title: str = "", ax: Any = None) -> None:
    """Add labels and title to an axis.

    Args:
        xlabel
        ylabel
        title
        ax: if specified, the axis on which to add the labels; defaults to the current axis
    """
    ax = ax or plt.gca()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def scatter(ax=None, *args, **kwargs) -> Any:
    """Plot a scatter plot using ``plot()`` instead of ``scatter()``.

    (not handled the same way, e.g. zindex)

    Args:
        ax: the axis to use, defaults to
    """
    ax = ax or plt.gca()
    if "linestyle" not in kwargs:
        kwargs["linestyle"] = ""
    return ax.plot(*args, **kwargs)


def pltshowm(full: bool = True, save: str = None, legend: bool = True, compact: bool = True, tight: bool = False) -> None:
    """Display the pyplot figure maximized.

    Args:
        legend: if ``False``, do not add a legend
        save: if specified, save path for the figure
        full: if ``False``, doe not go fullscreen
        compact: if ``True``, use ``plt.tight_layout``
        tight: if ``True``, autoscale with tight boundaries
    """
    if full:
        man = plt.get_current_fig_manager()
        try:
            man.frame.Maximize(True)
        except AttributeError:  # wrong backend, try QT
            try:
                man.window.showMaximized()
            except AttributeError:  # probably running a backend that can only save the image, not display it
                matplotlib.pyplot.gcf().set_size_inches(19.2, 10.8)
    if legend:
        plt.legend()
    if tight:
        plt.autoscale(tight=True)
    if compact:
        plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")
    plt.show()


def initFormat(size: float = 1, txtsize: float = None) -> None:
    """Sets up pyplot formatting for scientific publications.

    Args:
        size: the figure's size relative to the default one
        txtsize: scaling for the figure's fonts, if not set defaults to ``size``
    """

    def force_add_cycler(cyclers: list["Cycler"]) -> "Cycler":
        """add two cyclers, whatever their length"""
        if len(cyclers) == 2:
            cyc_a, cyc_b = cyclers
            try:
                return cyc_a + cyc_b
            except ValueError:
                return cyc_a * len(cyc_b) + cyc_b * len(cyc_a)
        else:
            return force_add_cycler([cyclers[0], force_add_cycler(cyclers[1:])])

    if txtsize is None:
        txtsize = size

    plt.style.use(["science", "grid"])
    plt_params = {
        "figure.figsize": (8 * size, 6 * size),
        "font.size": 22 * txtsize,
        "axes.labelsize": 20 * txtsize,
        "axes.titlesize": 20 * txtsize,
        "legend.fontsize": 16 * txtsize,
        "xtick.labelsize": 16 * txtsize,
        "ytick.labelsize": 16 * txtsize,
        "lines.markersize": 6 * size,
        "lines.linewidth": 2 * size,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        "legend.framealpha": 0.7,
        "legend.fancybox": False,
        "axes.prop_cycle": force_add_cycler(
            [
                cycler("color", ["#006BA4", "#FF800E", "#ABABAB", "#595959", "#5F9ED1", "#C85200", "#898989", "#A2C8EC", "#FFBC79", "#CFCFCF"]),
                cycler("linestyle", ["-", "--", "-.", ":"]),
                cycler("marker", ["o", "d", "v", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D", "|", "_"]),
            ]
        ),
        "image.cmap": "cividis",
    }
    matplotlib.rcParams.update(plt_params)


def rand_cmap(
    nlabels: int, type_: typing.Literal["bright", "soft"] = "bright", first_color_black: bool = True, last_color_black: bool = False
) -> "LinearSegmentedColormap":
    """Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks

    Args:
        nlabels: Number of labels (size of colormap)
        type_: 'bright' for strong colors, 'soft' for pastel colors
        first_color_black: Option to use first color as black
        last_color_black: Option to use last color as black

    Returns:
        colormap for matplotlib
    """
    import colorsys

    from pyloggrid.Libs.singlethread_numpy import np

    np.zeros(0)  # Do not remove - used to force singlethread_numpy to be imported before the other libraries when refactoring

    from matplotlib.colors import LinearSegmentedColormap

    if type_ not in ("bright", "soft"):
        raise ValueError('Please choose "bright" or "soft" for type')

    # Generate color map for bright colors, based on hsv
    if type_ == "bright":
        randHSVcolors = [(np.random.uniform(low=0.0, high=1), np.random.uniform(low=0.2, high=1), np.random.uniform(low=0.9, high=1)) for _ in range(nlabels)]

        randRGBcolors = [colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]) for HSVcolor in randHSVcolors]
        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list("new_map", randRGBcolors, N=nlabels)

    elif type_ == "soft":
        low = 0.6
        high = 0.95
        randRGBcolors: list = [
            (np.random.uniform(low=low, high=high), np.random.uniform(low=low, high=high), np.random.uniform(low=low, high=high)) for _ in range(nlabels)
        ]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list("new_map", randRGBcolors, N=nlabels)
    else:
        raise ValueError

    return random_colormap


def save_animation_from_slider(slider: Slider, save: str, temp_dir: str, N: int = 100) -> None:
    """Save a matplotlib graph with a slider as an animation.

    Args:
        slider: the slider object
        save: the output file name
        temp_dir: where to store the intermediate images
        N: number of frames
    """

    maxw, maxh = 0, 0
    for n in np.linspace(slider.valmin, slider.valmax, N):
        slider.set_val(n)
        plt.savefig(f"{temp_dir}/{n}.png")
        img = imageio.imread(f"{temp_dir}/{n}.png")
        w, h = img.shape[:-1]
        maxw, maxh = max(w, maxw), max(h, maxh)

    # noinspection PyTypeChecker
    with imageio.get_writer(save, format="FFMPEG", mode="I", fps=10, output_params=["-vf", "scale=500:500", "-sws_flags", "neighbor"]) as writer:
        for n in np.linspace(slider.valmin, slider.valmax, N):
            img = imageio.imread(f"{temp_dir}/{n}.png")
            w, h = img.shape[:-1]
            img = rightpad_array_2D(img, maxw - w, maxh - h)
            writer.append_data(img.astype(np.uint8))


def enable_slider_save(f: Callable) -> Callable:
    """Decorator to save an animation from a slider graph.

    This adds two arguments to the decorated function, ``save`` and ``temp_path``, which if supplied are forwarded to :func:`save_animation_from_slider`.
    """

    def wrapper(*args, save: str = None, temp_path: str = None, **kwargs):
        """:param save if not None, save an animation of the slider as mp4 to a file named <save>
        :param temp_path if save is not None, where to store generated images
        """
        slider = f(*args, **kwargs)
        if save:
            save_animation_from_slider(slider, save, temp_path)
        return slider

    return wrapper


@enable_slider_save
def interactive_3D_logplot_by_z(X: np.ndarray, Y: np.ndarray, V: np.ndarray, Z: np.ndarray = None) -> tuple[Any, Any]:
    """An interactive 3D logplot, sliced by ``Z`` if given.

    Warning:
        As for all interactive functions, the returned val must be assigned, ex via ``_ = my_interactive_function(...)``

    You can either provide 3D arrays (X, Y, V, Z[:,:,0]) or 2D arrays (X, Y, V).

    Args:
        X
        Y
        V: values
        Z: optional 1D Z axis
    """
    import matplotlib.ticker as mticker
    from matplotlib.widgets import Button, Slider

    # Set the loglog scale for the x, y, and z axes
    # noinspection PyMissingOrEmptyDocstring
    def asinh(data, a0: 1):
        return np.arcsinh(data / a0) * a0

    # noinspection PyMissingOrEmptyDocstring
    def sinh(data, a0: 1):
        return np.sinh(data / a0) * a0

    ## animation code
    factk, fact0, z_mode = None, None, Z is None

    # noinspection PyMissingOrEmptyDocstring
    def get_toplot(doall=False):
        if doall:
            toplot_ = V
        elif z_mode:
            toplot_ = V[int(n_slider.val) - 1]
        else:
            toplot_ = V[int(n_slider.val) - 1, :, :, int(z_slider.val)]
        return toplot_

    # noinspection PyMissingOrEmptyDocstring
    def update_graph(_=None):
        nonlocal factk, fact0
        n_z = int(z_slider.val) if Z is not None else None
        toplot_ = get_toplot()
        X_, Y_ = (X, Y) if z_mode else (X[:, :, n_z], Y[:, :, n_z])

        graph._offsets3d = np.array([asinh(X_.flatten(), factk), asinh(Y_.flatten(), factk), asinh(toplot_.real.flatten(), fact0)])  # position
        if z_mode and Z is not None:
            sel = np.zeros_like(toplot_.real)
            sel[:] = -1
            sel[:, :, n_z] = 1
            graph.set_array(sel.flatten())
            sizes = np.zeros_like(toplot_.real)
            sizes[:] = 10
            sizes[:, :, n_z] = 30
            graph.set_sizes(sizes.flatten())
        else:
            graph.set_array(toplot_.imag.flatten() / np.abs(toplot_.flatten()))
            graph.set_sizes(np.ones_like(toplot_.real.flatten()) * 10)

        if Z is not None:
            z_slider_text.set_text(f"z={Z[n_z]:.2e}")

        fig.canvas.draw_idle()

    def rescale(_=None, doall=False):
        """rescale the graph"""
        nonlocal factk, fact0
        toplot_ = get_toplot(doall)

        factk = np.min(np.abs(X)) / 10
        fact0 = max(1e-250, np.min(np.abs(toplot_.real[toplot_.real != 0])) / 10)

        ax.set_xscale("asinh", linear_width=factk)  # just for the lines, before setting the ticks
        ax.set_yscale("asinh", linear_width=factk)
        ax.set_zscale("asinh", linear_width=fact0)

        ax.set_xlim(np.min(asinh(X.flatten(), factk)), np.max(asinh(X.flatten(), factk)))
        ax.set_ylim(np.min(asinh(Y.flatten(), factk)), np.max(asinh(Y.flatten(), factk)))
        ax.set_zlim(np.min(asinh(toplot_.real.flatten(), fact0)), np.max(asinh(toplot_.real.flatten(), fact0)))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{sinh(val, factk):.2e}"))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{sinh(val, factk):.2e}"))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{sinh(val, fact0):.2e}"))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.zaxis.set_major_locator(mticker.AsinhLocator(linear_width=fact0, subs=(2, 5), symthresh=0.0, numticks=5))

        update_graph()

    # noinspection PyMissingOrEmptyDocstring
    def toggle_z(_=None):
        nonlocal z_mode
        z_mode = not z_mode
        update_graph()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ## Sliders & Buttons
    # fig.subplots_adjust(bottom=-0.1)
    n_slider = Slider(
        ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]),
        label="Step",
        valfmt="%d",
        valmin=1,
        valmax=V.shape[0],
        valinit=1,
        valstep=1,
    )
    n_slider.on_changed(update_graph)
    z_slider, btn_toggle_z = None, None
    if Z is not None:
        z_slider_ax = fig.add_axes([0.1, 0.25, 0.0225, 0.63])
        z_slider = Slider(ax=z_slider_ax, label="n_z", valfmt="%d", valmin=0, valmax=X.shape[2] - 1, valinit=X.shape[2] // 2, valstep=1, orientation="vertical")
        z_slider_text = z_slider_ax.text(0.5, 0.5, "")
        z_slider.on_changed(update_graph)

    btn_rescale = Button(fig.add_axes([0.8, 0.025, 0.1, 0.04]), "Rescale", hovercolor="0.975")
    btn_rescale.on_clicked(rescale)
    if Z is not None:
        btn_toggle_z = Button(fig.add_axes([0.6, 0.025, 0.1, 0.04]), "Toggle z mode", hovercolor="0.975")
        btn_toggle_z.on_clicked(toggle_z)

    graph = ax.scatter([], [], [], label=r"$\Re ux$", alpha=0.5)
    ax.set_xlabel("$X$")
    ax.set_ylabel("$Y$")
    rescale(doall=True)
    graph.set_cmap("coolwarm")

    return ax, (graph, z_slider, n_slider, btn_rescale, btn_toggle_z)


@enable_slider_save
def interactive_spectrum(ts: np.ndarray, ks: np.ndarray, spectra: dict[str, np.ndarray], quantities: dict[str, np.ndarray]) -> tuple[Slider, Any, Any]:
    r"""An interactive spectrum + arbitrary time-dependant quantities.

    Warning:
        As for all interactive functions, the returned val must be assigned, ex via ``_ = my_interactive_function(...)``

    LHS: spectrum vs k
    RHS: quantity vs t
    current time is changed via a slider

    Args:
        ts: time array
        ks: ks array for each time
        spectra: dict label->array
        quantities: dict label->quantity

    Todo:
        Add an example of usage ?
    """

    fig, axs = plt.subplot_mosaic("AB")
    ax1, ax2 = axs["A"], axs["B"]

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    padfact = 2
    ax1.set_xlim(min(np.min(i) for i in ks) / padfact, max(np.max(i) for i in ks) * padfact)
    ax1.set_ylim(1e-200, max(np.max(i) for i in list(spectra.values())[0]) * padfact * 1e5)
    ax1.set_xlabel("$k$")
    ax1.set_ylabel("spectra")

    # noinspection PyMissingOrEmptyDocstring
    def update_graph(_=None):
        i = int(n_slider.val - 1)
        t = ts[i]
        vl.set_xdata([t, t])
        for k, l in spectrum_lines.items():
            l.set_data(ks[i], spectra[k][i])

    # Sliders & Buttons
    fig.subplots_adjust(bottom=0.2)
    n_slider = Slider(ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]), label="Step", valfmt="%d", valmin=1, valmax=ts.size, valinit=1, valstep=1)
    n_slider.on_changed(update_graph)

    spectrum_lines = {}
    for k in spectra:
        spectrum_lines[k] = scatter(ax1, [0], [1], label=k)[0]
    ax1.legend()

    for k, v in quantities.items():
        scatter(ax2, ts, v, label=k)
    ax2.legend()
    vl = ax2.axvline(ts[0], ls="-", color="r", lw=1, zorder=10)
    ax2.set_xlabel("$t$")

    update_graph()
    return n_slider, ax1, ax2


@enable_slider_save
def interactive_grid_imshow(
    grid: Grid,
    update_data: Callable[[Any, float], np.ndarray],
    slider_params: tuple[float | None, float | None, float | None, str | None, str | None] = (0, 1, None, "", None),
) -> Slider:
    """Imshow with slider, with axes formatted by the grid.

    Warning:
        As for all interactive functions, the returned val must be assigned, ex via ``_ = my_interactive_function(...)``

    Args:
        grid
        update_data: a function that takes the plotting ax and the slider's value, and returns an array to plot of size (N, N)
        slider_params: (vmin, vmax, vstep, title, vformat), all optional
    """
    fig, ax = plt.subplots()
    N = grid.N_points

    # noinspection PyMissingOrEmptyDocstring
    def update_graph(_=None):
        im.set_array(update_data(ax, n_slider.val))
        im.autoscale()

    im = ax.imshow(np.zeros((N, N)), origin="lower", interpolation="none")
    ticks = np.linspace(0, N, 5)
    ax.set_xlabel("$k_2$")
    ax.set_ylabel("$k_1$")
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"$10^{{{int(np.log10(i))}}}$" for i in [grid.k_min * grid.l**j for j in ticks]])
    ax.set_yticklabels([f"$10^{{{int(np.log10(i))}}}$" for i in [grid.k_min * grid.l**j for j in ticks]])
    plt.colorbar(im)

    ## Sliders & Buttons
    fig.subplots_adjust(bottom=0.2)
    vmin, vmax, vstep, slidertitle, vformat = slider_params
    n_slider = Slider(ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]), label=slidertitle, valfmt=vformat, valmin=vmin, valmax=vmax, valinit=vmin, valstep=vstep)
    n_slider.on_changed(update_graph)

    update_graph()

    return n_slider


@enable_slider_save
def interactive_grid_3Dslice(
    grid: Grid,
    update_data: Callable[[Any, float], np.ndarray],
    slider_params: tuple[float | None, float | None, float | None, str | None, str | None] = (0, 1, None, "", None),
) -> Slider:
    """Interactive 3D slice of a grid field.

    Warning:
        As for all interactive functions, the returned val must be assigned, ex via ``_ = my_interactive_function(...)``

    Args:
        grid
        update_data: a function that takes the plotting ax and the slider's value, and returns a 3D array to plot
        slider_params: (vmin, vmax, vstep, title, vformat), all optional
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    cmap = matplotlib.colormaps.get("gist_rainbow")
    N = grid.N_points

    # noinspection PyMissingOrEmptyDocstring
    def update_graph(_=None):
        data = update_data(ax, n_slider.val)

        ax.cla()
        kxy = range(N)
        kz = range(N)
        KXY, KZ = np.meshgrid(kxy, kz)
        # primary plane
        data_plane1 = np.array([[data[i, N + i, N + j] for j in kz] for i in kxy])
        data_plane1 = data_plane1 / np.max(data_plane1)
        ax.plot_surface(KXY, KXY, KZ, facecolors=cmap(data_plane1), alpha=0.95, linewidth=0)
        # secondary plane
        data_plane2 = np.array([[data[j, N + i, N + i] for j in kz] for i in kxy])
        data_plane2 = data_plane2 / np.max(data_plane2)
        ax.plot_surface(KZ, KXY, KXY, facecolors=cmap(data_plane2), alpha=0.55, linewidth=0)

    ax.set_xlabel("$k_x$")
    ax.set_ylabel("$k_y$")
    ax.set_zlabel("$k_z$")
    m = matplotlib.cm.ScalarMappable(cmap=cmap)
    m.set_array([])
    plt.colorbar(m, ax=ax)

    ## Sliders & Buttons
    fig.subplots_adjust(bottom=0.2)
    vmin, vmax, vstep, slidertitle, vformat = slider_params
    n_slider = Slider(ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]), label=slidertitle, valfmt=vformat, valmin=vmin, valmax=vmax, valinit=vmin, valstep=vstep)
    n_slider.on_changed(update_graph)

    update_graph()

    return n_slider


@enable_slider_save
def interactive_3D_logplot_positive(
    update_data: Callable[[Any, float], np.ndarray],
    slider_params: tuple[float | None, float | None, float | None, str | None, str | None] = (0, 1, None, "", None),
    threshold_fact: float = 5,
) -> Slider:
    """3D logplot of scalar field on kx,ky,kz>0.

    Warning:
        As for all interactive functions, the returned val must be assigned, ex via ``_ = my_interactive_function(...)``

    Args:
        update_data: a function that takes the plotting ax and the slider's value, and returns a 3D array to plot
        slider_params: (vmin, vmax, vstep, title, vformat), all optional
        threshold_fact: which points to hide, relative to max(data)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set(xlabel="$k_x$")
    ax.set(ylabel="$k_y$")
    ax.set(zlabel="$k_z$")
    graph = ax.scatter([], [], [], s=50, cmap="bwr", vmin=0, vmax=1, alpha=0.5)
    graph.set_cmap("bwr")

    # noinspection PyMissingOrEmptyDocstring
    def update_graph(_=None):
        data = update_data(ax, n_slider.val)
        N_points = data.shape[0]
        ax.set(xlim3d=(0, N_points - 1))
        ax.set(ylim3d=(0, N_points - 1))
        ax.set(zlim3d=(0, N_points - 1))

        wmax = np.max(np.abs(data))
        threshold = wmax / threshold_fact
        data = data[:, N_points:, N_points:]
        mask = np.abs(data) >= threshold
        relevant = np.argwhere(mask)
        data_c = (data[mask] - threshold) / (wmax - threshold)

        graph._offsets3d = np.array([relevant[:, 0], relevant[:, 1], relevant[:, 2]])
        graph.set_array(data_c)

    ## Sliders & Buttons
    fig.subplots_adjust(bottom=0.2)
    vmin, vmax, vstep, slidertitle, vformat = slider_params
    n_slider = Slider(ax=fig.add_axes([0.25, 0.1, 0.65, 0.03]), label=slidertitle, valfmt=vformat, valmin=vmin, valmax=vmax, valinit=vmin, valstep=vstep)
    n_slider.on_changed(update_graph)

    update_graph()

    return n_slider


initFormat()
