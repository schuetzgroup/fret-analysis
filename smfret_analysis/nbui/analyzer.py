# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provide :py:class:`Analyzer` as a Jupyter notebook UI for smFRET analysis"""
from io import BytesIO
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union

import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from sdt import changepoint, nbui

from .. import base


class BaseAnalyzerNbUI:
    """Jupyter notebook UI for analysis of smFRET tracks

    Analyze and filter smFRET tracks produced by :py:class:`Tracker`.
    """

    def __init__(self):
        super().__init__()

        self._segment_fig = None
        self._brightness_fig = None
        self._thresholder = None
        self._beam_shape_fig = None
        self._beam_shape_artists = [None, None]
        self._population_fig = None
        self._eff_stoi_fig = None

    def find_beam_shape_thresh(self) -> ipywidgets.Widget:
        """Display a widget to set the laser intensity threshold

        Use :py:meth:`filter_beam_shape_region` to select only datapoints
        within the region of bright laser excitation.
        """
        if self._beam_shape_fig is None:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            fig.add_subplot(1, 2, 2)

            self._beam_shape_fig = fig

        ax = self._beam_shape_fig.axes

        channel_sel = ipywidgets.Dropdown(options=list(self.flatfield),
                                          description="channel")
        thresh_sel = ipywidgets.BoundedIntText(value=50, min=0, max=100,
                                               description="threshold")

        def channel_changed(change=None):
            bs = self.flatfield[channel_sel.value]
            if self._beam_shape_artists[0] is not None:
                self._beam_shape_artists[0].remove()
            self._beam_shape_artists[0] = ax[0].imshow(bs.corr_img)
            update()

        def update(change=None):
            bs = self.flatfield[channel_sel.value]
            if self._beam_shape_artists[1] is not None:
                self._beam_shape_artists[1].remove()
            self._beam_shape_artists[1] = ax[1].imshow(
                bs.corr_img * 100 > thresh_sel.value)
            self._beam_shape_fig.tight_layout()
            self._beam_shape_fig.canvas.draw()

        channel_sel.observe(channel_changed, "value")
        thresh_sel.observe(update, "value")

        channel_changed()

        return ipywidgets.VBox([channel_sel, thresh_sel,
                                self._beam_shape_fig.canvas])

    def _make_dataset_selector(self, state: Dict,
                               callback: Callable[[Optional[Any]], None],
                               show_file_selector: bool = True
                               ) -> ipywidgets.HBox:
        """Create a drop-down for selecting the dataset in UIs

        Parameters
        ----------
        state
            Keep some state of the widget. This creates/overwrites "tr",
            "pnos", and "files" entries containing the dataset's tracking data,
            unique particle numbers, and source files, respectively.
        callback
            Gets called whenever the dataset was changed.

        Returns
        -------
        The selection UI element
        """
        d_sel = ipywidgets.Dropdown(options=list(self.sm_data.keys()),
                                    description="dataset")
        f_sel = ipywidgets.Dropdown(description="file")

        def change_dataset(change=None):
            src = getattr(self, "sources", {}).get(d_sel.value, {})
            fids = {src.get(i, i): i
                    for i in sorted(self.sm_data[d_sel.value].keys())}
            f_sel.options = fids
            f_sel.layout = ipywidgets.Layout(
                display=None if show_file_selector else "none")
            change_file()

        def change_file(change=None):
            trc = self._apply_filters(
                self.sm_data[d_sel.value][f_sel.value]["donor"])
            nan_mask = (np.isfinite(trc["fret", "a_mass"]) &
                        np.isfinite(trc["fret", "d_mass"]))
            all_p = trc["fret", "particle"].unique()
            bad_p = trc.loc[~nan_mask, ("fret", "particle")].unique()
            state["id"] = (d_sel.value, f_sel.value)
            state["trc"] = trc
            state["pnos"] = np.setdiff1d(all_p, bad_p)

            callback()

        d_sel.observe(change_dataset, names="value")
        f_sel.observe(change_file, names="value")
        change_dataset()

        return ipywidgets.HBox([d_sel, f_sel])

    def find_subpopulations(self) -> ipywidgets.Widget:
        """Interactively select component for calculation of excitation eff.

        The excitation efficiency factor is computer from data with known 1:1
        stoichiomtry. This displays datasets and allows for fitting Gaussian
        mixture models to select the right component in an E–S plot. Parameters
        can be passed to :py:meth:`calc_excitation_eff`.

        Returns
        -------
        UI element

        See also
        --------
        calc_excitation_eff
        """
        if self._population_fig is None:
            self._population_fig = plt.subplots()[0]

        self._population_fig.axes[0].cla()
        state = {"artists": []}

        n_comp_sel = ipywidgets.IntText(description="components", value=1)

        def update(change=None):
            d = pd.concat([self._apply_filters(v["donor"])
                           for v in self.sm_data[state["id"][0]].values()],
                          ignore_index=True)
            # _excitation_eff_filter is defined in base.BaseAnalyzer child
            # class to select data suitable for excitation efficiency
            # calculation
            d = d[self._excitation_eff_filter(d) &
                  np.isfinite(d["fret", "eff_app"]) &
                  np.isfinite(d["fret", "stoi_app"])].copy()
            labels = base.gaussian_mixture_split(d, n_comp_sel.value)[0]

            ax = self._population_fig.axes[0]

            for ar in state["artists"]:
                ar.remove()
            state["artists"] = []

            for lab in range(n_comp_sel.value):
                dl = d[labels == lab]
                sel = (np.isfinite(dl["fret", "eff_app"]) &
                       np.isfinite(dl["fret", "stoi_app"]))
                ar = ax.scatter(dl.loc[sel, ("fret", "eff_app")],
                                dl.loc[sel, ("fret", "stoi_app")],
                                marker=".", label=str(lab), alpha=0.6,
                                color=f"C{lab%10}")
                state["artists"].append(ar)
            ax.legend(loc=0, title="index")
            ax.set_xlabel("apparent eff.")
            ax.set_ylabel("apparent stoi.")
            self._population_fig.canvas.draw()

        d_sel = self._make_dataset_selector(state, update,
                                            show_file_selector=False)
        n_comp_sel.observe(update, "value")

        return ipywidgets.VBox([d_sel, n_comp_sel,
                                self._population_fig.canvas])

    def find_segmentation_options(self) -> nbui.Thresholder:
        """UI for finding parameters for segmenting images

        Parameters can be passed to :py:meth:`apply_segmentation`.

        Returns
        -------
        Widget instance.
        """
        if self._thresholder is None:
            self._thresholder = nbui.Thresholder()

        self._thresholder.image_selector.images = {
            self.sources[did][fid]: v[0]
            for did, dset in self.segment_images.items()
            for fid, v in dset.items()}

        return self._thresholder


class IntramolecularAnalyzer(base.IntramolecularAnalyzer, BaseAnalyzerNbUI):
    def find_changepoint_options(self) -> ipywidgets.Widget:
        """Find options for :py:meth:`segment_mass`

        Returns
        -------
        The selection UI element
        """
        state = {}

        p_sel = ipywidgets.IntText(value=0, description="particle")
        p_label = ipywidgets.Label()
        pen_a_sel = ipywidgets.FloatText(value=2e7, description="acc. penalty")
        pen_d_sel = ipywidgets.FloatText(value=2e7, description="don. penalty")

        if self._segment_fig is None:
            self._segment_fig, ax = plt.subplots(1, 3, figsize=(8, 4))
            ax[1].twinx()

        ax_dm, ax_am, ax_eff, ax_hn = self._segment_fig.axes

        def show_track(change=None):
            for a in self._segment_fig.axes:
                a.cla()

            pi = state["pnos"][p_sel.value]
            p_label.value = f"Real particle number: {pi}"
            trc = state["trc"]
            t = trc[trc["fret", "particle"] == pi]

            fd = t["fret", "frame"].values
            dm = t["fret", "d_mass"].values
            am = t["fret", "a_mass"].values
            ef = t["fret", "eff_app"].values
            hn = t["fret", "has_neighbor"].values

            cp_a = self.cp_detector.find_changepoints(am, pen_a_sel.value) - 1
            cp_d = self.cp_detector.find_changepoints(dm, pen_d_sel.value) - 1
            cp_d_all = np.concatenate([np.searchsorted(fd, fd[cp_a]) - 1,
                                       cp_d])
            cp_d_all = np.unique(cp_d_all)
            cp_d_all.sort()

            if len(fd):
                changepoint.plot_changepoints(dm, cp_d, time=fd, ax=ax_dm)
                changepoint.plot_changepoints(ef, cp_d_all, time=fd, ax=ax_eff)
                changepoint.plot_changepoints(am, cp_a, time=fd, ax=ax_am)
            ax_hn.plot(fd, hn, c="C2", alpha=0.2)

            ax_dm.relim(visible_only=True)
            ax_am.relim(visible_only=True)
            ax_eff.set_ylim(-0.05, 1.05)
            ax_hn.set_ylim(-0.05, 1.05)

            ax_dm.set_title("donor ex.")
            ax_am.set_title("acceptor ex.")
            ax_eff.set_title("app. eff")

            self._segment_fig.tight_layout()
            self._segment_fig.canvas.draw()

        p_sel.observe(show_track, "value")
        pen_d_sel.observe(show_track, "value")
        pen_a_sel.observe(show_track, "value")

        d_sel = self._make_dataset_selector(state, show_track)

        return ipywidgets.VBox([d_sel, p_sel, pen_d_sel, pen_a_sel,
                                self._segment_fig.canvas, p_label])


class IntermolecularAnalyzer(base.IntermolecularAnalyzer, BaseAnalyzerNbUI):
    pass


class DensityPlots(ipywidgets.Box):
    """A widget for displaying density plots

    With this, one can plot data after individual filtering steps to see
    the progress
    """

    def __init__(self, analyzer: base.BaseAnalyzer, datasets: Sequence[str],
                 columns: Sequence[Sequence[str]] = [("fret", "eff_app"),
                                                     ("fret", "stoi_app")],
                 bounds: Sequence[Sequence[float]] = [(-0.5, 1.5),
                                                      (-0.5, 1.5)],
                 direction: Literal["horizontal", "vertical"] = "vertical",
                 **kwargs):
        """Parameters
        ----------
        analyzer
            :py:class:`Analyzer` instance to get data from.
        datasets
            Which datasets to plot
        columns:
            Dataframe column names to plot
        bounds
            Plot only data within bounds
        direction
            In which direction to add new plots. Does not work (yet).
        **kwargs
            Passed to :py:class:`ipywidgets.Box` constructor.
        """
        super().__init__(**kwargs)
        la = self.layout or ipywidgets.Layout()
        la.overflow = la.overflow or "auto"
        la.display = la.display or "block"
        if direction.startswith("h"):
            la.flex_flow = "row nowrap"
            la.width = la.width or "100%"
        else:
            la.flex_flow = "column nowrap"
            la.height = la.height or "100%"
        self.layout = la

        self._ana = analyzer
        self._datasets = datasets
        self._cols = columns

        self._bounds = bounds

    @staticmethod
    def _fmt_axis_label(label: Union[Any, Sequence[Any]]) -> str:
        """Format (multi-index) column name for axis label display

        Parameters
        ----------
        label
            Single- or multi-index column name

        Returns
        -------
        Comma-separated parts of the name
        """
        if not isinstance(label, (tuple, list)):
            label = (label,)
        return ", ".join(str(la) for la in label)

    def _do_plot(self, title: str, plot_func: Callable):
        """Draw a plot

        This will get the current data from the Analyzer and create a plot,
        which is appended to the Box widget.

        Parameters
        ----------
        title
            Title of the plot
        plot_func
            Uses a :py:class:`matplotlib.axes.Axes` (first argument) to plot
            x–y data (second and third args).
        """
        n_plots = len(self._datasets)
        fig = plt.Figure(constrained_layout=True, figsize=(4 * n_plots, 4))
        grid = fig.add_gridspec(1, n_plots)
        ax = []
        for i, g in enumerate(grid):
            ax.append(fig.add_subplot(g, sharex=None if not ax else ax[0],
                                      sharey=None if not ax else ax[0]))

        for d, a in zip(self._datasets, ax):
            t = pd.concat([self._ana._apply_filters(v["donor"])
                           for v in self._ana.sm_data[d].values()],
                          ignore_index=True)
            x = t[self._cols[0]].to_numpy()
            y = t[self._cols[1]].to_numpy()

            fin = np.isfinite(x) & np.isfinite(y)
            x = x[fin]
            y = y[fin]

            bounds = ((self._bounds[0][0] < x) & (x < self._bounds[0][1]) &
                      (self._bounds[1][0] < y) & (y < self._bounds[1][1]))
            x = x[bounds]
            y = y[bounds]

            plot_func(a, x, y)

            a.set_xlabel(self._fmt_axis_label(self._cols[0]))
            a.set_ylabel(self._fmt_axis_label(self._cols[1]))
            a.set_title(d)

        for a in ax[1:]:
            a.tick_params(left=False, labelleft=False)

        fig.suptitle(title)
        bio = BytesIO()
        fig.savefig(bio, format="png")

        self.children = [*self.children,
                         ipywidgets.Image(value=bio.getvalue())]

    def dscatter(self, title: Optional[str] = None):
        """Scatter plot with datapoints colored according to density

        Parameters
        ----------
        title
            Plot supertitle
        """
        self._do_plot(title, self._do_scatter)

    def contourf(self, title: Optional[str] = None, n_levels: int = 15):
        """Contour plot contours according to density

        Parameters
        ----------
        title
            Plot supertitle
        n_levels
            Number of contour levels
        """
        self._do_plot(title,
                      lambda a, x, y: self._do_contour(a, x, y, n_levels))

    def colormesh(self, title: Optional[str] = None):
        """Colored plot of density

        Parameters
        ----------
        title
            Plot supertitle
        """
        self._do_plot(title, self._do_mesh)

    @staticmethod
    def _calc_kde(kde_x: np.ndarray, kde_y: np.ndarray, x: np.ndarray,
                  y: np.ndarray, n_samp: int = 1000,
                  random_state: Optional[np.random.RandomState] = None):
        """Do a Gaussian KDE on data

        Density estimate is calculated from a subset of the data.

        Parameters
        ----------
        kde_x, kde_y
            x and y values to use for KDE
        x, y
            x and y values to evaluate KDE at
        n_samp
            Number datapoints to use from kde_x and kde_y
        random_state
            This instance is used to draw `n_samp` datapoints from ``kde_x``
            and ``kde_y``. If `None`, create a new instance.

        Returns
        -------
        Density estimate evaluated at each point (x[i], y[i]).
        """
        if random_state is None:
            random_state = np.random.RandomState()
        if n_samp < len(kde_x):
            idx = random_state.choice(len(kde_x), replace=False, size=n_samp)
        else:
            idx = slice(None, None)
        kde = scipy.stats.gaussian_kde(np.array([kde_x[idx], kde_y[idx]]))

        dens = np.empty(len(x), dtype=float)
        max_data = 100000  # kde crashes if too many data at once
        for s in range(0, len(x), max_data):
            e = s + max_data
            dens[s:e] = kde(np.array([x[s:e], y[s:e]]))
        return dens

    def _do_scatter(self, a: mpl.axes.Axes, x: np.ndarray, y: np.ndarray):
        """Create density scatter plot

        Parameters
        ----------
        a
            Axes to draw on
        x, y
            data
        """
        dens = self._calc_kde(x, y, x, y)
        sort_idx = np.argsort(dens)
        dens = dens[sort_idx]
        x = x[sort_idx]
        y = y[sort_idx]

        a.scatter(x, y, c=dens, cmap="viridis", marker=".")

    def _do_contour(self, a: mpl.axes.Axes, x: np.ndarray, y: np.ndarray,
                    n_levels: int):
        """Create density contour plot

        Parameters
        ----------
        a
            Axes to draw on
        x, y
            data
        n_levels
            Number of contour levels
        """
        contour_grid = np.array(
            np.meshgrid(np.linspace(*self._bounds[0], 100),
                        np.linspace(*self._bounds[1], 100)))
        contour_grid_flat = np.reshape(contour_grid, (2, -1))
        a.contourf(*contour_grid,
                   self._calc_kde(x, y, *contour_grid_flat).reshape(
                      contour_grid.shape[1:]),
                   cmap=mpl.cm.get_cmap("viridis", n_levels),
                   levels=n_levels)

    def _do_mesh(self, a: mpl.axes.Axes, x: np.ndarray, y: np.ndarray):
        """Create density plot

        Parameters
        ----------
        a
            Axes to draw on
        x, y
            data
        """
        grid = np.array(
            np.meshgrid(np.linspace(*self._bounds[0], 200),
                        np.linspace(*self._bounds[1], 200)))
        grid_flat = np.reshape(grid, (2, -1))
        c = self._calc_kde(x, y, *grid_flat).reshape(grid.shape[1:])
        a.pcolormesh(*grid, c)
