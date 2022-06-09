# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import re
from typing import Any, Dict, Tuple, Optional, Union

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sdt import plot

from .analyzer import Analyzer
from ..data_store import DataStore


class Plotter:
    def __init__(self):
        super().__init__()
        self.sm_data = {}

    @classmethod
    def load(cls, file_prefix="tracking"):
        ds = DataStore.load(file_prefix, sm_data=True, segment_images=False,
                            flatfield=False)
        ret = cls()
        ret.sm_data = ds.sm_data
        return ret

    def scatter(self, xdata: Any = ("fret", "eff"),
                ydata: Any = ("fret", "stoi"),
                frame: Optional[int] = None, columns: int = 2,
                size: float = 5.0,
                xlim: Tuple[Optional[float], Optional[float]] = (None, None),
                ylim: Tuple[Optional[float], Optional[float]] = (None, None),
                xlabel: Optional[str] = None, ylabel: Optional[str] = None,
                scatter_args: Dict = {}, grid: bool = True,
                ax: Optional[mpl.axes.Axes] = None
                ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Make scatter plots of multiple smFRET datasets

        Parameters
        ----------
        x_data, y_data
            Column indices of data to plot on the x (y) axis.
        frame
            If given, only plot data from a certain frame.
        columns
            In how many columns to lay out plots.
        size
            Size per plot.
        xlim, ylim
            Set x (y) axis limits. Use `None` for automatic determination.
        xlabel, ylabel
            Label for x (y) axis. If `None`, use `x_data` (`y_data`).
        scatter_args : dict, optional
            Further arguments to pass as keyword arguments to the scatter
            :py:meth:`matplotlib.axes.Axes.scatter` function.
        grid
            Whether to draw a grid in the plots.
        ax
            Axes to use for plotting. If `None`, a new figure with axes is
            created.

        Returns
        -------
        Figure and axes instances used for plotting
        """
        if ax is None:
            rows = math.ceil(len(self.sm_data) / columns)
            fig, ax = plt.subplots(rows, columns, sharex=True, sharey=True,
                                   squeeze=False,
                                   figsize=(columns*size, rows*size),
                                   constrained_layout=True)
        else:
            ax = np.array(ax).reshape((-1, columns))
            fig = ax.flatten()[0].figure

        for (key, dset), a in zip(self.sm_data.items(), ax.flatten()):
            x = []
            y = []
            for d in dset.values():
                f = Analyzer._apply_filters(d["donor"])
                if frame is not None:
                    f = f[f["fret", "frame"] == frame]
                sub_x = f[xdata].to_numpy(dtype=float)
                sub_y = f[ydata].to_numpy(dtype=float)
                m = np.isfinite(sub_x) & np.isfinite(sub_y)
                x.append(sub_x[m])
                y.append(sub_y[m])
            x = np.concatenate(x)
            y = np.concatenate(y)
            try:
                plot.density_scatter(x, y, ax=a, **scatter_args)
            except Exception:
                a.scatter(x, y, **scatter_args)
            a.set_title(key)

        for a in ax.flatten()[len(self.sm_data):]:
            a.axis("off")

        xlabel = xlabel if xlabel is not None else " ".join(xdata)
        ylabel = ylabel if ylabel is not None else " ".join(ydata)

        for a in ax.flatten():
            if grid:
                a.grid()
            a.set_xlim(*xlim)
            a.set_ylim(*ylim)

        for a in ax[-1]:
            a.set_xlabel(xlabel)
        for a in ax[:, 0]:
            a.set_ylabel(ylabel)

        return fig, ax

    def hist(self, data: Any = ("fret", "eff"),
             frame: Optional[int] = None, columns: int = 2,
             size: float = 5,
             xlim: Tuple[Optional[float], Optional[float]] = (None, None),
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             group_re: Union[None, str, re.Pattern] = None,
             hist_args: Dict = {}, ax: Optional[np.ndarray] = None
             ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
        """Make histogram plots of multiple smFRET datasets

        Parameters
        ----------
        data
            Column indices of data to plot.
        frame
            If given, only plot data from a certain frame.
        columns
            In how many columns to lay out plots.
        size
            Size per plot.
        xlim
            Set x axis limits. Use `None` for automatic determination.
        xlabel, ylabel
            Label for x (y) axis. If `None`, use `data` for the x axis and
            ``"# events"`` on the y axis.
        group_re
            Regular expression matched against dataset names to determine
            which datasets should be plotted into the same axes. The reg ex
            should have two named groups: "group" and "label". Datasets with
            the same "group" will be in the same axes and are identified via
            the "label" in the legend. If `None`, each dataset is plotted into
            individual axes.
        hist_args
            Further arguments to pass as keyword arguments to
            :py:meth:`matplotlib.pyplot.Axes.hist`.
        ax
            Axes to use for plotting. If `None`, a new figure with axes is
            created.

        Returns
        -------
        Figure and axes instances used for plotting
        """
        if group_re is not None:
            if isinstance(group_re, str):
                group_re = re.compile(group_re)

            grouped = {}
            for key, dset in self.sm_data.items():
                m = group_re.search(key)
                if m is None:
                    raise ValueError(f"regular expression {group_re} does not "
                                     f"match dataset name {key}")
                grp = m.group("group")
                key = m.group("label")
                grouped.setdefault(grp, []).append((key, dset))
        else:
            grouped = {k: [(None, v)] for k, v in self.sm_data.items()}

        if ax is None:
            rows = math.ceil(len(grouped) / columns)
            fig, ax = plt.subplots(rows, columns, squeeze=False, sharex=True,
                                   figsize=(columns*size, rows*size),
                                   constrained_layout=True)
        else:
            ax = np.array(ax).reshape((-1, columns))
            fig = ax.flatten()[0].figure

        hist_args.setdefault("bins", np.linspace(-0.5, 1.5, 50))
        hist_args.setdefault("density", False)

        for (g_key, items), a in zip(grouped.items(), ax.flatten()):
            show_legend = False
            for label, dset in items:
                x = []
                for d in dset.values():
                    f = Analyzer._apply_filters(d["donor"])
                    if frame is not None:
                        f = f[f["fret", "frame"] == frame]
                    sub_x = f[data].to_numpy(dtype=float)
                    m = np.isfinite(sub_x)
                    x.append(sub_x[m])

                a.hist(np.concatenate(x), label=label, **hist_args)
                if label:
                    show_legend = True
            a.set_title(g_key)
            if show_legend:
                a.legend(loc=0)

        for a in ax.flatten()[len(grouped):]:
            a.axis("off")

        xlabel = xlabel if xlabel is not None else " ".join(data)
        if ylabel is None:
            if hist_args.get("density", False):
                ylabel = "# events"
            else:
                ylabel = "probability density"

        for a in ax.flatten():
            a.set_xlabel(xlabel)
            a.set_ylabel(ylabel)
            a.grid()
            a.set_xlim(*xlim)

        return fig, ax
