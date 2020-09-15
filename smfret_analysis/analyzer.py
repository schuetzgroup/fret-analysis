# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provide :py:class:`Analyzer` as a Jupyter notebook UI for smFRET analysis"""
import collections
from io import BytesIO
import itertools
import math
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union
try:
    from typing import Literal
except ImportError:
    # Python < 3.8
    from .typing_extensions import Literal
import warnings

import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from sdt import changepoint, flatfield, fret, image, nbui, plot, roi

from .tracker import Tracker
from .version import output_version


class Analyzer:
    """Jupyter notebook UI for analysis of smFRET tracks

    Analyze and filter smFRET tracks produced by :py:class:`Tracker`.
    """
    rois: Dict[str, roi.ROI]
    """channel name -> ROI"""
    excitation_seq: np.ndarray
    """Array of characters describing the excitation sequence."""
    analyzers: Dict[str, fret.SmFretAnalyzer]
    """dataset key -> analyzer class instance"""
    sources: Dict[str, Dict]
    """dataset key -> source file information. The information is a dict
    mapping "files" -> list of file names, "cells" -> bool as passed to
    :py:meth:`Tracker.add_dataset`.
    """
    cell_images: Dict[str, np.ndarray]
    """file name -> 3D array where the first index specifies which of possible
    multiple images to use.
    """
    flatfield: Dict[str, flatfield.Corrector]
    """channel name -> flatfield correction class instance"""

    def __init__(self, file_prefix: str = "tracking"):
        """Parameters
        ----------
        file_prefix
            Prefix for :py:class:`Tracker` save files to load. Same as
            ``file_prefix`` argument passed to :py:meth:`Tracker.save`.
        """
        cfg = Tracker.load_data(file_prefix, loc=False)

        self.rois = cfg["rois"]
        self.excitation_seq = cfg["tracker"].excitation_seq
        self.analyzers = {k: fret.SmFretAnalyzer(v)
                          for k, v in cfg["track_data"].items()}
        self.sources = cfg["sources"]
        self.cell_images = cfg["cell_images"]
        self.flatfield = cfg["flatfield"]

        self._segment_fig = None
        self._brightness_fig = None
        self._thresholder = None
        self._beam_shape_fig = None
        self._beam_shape_artists = [None, None]
        self._beta_population_fig = None
        self._eff_stoi_fig = None

    def present_at_start(self, frame: Optional[int] = None,
                         special: Optional[Sequence["str"]] = None):
        """Remove tracks that are not present in the beginning

        Parameters
        ----------
        frame
            Start frame number. If `None`, use first donor excitation frame
            except for samples with ``special="acc-only"``, where the first
            acceptor excitation frame is used.
        special
            Only apply filter to datasets with ``special=`` set to something
            in this list.
        """
        if frame is None:
            frame_d = np.nonzero(self.excitation_seq == "d")[0][0]
            frame_a = np.nonzero(self.excitation_seq == "a")[0]
            # If there was no acceptor excitation…
            frame_a = frame_a[0] if frame_a.size else -1
        else:
            frame_d = frame_a = frame

        for k, v in self.sources.items():
            if special is not None and v["special"] not in special:
                continue
            if v["special"].startswith("a"):
                # acceptor-only
                f = frame_a
            else:
                f = frame_d

            self.analyzers[k].query_particles(f"donor_frame == {f}")

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

    def filter_beam_shape_region(self, channel: Literal["donor", "accetpor"],
                                 thresh: float):
        """Select only datapoints within the region of bright laser excitation

        Parameters
        ----------
        channel
            Which channel's laser profile to use.
        thresh
            Threshold in percent of the amplitude
        """
        mask = self.flatfield[channel].corr_img * 100 > thresh
        for a in self.analyzers.values():
            a.image_mask(mask, channel)

    def flatfield_correction(self):
        """Apply flatfield correction to brightness data

        See also
        --------
        fret.SmFretAnalyzer.flatfield_correction
        """
        for a in self.analyzers.values():
            a.flatfield_correction(self.flatfield["donor"],
                                   self.flatfield["acceptor"])

    def calc_fret_values(self, *args, **kwargs):
        """Calculate FRET-related quantities

        Compute apparent FRET efficiencies and stoichiometries,
        the total brightness (mass) upon donor excitation, and the acceptor
        brightness (mass) upon direct excitation, which is interpolated for
        donor excitation datapoints in order to allow for calculation of
        stoichiometries.

        See also
        --------
        fret.SmFretAnalyzer.calc_fret_values
        """
        for a in self.analyzers.values():
            a.calc_fret_values(*args, **kwargs)

    def _make_dataset_selector(self, state: Dict,
                               callback: Callable[[Optional[Any]], None]) \
            -> ipywidgets.Dropdown:
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
        d_sel = ipywidgets.Dropdown(options=list(self.analyzers.keys()),
                                    description="dataset")

        def change_dataset(key=None):
            trc = self.analyzers[d_sel.value].tracks
            state["id"] = d_sel.value
            state["trc"] = trc
            state["pnos"] = sorted(trc["fret", "particle"].unique())
            state["files"] = \
                trc.index.remove_unused_levels().levels[0].unique()

            callback()

        d_sel.observe(change_dataset, names="value")
        change_dataset()

        return d_sel

    def find_segment_options(self) -> ipywidgets.Widget:
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
            td = t[t["fret", "exc_type"] == "d"]
            ta = t[t["fret", "exc_type"] == "a"]

            fd = td["donor", "frame"].values
            dm = td["fret", "d_mass"].values
            fa = ta["donor", "frame"].values
            am = ta["fret", "a_mass"].values
            ef = td["fret", "eff_app"].values
            hn = td["fret", "has_neighbor"].values

            cp_a = self.analyzers[state["id"]].cp_detector.find_changepoints(
                am, pen_a_sel.value) - 1
            cp_d = self.analyzers[state["id"]].cp_detector.find_changepoints(
                dm, pen_d_sel.value) - 1
            cp_d_all = np.concatenate([np.searchsorted(fd, fa[cp_a]) - 1,
                                       cp_d])
            cp_d_all = np.unique(cp_d_all)
            cp_d_all.sort()

            if len(fd):
                changepoint.plot_changepoints(dm, cp_d, time=fd, ax=ax_dm)
                changepoint.plot_changepoints(ef, cp_d_all, time=fd, ax=ax_eff)
            if len(fa):
                changepoint.plot_changepoints(am, cp_a, time=fa, ax=ax_am)
            ax_hn.plot(fd, hn, c="C2", alpha=0.2)

            ax_dm.relim(visible_only=True)
            ax_am.relim(visible_only=True)
            ax_eff.set_ylim(-0.05, 1.05)
            ax_hn.set_ylim(-0.05, 1.05)

            ax_dm.set_title("d_mass")
            ax_am.set_title("a_mass")
            ax_eff.set_title("app. eff")

            self._segment_fig.tight_layout()
            self._segment_fig.canvas.draw()

        p_sel.observe(show_track, "value")
        pen_d_sel.observe(show_track, "value")
        pen_a_sel.observe(show_track, "value")

        d_sel = self._make_dataset_selector(state, show_track)

        return ipywidgets.VBox([d_sel, p_sel, pen_d_sel, pen_a_sel,
                                self._segment_fig.canvas, p_label])

    def segment_mass(self, channel: Literal["donor", "acceptor"], **kwargs):
        """Segment tracks by changepoint detection in brightness time traces

        This adds a column with segment numbers for each track.

        Parameters
        ----------
        channel
            Whether to perform changepoint detection in the donor or acceptor
            emission channel.
        **kwargs
            Arguments passed to the changepoint detection algorithm
            (:py:meth:`sdt.changepoint.Pelt.find_changepoints`).

        See also
        --------
        fret.SmFretAnalyzer.segment_mass
        """
        for a in self.analyzers.values():
            a.segment_mass(channel, **kwargs)

    def filter_bleach_step(self, donor_thresh: float, acceptor_thresh: float):
        """Remove tracks that do not show expected bleaching steps

        Acceptor must bleach in a single step, while donor must not show more
        than one step.

        Parameters
        ----------
        donor_thresh, acceptor_thresh
            Consider the donor / acceptor bleached if the segment median is
            below donor_thresh/acceptor_thresh.

        See also
        --------
        fret.SmFretAnalyzer.bleach_step
        """
        for k, v in self.sources.items():
            if v["special"].startswith("d") or v["special"].startswith("a"):
                # Don't filter donor-only and acceptor-only samples
                continue
            a = self.analyzers[k]
            a.bleach_step(donor_thresh, acceptor_thresh, truncate=False)

    def set_leakage(self, leakage: float):
        r"""Set the leakage correction factor for all datasets

        Parameters
        ----------
        leakage
            Leakage correction factor (often denoted :math:`\alpha`)

        See also
        --------
        fret.SmFretAnalyzer.leakage
        """
        for a in self.analyzers.values():
            a.leakage = leakage

    def calc_leakage(self, remove: bool = True):
        """Calculate leakage correction factor from donor-only sample

        This uses the first (probably only) dataset with ``special=don-only``
        set when calling :py:meth:`add_dataset`.

        Parameters
        ----------
        remove
            Remove donor-only dataset afterwards since it is probably not
            of further interest.

        See also
        --------
        calc_leakage_from_bleached
        fret.SmFretAnalyzer.calc_leakage
        """
        dataset = next(k for k, v in self.sources.items()
                       if v["special"].startswith("d"))
        a = self.analyzers[dataset]
        a.calc_leakage()
        self.set_leakage(a.leakage)
        if remove:
            self.analyzers.pop(dataset)

    # TODO: Move to sdt package
    def calc_leakage_from_bleached(
            self, datasets: Union[str, Sequence[str], None] = None,
            print_summary: bool = False):
        """Calculate leakage correction factor from bleached acceptor traces

        This takes those parts of traces where the acceptor is bleached, but
        the donor isn't.

        Parameters
        ----------
        datasets
            dataset(s) to use. If `None`, use all.
        print_summary
            Print number of datapoints and result.

        See also
        --------
        calc_leakage
        fret.SmFretAnalyzer.calc_leakage
        """
        if datasets is None:
            datasets = list(self.analyzers)
        elif isinstance(datasets, str):
            datasets = [datasets]

        effs = []
        for d in datasets:
            trc = self.analyzers[d].tracks
            # search for donor excitation frames where acceptor is bleached
            # but donor isn't
            sel = ((trc["fret", "exc_type"] == "d") &
                   (trc["fret", "has_neighbor"] == 0) &
                   (trc["fret", "a_seg"] == 1) &
                   (trc["fret", "d_seg"] == 0))
            effs.append(trc.loc[sel, ("fret", "eff_app")].values)
        effs = np.concatenate(effs)

        n_data = len(effs)
        m_eff = np.mean(effs)
        leakage = m_eff / (1 - m_eff)

        self.set_leakage(leakage)

        if print_summary:
            print(f"leakage: {leakage:.4f} (from {n_data} datapoints)")

    def set_direct_excitation(self, dir_exc: float):
        r"""Set the direct (cross-) excitation factor for all datasets

        Parameters
        ----------
        dir_exc
            Direct excitation correction factor (often denoted :math:`\delta`)

        See also
        --------
        fret.SmFretAnalyzer.direct_excitation
        """
        for a in self.analyzers.values():
            a.direct_excitation = dir_exc

    def calc_direct_excitation(self, remove: bool = True):
        """Calculate direct excitation correction factor from acc-only sample

        This uses the first (probably only) dataset with ``special=acc-only``
        set when calling :py:meth:`add_dataset`.

        Parameters
        ----------
        remove
            Remove donor-only dataset afterwards since it is probably not
            of further interest.

        See also
        --------
        calc_direct_excitation_from_bleached
        fret.SmFretAnalyzer.calc_direct_excitation
        """
        dataset = next(k for k, v in self.sources.items()
                       if v["special"].startswith("a"))
        a = self.analyzers[dataset]
        a.calc_direct_excitation()
        self.set_direct_excitation(a.direct_excitation)
        if remove:
            self.analyzers.pop(dataset)

    # TODO: Move to sdt package
    def calc_direct_excitation_from_bleached(
            self, datasets: Union[str, Sequence[str], None] = None,
            print_summary: bool = False):
        """Calculate dir. exc. correction factor from bleached donor traces

        This takes those parts of traces where the donor is bleached, but
        the acceptor isn't.

        Parameters
        ----------
        datasets
            dataset(s) to use. If `None`, use all.
        print_summary
            Print number of datapoints and result.

        See also
        --------
        calc_direct_excitation
        fret.SmFretAnalyzer.calc_direct_excitation
        """
        if datasets is None:
            datasets = list(self.analyzers)
        elif isinstance(datasets, str):
            datasets = [datasets]

        stois = []
        for d in datasets:
            trc = self.analyzers[d].tracks
            # search for donor excitation frames where donor is bleached
            # but acceptor isn't
            sel = ((trc["fret", "exc_type"] == "d") &
                   (trc["fret", "has_neighbor"] == 0) &
                   (trc["fret", "a_seg"] == 0) &
                   (trc["fret", "d_seg"] == 1))
            stois.append(trc.loc[sel, ("fret", "stoi_app")].values)
        stois = np.concatenate(stois)

        n_data = len(stois)
        m_stoi = np.mean(stois)
        dir_exc = m_stoi / (1 - m_stoi)

        self.set_direct_excitation(dir_exc)

        if print_summary:
            print(f"direct exc.: {dir_exc:.4f} (from {n_data} datapoints)")

    def set_detection_eff(self, eff: float):
        r"""Set the detection efficiency factor for all datasets

        Parameters
        ----------
        eff
            Detection efficiency correction factor (often denoted
            :math:`\gamma`)

        See also
        --------
        sdt.fret.SmFretAnalyzer.detection_eff
        """
        for a in self.analyzers.values():
            a.detection_eff = eff

    def calc_detection_eff(self, min_part_len: int = 5,
                           how: Union[Callable[[np.array], float],
                                      Literal["individual"]] = "individual",
                           aggregate: Literal["dataset", "all"] = "dataset",
                           dataset: Optional[str] = None):
        r"""Calculate detection efficieny correction factor

        The detection efficiency ratio is the ratio of decrease in acceptor
        brightness to the increase in donor brightness upon acceptor
        photobleaching.

        Parameters
        ----------
        min_part_len
            How many data points need to be present before and after the
            bleach step to ensure a reliable calculation of the mean
            intensities. If there are fewer data points, a value of NaN will be
            assigned.
        how
            If "individual", the :math:`\gamma` value for each track will be
            stored and used to correct the values individually when calling
            :py:meth:`fret_correction`. If a function, apply this function
            to the :math:`\gamma` array and its return value as a global
            correction factor. A sensible example for such a function would be
            :py:func:`numpy.nanmean`. Beware that some :math:`\gamma` may be
            NaN.
        aggregate
            Whether to use a callable `how` argument to calculate the
            correction for each dataset individually or a single factor for all
            datasets. Ignored if `how` is not callable or `dataset` is not
            `None`.
        dataset
            If `how` is callable, use the given dataset to compute a global
            correction factor for all datasets.

        See also
        --------
        sdt.fret.SmFretAnalyzer.calc_detection_eff
        """
        if callable(how):
            if dataset is not None:
                a = self.analyzers[dataset]
                a.calc_detection_eff(min_part_len, how)
                self.set_detection_eff(a.detection_eff)
            elif aggregate == "dataset":
                for a in self.analyzers.values():
                    a.calc_detection_eff(min_part_len, how)
            elif aggregate == "all":
                effs = []
                for a in self.analyzers.values():
                    a.calc_detection_eff(min_part_len, "individual")
                    effs.append(a.detection_eff)
                self.set_detection_eff(how(pd.concat(effs)))
            else:
                raise ValueError(
                    "`aggregate` must be in {\"dataset\", \"all\"}.")
        elif how == "individual":
            for a in self.analyzers.values():
                a.calc_detection_eff(min_part_len, how)
        else:
            raise ValueError(
                "`how` must be either callable or \"individual\".")

    def set_excitation_eff(self, eff: float):
        r"""Set the excitation efficiency factor for all datasets

        Parameters
        ----------
        eff
            Excitation efficiency correction factor (often denoted
            :math:`\beta`)

        See also
        --------
        sdt.fret.SmFretAnalyzer.excitation_eff
        """
        for a in self.analyzers.values():
            a.excitation_eff = eff

    def find_excitation_eff_component(self) -> ipywidgets.Widget:
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
        if self._beta_population_fig is None:
            self._beta_population_fig, _ = plt.subplots()

        state = {}

        n_comp_sel = ipywidgets.IntText(description="components", value=1)

        cmaps = ["viridis", "gray", "plasma", "jet"]

        def update(change=None):
            d = state["trc"]
            d = d[d["fret", "a_seg"] == 0].copy()
            split = fret.gaussian_mixture_split(d, n_comp_sel.value)

            ax = self._beta_population_fig.axes[0]
            ax.cla()
            for v, cm in zip(split, itertools.cycle(cmaps)):
                sel = (d["fret", "particle"].isin(v) &
                       np.isfinite(d["fret", "eff_app"]) &
                       np.isfinite(d["fret", "stoi_app"]))
                plot.density_scatter(d.loc[sel, ("fret", "eff_app")],
                                     d.loc[sel, ("fret", "stoi_app")],
                                     marker=".", label=f"{len(v)}", ax=ax,
                                     cmap=cm)
            ax.legend(loc=0)
            ax.set_xlabel("apparent eff.")
            ax.set_ylabel("apparent stoi.")
            self._beta_population_fig.canvas.draw()

        d_sel = self._make_dataset_selector(state, update)
        n_comp_sel.observe(update, "value")

        return ipywidgets.VBox([d_sel, n_comp_sel,
                                self._beta_population_fig.canvas])

    def calc_excitation_eff(self, dataset: str, n_components: int = 1,
                            component: int = 0):
        """Calculate excitation efficieny correction factor

        Utilize data with known 1:1 stoichiometry to this end. To find the
        right parameters for this call, use
        :py:meth:`find_excitation_eff_component`.

        Parameters
        ----------
        dataset
            Dataset to use.
        n_components
            Perform an n-component Gaussian mixture model fit to separate
            subpopulations in E–S space.
        component
            Identifier of the component to use. They are sorted in descending
            order w.r.t. mean stoichiometry.

        See also
        --------
        sdt.fret.gaussian_mixture_split
        sdt.fret.SmFretAnalyzer.calc_excitation_eff
        """
        a = self.analyzers[dataset]
        a.calc_excitation_eff(n_components, component)
        self.set_excitation_eff(a.excitation_eff)

    def fret_correction(self, *args, **kwargs):
        """Apply corrections to calculate real FRET-related values

        Parameters
        ----------
        *args, **kwargs
            Passed to :py:meth:`sdt.fret.SmFretAnalyzer.fret_correction`

        See also
        --------
        sdt.fret.SmFretAnalyzer.fret_correction
        """
        for a in self.analyzers.values():
            a.fret_correction(*args, **kwargs)

    def query(self, expr: str, mi_sep: str = "_"):
        """Filter localizations

        Use :py:meth:`sdt.fret.SmFretAnalyzer.query` to select only
        datapoints (i.e., lines in tracking tables) that fulfill `expr`.

        Parameters
        ----------
        expr
            Filter expression. See :py:meth:`pandas.DataFrame.eval` for
            details.
        mi_sep
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".

        See also
        --------
        pandas.DataFrame.eval
        sdt.fret.SmFretAnalyzer.query
        """
        for a in self.analyzers.values():
            a.query(expr, mi_sep)

    def query_particles(self, expr: str, min_abs: int = 1,
                        min_rel: float = 0.0, mi_sep: str = "_"):
        """Filter tracks

        Use :py:meth:`sdt.fret.SmFretAnalyzer.query_particles` to select only
        trajectories that fulfill `expr` sufficiently many times.

        Parameters
        ----------
        expr
            Filter expression. See :py:meth:`pandas.DataFrame.eval` for
            details.
        min_abs
            Minimum number of times a particle has to fulfill `expr`. If
            negative, this means "all but ``abs(min_count)``". If 0, it has
            to be fulfilled in all frames.
        min_rel
            Minimum fraction of data points that have to fulfill `expr` for a
            particle not to be removed
        mi_sep
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".

        See also
        --------
        pandas.DataFrame.eval
        sdt.fret.SmFretAnalyzer.query_particles
        """
        for a in self.analyzers.values():
            a.query_particles(expr, min_abs, min_rel, mi_sep)

    def find_cell_mask_params(self) -> nbui.Thresholder:
        """UI for finding parameters for thresholding cell images

        Parameters can be passed to :py:meth:`apply_cell_masks`.

        Returns
        -------
            Widget instance.
        """
        if self._thresholder is None:
            self._thresholder = nbui.Thresholder()

        self._thresholder.images = collections.OrderedDict(
            [(k, v[0]) for k, v in self.cell_images.items()])

        return self._thresholder

    def apply_cell_masks(self, thresh_algorithm: str = "adaptive", **kwargs):
        """Remove datapoints from non-cell-occupied regions

        Threshold cell images according to the parameters and use the resulting
        mask to discard datapoints not underneath cells. This is applied only
        to datasets which have ``special="cells"`` set.

        Parameters
        ----------
        thresh_algorithm
            Use the ``thresh_algorithm + "_thresh"`` function from
            :py:mod:`sdt.image` for thresholding.
        **kwargs
            Passed to the thresholding function.
        """
        if isinstance(thresh_algorithm, str):
            thresh_algorithm = getattr(image, thresh_algorithm + "_thresh")

        for k, v in self.sources.items():
            if not v["special"].startswith("c"):
                continue

            ana = self.analyzers[k]
            files = np.unique(ana.tracks.index.levels[0])

            masks = []
            for f in files:
                ci = self.cell_images[f]

                # Get frame numbers of all cell images
                c_pos = np.nonzero(self.excitation_seq == "c")[0]
                n_rep = math.ceil(len(ci) / len(c_pos))

                all_c_pos = np.empty(n_rep * len(c_pos), dtype=int)
                for i in range(n_rep):
                    all_c_pos[i * len(c_pos):(i + 1) * len(c_pos)] = \
                        i * len(self.excitation_seq) + c_pos
                all_c_pos = all_c_pos[:len(ci)]

                # Apply each cell mask to all frames from the time of its
                # recording to the recording of the next cell image
                for i, (start, stop) in enumerate(zip(
                        all_c_pos, itertools.chain(all_c_pos[1:], [None]))):
                    masks.append({"key": f,
                                  "mask": thresh_algorithm(ci[i], **kwargs),
                                  "start": start,
                                  "stop": stop})
            ana.image_mask(masks, channel="donor")

    def save(self, file_prefix: str = "filtered"):
        """Save results to disk

        This will save filtered data to disk.

        Parameters
        ----------
        file_prefix
            Prefix for the file written by this method. It will be suffixed by
            the output format version (v{output_version}) and file extension.
        """
        outfile = Path(f"{file_prefix}-v{output_version:03}")

        with warnings.catch_warnings():
            import tables
            warnings.simplefilter("ignore", tables.NaturalNameWarning)

            with pd.HDFStore(outfile.with_suffix(".h5")) as s:
                for key, ana in self.analyzers.items():
                    s.put(f"{key}_trc", ana.tracks.astype(
                        {("fret", "exc_type"): str}))

    @staticmethod
    def load_data(file_prefix: str = "filtered") -> Dict[str, Any]:
        """Load data to a dictionary

        Parameters
        ----------
        file_prefix
            Prefix used for saving via :py:meth:`save`.

        Returns
        -------
            Dictionary of loaded data
        """
        ret = collections.OrderedDict()
        infile = Path(f"{file_prefix}-v{output_version:03}.h5")
        with pd.HDFStore(infile, "r") as s:
            for k in s.keys():
                if not k.endswith("_trc"):
                    continue
                key = k.lstrip("/")[:-4]
                ret[key] = s[k].astype({("fret", "exc_type"): "category"})
        return ret


class DensityPlots(ipywidgets.Box):
    """A widget for displaying density plots

    With this, one can plot data after individual filtering steps to see
    the progress
    """
    def __init__(self, analyzer: Analyzer, datasets: Sequence[str],
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
            t = self._ana.analyzers[d].tracks
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
