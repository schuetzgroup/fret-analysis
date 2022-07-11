# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for analyzing and filtering intramolecular smFRET data"""
from collections import defaultdict
import itertools
from typing import (Any, Callable, Iterable, Literal, Mapping, Optional,
                    Sequence, Union)

import numpy as np
import pandas as pd
from sdt import changepoint, helper

from .analyzer_base import BaseAnalyzer


class IntramolecularAnalyzer(BaseAnalyzer):
    """Class for analyzing and filtering of smFRET data

    This provides various analysis and filtering methods which act on the
    :py:attr:`tracks` attribute.

    Correction of FRET efficiencies and stoichiometries for donor leakage and
    direct acceptor excitation is implemented according to [Hell2018]_,
    while different detection efficiencies for the different
    fluorophores are accounted for as described in [MacC2010]_ as well
    the different excitation efficiencies according to [Lee2005]_.
    """

    bleach_threshold: Mapping[Literal["donor", "acceptor"], float
                              ] = {"donor": 500, "acceptor": 500}
    """Intensity (mass) thresholds upon donor and acceptor excitation,
    respecitively, below which a signal is considered bleached. Used for
    bleaching step analysis.
    """
    cp_detector: Any
    """Changepoint detector class instance used to perform bleching step
    detection.
    """

    _app_vals_columns = [("fret", "particle")]

    def __init__(self, cp_detector: Optional[Any] = None):
        """Parameters
        ----------
        cp_detector
            Changepoint detetctor. If `None`, create a
            :py:class:`changepoint.Pelt` instance with ``model="l2"``,
            ``min_size=1``, and ``jump=1``.
        """
        super().__init__()

        if cp_detector is None:
            cp_detector = changepoint.Pelt("l2", min_size=1, jump=1)
        self.cp_detector = cp_detector

    def _mass_changepoints_single(self, tracks: pd.DataFrame, channel: str,
                                  stat_funcs: Iterable[Callable],
                                  stat_names: Iterable[str],
                                  stat_margin: int,
                                  **kwargs):
        e_type = channel[0]
        mass_col = f"{e_type}_mass"
        seg_col = f"{e_type}_seg"

        trc = tracks[[("fret", "particle"), ("fret", "frame"),
                      ("fret", mass_col)]]
        trc = trc.sort_values([("fret", "particle"), ("fret", "frame")])
        trc["mask"] = self._apply_filters(tracks, type="mask")

        trc_split = helper.split_dataframe(
            trc, ("fret", "particle"), [("fret", mass_col), ("mask",)],
            type="array_list", sort=False, keep_index=True)

        def cp_func(data):
            return self.cp_detector.find_changepoints(data, **kwargs)

        segments = []
        stat_results = defaultdict(list)
        for p, trc_p in trc_split:
            seg_p, stat_p = changepoint.segment_stats(
                trc_p[1], cp_func, stat_funcs, mask=trc_p[2],
                stat_margin=stat_margin, return_len="data")
            segments.append(pd.Series(seg_p, index=trc_p[0]))
            for name, st in zip(stat_names, stat_p.T):
                stat_results[f"{seg_col}_{name}"].append(
                    pd.Series(st, index=trc_p[0]))

        tracks["fret", seg_col] = pd.concat(segments)
        for col, st in stat_results.items():
            tracks["fret", col] = pd.concat(st)

    def mass_changepoints(self, channel: Literal["donor", "acceptor"],
                          stats: Union[Callable, str,
                                       Iterable[Union[Callable, str]]
                                       ] = "median",
                          stat_margin: int = 1,
                          **kwargs):
        """Segment tracks by changepoint detection in brightness time trace

        Changepoint detection is run on the donor or acceptor brightness
        (``mass``) time trace, depending on the `channels` argument.
        This appends py:attr:`tracks` with a ``("fret", "d_seg")`` or
        `("fret", "a_seg")`` column for donor or acceptor, resp. For
        each localization, this holds the number of the segment it belongs to.
        Furthermore, statistics (such as median brightness) can/should be
        calculated, which can later be used to analyze stepwise bleaching
        (see :py:meth:`bleach_step`).

        **:py:attr:`tracks` will be sorted according to
        ``("fret", "particle")`` and ``("donor", self.columns["time"])`` in the
        process.**

        Parameters
        ----------
        channel
            In which channel (``"donor"`` or ``"acceptor"``) to perform
            changepoint detection.
        stats
            Statistics to calculate for each track segment. For each entry
            ``s``, a column named ``"{channel}_seg_{s}"`` is appendend, where
            ``channel`` is ``d`` for donor and ``a`` for acceptor.
            ``s`` can be the name of a numpy function or a callable returning
            a statistic, such as :py:func:`numpy.mean`.
        stat_margin
            Number of data points around a changepoint to exclude from
            statistics calculation. This can prevent bias in the statistics due
            to recording a bleaching event in progress.
        **kwargs
            Keyword arguments to pass to :py:attr:`cp_detector`
            `find_changepoints()` method.

        Examples
        --------
        Pass ``penalty=1e6`` to the changepoint detector's
        ``find_changepoints`` method, perform detection both channels:

        >>> ana.mass_changepoints("donor", penalty=1e6)
        >>> ana.mass_changepoints("acceptor", penalty=1e6)
        """
        if isinstance(stats, str) or callable(stats):
            stats = [stats]
        stat_funcs = []
        stat_names = []
        for st in stats:
            if isinstance(st, str):
                stat_names.append(st)
                stat_funcs.append(getattr(np, st))
            elif callable(st):
                stat_names.append(st.__name__)
                stat_funcs.append(st)
            else:
                stat_names.append(st[0])
                stat_funcs.append(st[1])

        # iterate datasets
        for dset in itertools.chain(self.sm_data.values(),
                                    self.special_sm_data.values()):
            # iterate files
            for d in dset.values():
                self._mass_changepoints_single(d["donor"], channel, stat_funcs,
                                               stat_names, stat_margin,
                                               **kwargs)

    def _bleaches(self, steps: Sequence[float],
                  channel: Literal["donor", "acceptor"]) -> bool:
        """Returns whether there is single-step bleaching

        Parameters
        ----------
        steps
            Intensity values (e.g., mean intensity) for each step
        channel
            0 for donor, 1 for acceptor. Used to get bleaching thershold from
            :py:attr:`bleach_thresh`.

        Returns
        -------
        `True` if track exhibits single-step bleaching, `False` otherwise.
        """
        return (len(steps) > 1 and
                all(s < self.bleach_threshold[channel] for s in steps[1:]))

    def _bleaches_partially(self, steps: Sequence[float],
                            channel: Literal["donor", "acceptor"]) -> bool:
        """Returns whether there is partial bleaching

        Parameters
        ----------
        steps
            Intensity values (e.g., mean intensity) for each step
        channel
            0 for donor, 1 for acceptor. Used to get bleaching thershold from
            :py:attr:`bleach_thresh`.

        Returns
        -------
        `True` if there is a bleaching step that does not go below threshold,
        `False` if there is no bleaching step or bleaching goes below
        threshold in a single step.
        """
        return (len(steps) > 1 and
                any(s > self.bleach_threshold[channel] for s in steps[1:]))

    def _bleach_step_single(
            self, tracks: Mapping[Literal["donor", "acceptor"], pd.DataFrame],
            condition: Literal["donor", "acceptor", "donor or acceptor",
                               "no partial"],
            stat: str, reason: str, reason_cnt: int):
        trc = tracks["donor"].sort_values([("fret", "particle"),
                                           ("fret", "frame")])

        trc_split = helper.split_dataframe(
            trc, ("fret", "particle"),
            [("fret", "d_seg"), ("fret", f"d_seg_{stat}"),
             ("fret", "a_seg"), ("fret", f"a_seg_{stat}")],
            type="array_list", sort=False)

        good_p = []
        for p, trc_p in trc_split:
            is_good = True
            if -1 in trc_p[0] or -1 in trc_p[2]:
                # -1 as segment number means that changepoint detection failed
                continue

            # Get change changepoints upon acceptor exc from segments
            cps_a = np.nonzero(np.diff(trc_p[2]))[0] + 1
            split_a = np.array_split(trc_p[3], cps_a)
            stat_a = [s[0] for s in split_a]

            # Get change changepoints upon donor exc from segments
            cps_d = np.nonzero(np.diff(trc_p[0]))[0] + 1
            split_d = np.array_split(trc_p[1], cps_d)
            stat_d = [s[0] for s in split_d]

            if condition == "donor":
                is_good = (self._bleaches(stat_d, "donor") and not
                           self._bleaches_partially(stat_a, "acceptor"))
            elif condition == "acceptor":
                is_good = (self._bleaches(stat_a, "acceptor") and not
                           self._bleaches_partially(stat_d, "donor"))
            elif condition in ("donor or acceptor", "acceptor or donor"):
                is_good = ((self._bleaches(stat_d, "donor") and not
                            self._bleaches_partially(stat_a, "acceptor")) or
                           (self._bleaches(stat_a, "acceptor") and not
                            self._bleaches_partially(stat_d, "donor")))
            elif condition == "no partial":
                is_good = not (self._bleaches_partially(stat_d, "donor") or
                               self._bleaches_partially(stat_a, "acceptor"))
            else:
                raise ValueError(f"unknown strategy: {condition}")

            if is_good:
                good_p.append(p)

        filtered_d = tracks["donor"]["fret", "particle"].isin(good_p)
        self._update_filter(tracks["donor"],
                            np.asarray(~filtered_d, dtype=np.intp),
                            reason, reason_cnt)
        filtered_a = tracks["acceptor"]["fret", "particle"].isin(good_p)
        self._update_filter(tracks["acceptor"],
                            np.asarray(~filtered_a, dtype=np.intp),
                            reason, reason_cnt)

    def bleach_step(self, condition: str = "donor or acceptor",
                    stat: str = "median", reason: str = "bleach_step"):
        """Find tracks with acceptable fluorophore bleaching behavior

        What "acceptable" means is specified by the `condition` parameter.

        ``("fret", "d_seg")``, ``("fret", f"d_seg_{stat}")``, ``("fret",
        "a_seg")``, and ``("fret", f"a_seg_{stat}")`` (where ``{stat}`` is
        replaced by the value of the `stat` parameter) columns need to be
        present in :py:attr:`tracks` for this to work, which can be achieved by
        performing changepoint in both channels using :py:meth:`segment_mass`.

        The donor considered bleached if its ``("fret", f"d_seg_{stat}")``
        is below :py:attr:`bleach_threshold` ``["donor"]``. The acceptor
        considered bleached if its ``("fret", f"a_seg_{stat}")`` is below
        :py:attr:`bleach_threshold` ``["acceptor"]``

        Parameters
        ----------
        condition
            If ``"donor"``, accept only tracks where the donor bleaches in a
            single step and the acceptor shows either no bleach step or
            completely bleaches in a single step.
            Likewise, ``"acceptor"`` will accept only tracks where the acceptor
            bleaches fully in one step and the donor shows no partial
            bleaching.
            ``donor or acceptor`` requires that one channel bleaches in a
            single step while the other either also bleaches in one step or not
            at all (no partial bleaching).
            If ``"no partial"``, there may be no partial bleaching, but
            bleaching is not required.
        stat
            Statistic to use to determine bleaching steps. Has to be one that
            was passed to via ``stats`` parameter to
            :py:meth:`mass_changepoints`.
        reason
            Filtering reason / column name to use.

        Examples
        --------
        Consider acceptors with a brightness ``("fret", "a_mass")`` of less
        than 500 counts and donors with a brightness ``("fret", "d_mass")`` of
        less than 800 counts bleached. Remove all tracks that don't show
        acceptable bleaching behavior.

        >>> ana.bleach_threshold = {"donor": 800, "acceptor": 500}
        >>> ana.bleach_step("donor or acceptor")
        """
        reason_cnt = self._increment_reason_counter(reason)

        # iterate datasets
        # TODO: This makes no sense e.g. for donor-only and acceptor-only
        # datasets
        for dset in itertools.chain(self.sm_data.values(),
                                    self.special_sm_data.values()):
            # iterate files
            for d in dset.values():
                self._bleach_step_single(d, condition, stat, reason,
                                         reason_cnt)

    def _query_particles_single(
            self, tracks: Mapping[Literal["donor", "acceptor"], pd.DataFrame],
            expr: str, min_abs: int, min_rel: float, mi_sep: str,
            reason: str, reason_cnt: int):
        pre_filtered = self._apply_filters(tracks["donor"])
        e = self._eval(pre_filtered, expr, mi_sep).to_numpy()
        all_p = pre_filtered["fret", "particle"].to_numpy()
        p, c = np.unique(all_p[e], return_counts=True)
        p_sel = np.ones(len(p), dtype=bool)

        if min_abs is not None:
            if min_abs <= 0:
                p2 = pre_filtered.loc[pre_filtered["fret", "particle"].isin(p),
                                      ("fret", "particle")].to_numpy()
                min_abs = np.unique(p2, return_counts=True)[1] + min_abs
            p_sel &= c >= min_abs
        if min_rel:
            p2 = pre_filtered.loc[pre_filtered["fret", "particle"].isin(p),
                                  ("fret", "particle")].to_numpy()
            c2 = np.unique(p2, return_counts=True)[1]
            p_sel &= (c / c2 >= min_rel)

        good_p = p[p_sel]
        bad_p = np.setdiff1d(all_p, good_p)

        for sub in tracks.values():
            good = sub["fret", "particle"].isin(good_p).to_numpy()
            bad = sub["fret", "particle"].isin(bad_p).to_numpy()
            flt = np.full(len(sub), -1, dtype=np.intp)
            flt[good] = 0
            flt[bad] = 1
            self._update_filter(sub, flt, reason, reason_cnt)

    def query_particles(self, expr: str, min_abs: int = 1,
                        min_rel: float = 0.0, mi_sep: str = "_",
                        reason: str = "query_p"):
        """Remove particles that don't fulfill `expr` enough times

        Any particle that does not fulfill `expr` at least `min_abs` times AND
        during at least a fraction of `min_rel` of its length is removed from
        :py:attr:`tracks`.

        The column MultiIndex is flattened for this purpose.

        Parameters
        ----------
        expr
            Filter expression. See :py:meth:`pandas.DataFrame.eval` for
            details.
        min_abs
            Minimum number of times a particle has to fulfill `expr`. If
            negative, this means "all but ``abs(min_abs)``". If 0, it has
            to be fulfilled in all frames.
        min_rel
            Minimum fraction of data points that have to fulfill `expr` for a
            particle not to be removed.
        mi_sep
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".
        reason
            Filtering reason / column name to use.

        Examples
        --------
        Remove any particles where not ("fret", "a_mass") > 500 at least twice
        from :py:attr:`tracks`.

        >>> filt.query_particles("fret_a_mass > 500", 2)

        Remove any particles where ("fret", "a_mass") <= 500 in more than one
        frame:

        >>> filt.query_particles("fret_a_mass > 500", -1)

        Remove any particle where not ("fret", "a_mass") > 500 for at least
        75 % of the particle's data points, with a minimum of two data points:

        >>> filt.query_particles("fret_a_mass > 500", 2, min_rel=0.75)
        """
        reason_cnt = self._increment_reason_counter(reason)

        # iterate datasets
        for dset in itertools.chain(self.sm_data.values(),
                                    self.special_sm_data.values()):
            # iterate files
            for d in dset.values():
                self._query_particles_single(d, expr, min_abs, min_rel, mi_sep,
                                             reason, reason_cnt)

    def calc_leakage_from_bleached(
            self, datasets: Union[str, Sequence[str], None] = None,
            seg_stat: str = "median", print_summary: bool = False):
        """Calculate leakage correction factor from bleached acceptor traces

        This takes those parts of traces where the acceptor is bleached, but
        the donor isn't.

        Parameters
        ----------
        datasets
            dataset(s) to use. If `None`, use all.
        seg_stat
            Statistic to use to determine bleaching steps. Has to be one that
            was passed to via ``stats`` parameter to
            :py:meth:`mass_changepoints`.
        print_summary
            Print number of datapoints and result.

        See also
        --------
        calc_leakage
        """
        def selector(trc):
            # search for donor excitation frames where acceptor is bleached
            # but donor isn't
            return ((trc["fret", "has_neighbor"] == 0) &
                    (trc["fret", "a_seg"] >= 1) &
                    (trc["fret", f"a_seg_{seg_stat}"] <
                     self.bleach_threshold["acceptor"]) &
                    (trc["fret", "d_seg"] == 0))
        self._calc_leakage_from_data(datasets, selector, print_summary)

    def calc_direct_excitation_from_bleached(
            self, datasets: Union[str, Sequence[str], None] = None,
            seg_stat: str = "median", print_summary: bool = False):
        """Calculate dir. exc. correction factor from bleached donor traces

        This takes those parts of traces where the donor is bleached, but
        the acceptor isn't.

        Parameters
        ----------
        datasets
            dataset(s) to use. If `None`, use all.
        seg_stat
            Statistic to use to determine bleaching steps. Has to be one that
            was passed to via ``stats`` parameter to
            :py:meth:`mass_changepoints`.
        print_summary
            Print number of datapoints and result.

        See also
        --------
        calc_direct_excitation
        """
        def selector(trc):
            return ((trc["fret", "has_neighbor"] == 0) &
                    (trc["fret", "a_seg"] == 0) &
                    (trc["fret", "d_seg"] > 0) &
                    (trc["fret", f"d_seg_{seg_stat}"] <
                     self.bleach_threshold["donor"]))
        self._calc_direct_excitation_from_data(datasets, selector,
                                               print_summary)

    def _calc_detection_eff_single(self, tracks, eff_app_thresh, min_seg_len,
                                   stat):
        gammas = pd.Series(np.NaN,
                           index=tracks["donor"]["fret", "particle"].unique())

        trc = self._apply_filters(tracks["donor"]).sort_values(
            [("fret", "particle"), ("fret", "frame")])
        trc_split = helper.split_dataframe(
            trc, ("fret", "particle"),
            [("donor", "mass"), ("acceptor", "mass"), ("fret", "a_seg"),
             ("fret", "eff_app")],
            type="array_list", sort=False)
        for p, t in trc_split:
            fin_mask = np.isfinite(t[0]) & np.isfinite(t[1])
            pre_mask = (t[2] == 0) & (t[3] >= eff_app_thresh) & fin_mask
            post_mask = (t[2] == 1) & fin_mask

            i_dd_pre = t[0][pre_mask]
            i_dd_post = t[0][post_mask]
            i_da_pre = t[1][pre_mask]
            i_da_post = t[1][post_mask]

            if len(i_dd_pre) < min_seg_len or len(i_dd_post) < min_seg_len:
                continue

            gammas[p] = ((stat(i_da_pre) - stat(i_da_post)) /
                         (stat(i_dd_post) - stat(i_dd_pre)))
        return gammas

    _detection_eff_particle_column = ("fret", "particle")

    def _excitation_eff_filter(self, d: pd.DataFrame) -> pd.Series:
        return (d["fret", "a_seg"] == 0) & (d["fret", "d_seg"] == 0)

    def present_at_start(self, frame: Optional[int] = None,
                         filter_reason: str = "at_start"):
        """Remove tracks that are not present in the beginning

        Parameters
        ----------
        frame
            Start frame number. If `None`, use first donor excitation frame
            except for the acceptor-only sample, where the first acceptor
            excitation frame is used.
        """
        if frame is None:
            if self.frame_selector is None:
                raise ValueError("`frame` not specified and `frame_selector` "
                                 "attribute not set.")
            e_seq = self.frame_selector.eval_seq(-1)
            frame = np.nonzero(e_seq == "d")[0][0]

        # TODO: Do not apply to acceptor-only, maybe also not to donor-only
        # This actually depends on whether acc or don excitation is first.
        # Maybe don't apply to both?
        self.query_particles(f"fret_frame == {frame}", reason=filter_reason)

    _save_attrs = [*BaseAnalyzer._save_attrs, "bleach_threshold"]
