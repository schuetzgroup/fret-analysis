# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Module containing base class for analyzing and filtering smFRET data"""
from collections import defaultdict
import contextlib
import itertools
import numbers
from typing import (Any, Callable, Dict, Iterable, Literal, Mapping, Optional,
                    Sequence, Tuple, Union)

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sdt import changepoint, flatfield, helper, roi

from ..data_store import DataStore


def gaussian_mixture_split(data: pd.DataFrame, n_components: int,
                           columns: Sequence[Tuple[str, str]] = [
                               ("fret", "eff_app"), ("fret", "stoi_app")],
                           random_seed: int = 0
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Fit Gaussian mixture model and predict component for each particle

    First, all datapoints are used to fit a Gaussian mixture model. Then each
    particle is assigned the component in which most of its datapoints lie.

    This requires scikit-learn (sklearn).

    Parameters
    ----------
    data
        Single-molecule FRET data
    n_components
        Number of components in the mixture
    columns
        Which columns to fit.
    random_seed
        Seed for the random number generator used to initialize the Gaussian
        mixture model fit.

    Returns
    -------
    Component label for each entry in `data` and mean values for each
    component, one line per component.
    """
    from sklearn.mixture import GaussianMixture

    rs = np.random.RandomState(random_seed)
    d = data.loc[:, columns].values
    valid = np.all(np.isfinite(d), axis=1)
    d = d[valid]
    gmm = GaussianMixture(n_components=n_components, random_state=rs).fit(d)
    labels = gmm.predict(d)
    mean_sort_idx = np.argsort(gmm.means_[:, 0])[::-1]
    sorter = np.argsort(mean_sort_idx)
    labels = sorter[labels]  # sort according to descending mean
    return labels, gmm.means_[mean_sort_idx]


class Analyzer:
    """Class for analyzing and filtering of smFRET data

    This provides various analysis and filtering methods which act on the
    :py:attr:`tracks` attribute.

    Correction of FRET efficiencies and stoichiometries for donor leakage and
    direct acceptor excitation is implemented according to [Hell2018]_,
    while different detection efficiencies for the different
    fluorophores are accounted for as described in [MacC2010]_ as well
    the different excitation efficiencies according to [Lee2005]_.
    """

    bleach_thresh: Sequence[float] = (500.0, 500.0)
    """Intensity (mass) thresholds upon donor and acceptor excitation,
    respecitively, below which a signal is considered bleached. Used for
    bleaching step analysis.
    """
    sm_data: Dict[str, Dict[Literal["donor", "acceptor"], pd.DataFrame]]
    """smFRET tracking data, mapping dataset -> excitation channel -> data"""
    special_sm_data: Dict[str, Dict[Literal["donor", "acceptor"],
                                    pd.DataFrame]]
    """smFRET tracking data for special purposes, such as correction factor
    calculation. Maps dataset -> excitation channel -> data
    """
    flatfield: Dict[Literal["donor", "acceptor"], flatfield.Corrector]
    """channel name -> flatfield correction class instance"""
    cp_detector: Any
    """Changepoint detector class instance used to perform bleching step
    detection.
    """
    leakage: float = 0.0
    r"""Correction factor for donor leakage into the acceptor channel;
    :math:`\alpha` in [Hell2018]_
    """
    direct_excitation: float = 0.0
    r"""Correction factor for direct acceptor excitation by the donor
    laser; :math:`\delta` in [Hell2018]_
    """
    detection_eff: float = 1.0
    r"""Correction factor(s) for the detection efficiency difference
    beteen donor and acceptor fluorophore; :math:`\gamma` in [Hell2018].

    Can be a scalar for global correction or a :py:class:`pandas.Series`
    for individual correction. In the latter case, the index is the
    particle number and the value is the correction factor.
    """
    excitation_eff: float = 1.0
    r"""Correction factor(s) for the excitation efficiency difference
    beteen donor and acceptor fluorophore; :math:`\beta` in [Hell2018].
    """

    def __init__(self, cp_detector: Optional[Any] = None):
        """Parameters
        ----------
        cp_detector
            Changepoint detetctor. If `None`, create a
            :py:class:`changepoint.Pelt` instance with ``model="l2"``,
            ``min_size=1``, and ``jump=1``.
        """
        self.sm_data = {}
        self.special_sm_data = {}

        if cp_detector is None:
            cp_detector = changepoint.Pelt("l2", min_size=1, jump=1)
        self.cp_detector = cp_detector

        self._reason_counter = defaultdict(lambda: 0)

    @staticmethod
    def _apply_filters(tracks: pd.DataFrame, include_negative: bool = False,
                       ignore: Union[str, Sequence[str]] = [],
                       type: str = "data", skip_neighbors: bool = True
                       ) -> Union[pd.DataFrame, np.array]:
        """Apply filters to a single DataFrame containing smFRET tracking data

        This removes all entries from `tracks` that have been marked
        as filtered via a ``"filter"`` column.

        Parameters
        ----------
        include_negative
            If `False`, include only entries for which all ``"filter"`` column
            values are zero. If `True`, include also entries with negative
            ``"filter"`` column values.
        ignore
            ``"filter"`` column(s) to ignore when deciding whether to include
            an entry or not. For instance, setting ``ignore="bleach_step"``
            will not consider the ``("filter", "bleach_step")`` column values.
        type
            If ``"data"``, return a copy of `tracks` excluding all entries that
            have been marked as filtered, i.e., that have a positive (or
            nonzero, see the `include_negative` parameter) entry in any
            ``"filter"`` column. If ``"mask"``, return a boolean array
            indicating whether an entry is to be removed or not
        skip_neighbors
            If `True`, remove localizations where ``("fret", "has_neighbor")``
            is `True`.

        Returns
        -------
        Copy of `tracks` with all filtered rows removed or corresponding
        boolean mask.
        """
        n_mask = np.ones(len(tracks), dtype=bool)

        if skip_neighbors and ("fret", "has_neighbor") in tracks:
            n_mask &= tracks["fret", "has_neighbor"] <= 0

        if "filter" not in tracks:
            if type == "data":
                return tracks[n_mask].copy()
            return n_mask

        flt = tracks["filter"].drop(ignore, axis=1)
        if include_negative:
            mask = flt > 0
        else:
            mask = flt != 0
        mask = ~np.any(mask, axis=1) & n_mask

        if type == "data":
            return tracks[mask].copy()
        return mask

    @classmethod
    def _reset_filters_single(cls, tracks: pd.DataFrame,
                              keep: Union[str, Iterable[str]] = []):
        if "filter" not in tracks:
            return
        if isinstance(keep, str):
            keep = [keep]
        cols = tracks.columns.get_loc_level("filter")[1]
        rm_cols = np.setdiff1d(cols, keep)
        tracks.drop(columns=[("filter", c) for c in rm_cols],
                    inplace=True)

    def reset_filters(self, keep: Union[str, Iterable[str]] = []):
        """Reset filters

        This drops filter columns from :py:attr:`tracks`.

        Parameters
        ----------
        keep
            Filter column name(s) to keep
        """
        # iterate datasets
        for dset in itertools.chain(self.sm_data.values(),
                                    self.special_sm_data.values()):
            # iterate files
            for d in dset.values():
                # iterate excitation channels
                for sub in d.values():
                    self._reset_filters_single(sub, keep)

    @classmethod
    def _calc_apparent_values_single(
            cls, tracks: Mapping[Literal["donor", "acceptor"], pd.DataFrame],
            a_mass_interp: str, skip_neighbors: bool):

        d_tracks = tracks["donor"]
        a_tracks = tracks["acceptor"]

        # calculate total mass upon acceptor excitation, interpolate to
        # donor excitation frames
        a_mask = cls._apply_filters(a_tracks, type="mask",
                                    skip_neighbors=skip_neighbors)
        a_filtered = a_tracks[a_mask].sort_values([("fret", "particle"),
                                                   ("fret", "frame")])

        a_split = dict(helper.split_dataframe(
                a_filtered, ("fret", "particle"),
                [("acceptor", "mass"), ("fret", "frame")],
                type="array_list", sort=False))

        a_mass = []
        for p, (idx, d_frames) in helper.split_dataframe(
                d_tracks.sort_values([("fret", "particle"),
                                      ("fret", "frame")]),
                ("fret", "particle"), [("fret", "frame")],
                type="array_list", keep_index=True, sort=False):
            a_direct, a_frames = a_split.get(p, ([], []))
            if len(a_direct) == 0:
                # No direct acceptor excitation, cannot do anything
                a_mass.append(pd.Series(np.NaN, index=idx))
                continue
            elif len(a_direct) == 1:
                # Only one direct acceptor excitation; use this value
                # for all data points of this particle
                a_mass.append(pd.Series(a_direct[0], index=idx))
                continue
            else:
                # Enough direct acceptor excitations for interpolation
                a_mass_func = interp1d(
                    a_frames, a_direct, a_mass_interp, copy=False,
                    fill_value=(a_direct[0], a_direct[-1]),
                    assume_sorted=True, bounds_error=False)
                # Calculate (interpolated) mass upon direct acceptor
                # excitation
                a_mass.append(pd.Series(a_mass_func(d_frames), index=idx))
        a_mass = pd.concat(a_mass)

        # Total mass upon donor excitation
        d_mass = d_tracks["donor", "mass"] + d_tracks["acceptor", "mass"]

        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore divide by zero and 0 / 0
            # FRET efficiency
            eff = d_tracks["acceptor", "mass"] / d_mass
            # FRET stoichiometry
            stoi = d_mass / (d_mass + a_mass)

        d_tracks["fret", "eff_app"] = eff
        d_tracks["fret", "stoi_app"] = stoi
        d_tracks["fret", "d_mass"] = d_mass
        d_tracks["fret", "a_mass"] = a_mass

    def calc_apparent_values(self, a_mass_interp: str = "nearest-up",
                             skip_neighbors: bool = True):
        r"""Calculate apparent, FRET-related values

        This needs to be called before the filtering methods and before
        calculating the true FRET efficiencies and stoichiometries. However,
        any corrections to the donor and acceptor localization data (such as
        :py:meth:`flatfield_correction`) need to be done before this.

        Calculated values apparent FRET efficiencies and stoichiometries,
        the total brightness (mass) upon donor excitation, and the acceptor
        brightness (mass) upon direct excitation, which is interpolated for
        donor excitation datapoints in order to allow for calculation of
        stoichiometries.

        For each localization in each track, the total brightness upon donor
        excitation is calculated by taking the sum of ``("donor", "mass")``
        and ``("acceptor", "mass")`` values. It is added as a
        ``("fret", "d_mass")`` column to the :py:attr:`sm_data` DataFrames. The
        apparent FRET efficiency (acceptor brightness (mass) divided by sum of
        donor and acceptor brightnesses) is added as a
        ``("fret", "eff_app")`` column to the :py:attr:`sm_data` DataFrames.

        The apparent stoichiometry value :math:`S_\text{app}` is given as

        .. math:: S_\text{app} = \frac{I_{DD} + I_{DA}}{I_{DD} + I_{DA} +
            I_{AA}}

        as in [Hell2018]_. :math:`I_{DD}` is the donor brightness upon donor
        excitation, :math:`I_{DA}` is the acceptor brightness upon donor
        excitation, and :math:`I_{AA}` is the acceptor brightness upon
        acceptor excitation. The latter is calculated by interpolation for
        frames with donor excitation.

        :math:`I_{AA}` is append as a ``("fret", "a_mass")`` column.
        The stoichiometry value is added in the ``("fret", "stoi_app")``
        column.

        Parameters
        ----------
        a_mass_interp
            How to interpolate the acceptor mass upon direct excitation in
            donor excitation frames. Sensible values are "linear" for linear
            interpolation; "nearest" to take the value of the closest
            direct acceptor excitation frame (using the previous frame in case
            of a tie); "nearest-up", which is similar to "nearest" but takes
            the next frame in case of a tie; "next" and "previous" to use the
            next and previous frames, respectively.
        skip_neighbors
            If `True`, skip localizations where ``("fret", "has_neighbor")`` is
            `True` when interpolating acceptor mass upon direct excitation.
        """
        # iterate datasets
        for dset in itertools.chain(self.sm_data.values(),
                                    self.special_sm_data.values()):
            # iterate files
            for d in dset.values():
                self._calc_apparent_values_single(d, a_mass_interp,
                                                  skip_neighbors)

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

    def _bleaches(self, steps: Sequence[float], channel: int) -> bool:
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
                all(s < self.bleach_thresh[channel] for s in steps[1:]))

    def _bleaches_partially(self, steps: Sequence[float], channel: int
                            ) -> bool:
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
                any(s > self.bleach_thresh[channel] for s in steps[1:]))

    def _increment_reason_counter(self, reason: str) -> int:
        self._reason_counter[reason] += 1
        return self._reason_counter[reason]

    @staticmethod
    def _update_filter(tracks: pd.DataFrame, flt: np.ndarray, reason: str,
                       cnt: int):
        """Update a filter column

        If it does not exist yet, append to :py:attr:`sm_data` DataFrames.
        Otherwise, each entry is updated as follows
        - If ``-1`` before, use the new value
        - If ``0`` before, leave at 0 if the new value is 0. Set to the
          appropriate reason count (i.e., ``1`` if the filter reason is used
          for the first time, ``2`` if used for the second time, and so on).
        - If greater than ``0`` before, leave as is.

        Parameters
        ----------
        tracks
            Single-molecule data for which to update filter column
        flt
            New filter data, one value per line in :py:attr:`tracks`. ``-1``
            means no decision about filtering, ``0`` means that the entry
            is accepted, ``1`` means that the entry is rejected.
        reason
            Filtering reason / column name to use.
        cnt
            Reason use count
        """
        if ("filter", reason) not in tracks:
            tracks["filter", reason] = flt
        else:
            fr = tracks["filter", reason].to_numpy()
            old_good = fr <= 0
            tracks.loc[old_good & (flt > 0), ("filter", reason)] = cnt
            tracks.loc[old_good & (flt == 0), ("filter", reason)] = 0

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
                is_good = (self._bleaches(stat_d, 0) and not
                           self._bleaches_partially(stat_a, 1))
            elif condition == "acceptor":
                is_good = (self._bleaches(stat_a, 1) and not
                           self._bleaches_partially(stat_d, 0))
            elif condition in ("donor or acceptor", "acceptor or donor"):
                is_good = ((self._bleaches(stat_d, 0) and not
                            self._bleaches_partially(stat_a, 1)) or
                           (self._bleaches(stat_a, 1) and not
                            self._bleaches_partially(stat_d, 0)))
            elif condition == "no partial":
                is_good = not (self._bleaches_partially(stat_d, 0) or
                               self._bleaches_partially(stat_a, 1))
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
        is below :py:attr:`bleach_thresh` ``[0]``. The acceptor considered
        bleached if its ``("fret", f"a_seg_{stat}")`` is below
        :py:attr:`bleach_thresh` ``[0]``

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

        >>> ana.bleach_thresh = (800, 500)
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

    @staticmethod
    def _eval(data: pd.DataFrame, expr: str, mi_sep: str = "_"):
        """Call ``eval(expr)`` for `data`

        Flatten the column MultiIndex and call the resulting DataFrame's
        `eval` method.

        Parameters
        ----------
        data
            Data frame
        expr
            Argument for eval. See :py:meth:`pandas.DataFrame.eval` for
            details.
        mi_sep
            Use this to separate levels when flattening the column
            MultiIndex. Defaults to "_".

        Returns
        -------
        pandas.Series, dtype(bool)
            Boolean Series indicating whether an entry fulfills `expr` or not.

        Examples
        --------
        Get a boolean array indicating lines where ("fret", "a_mass") <= 500
        in :py:attr:`tracks`

        >>> filt._eval(filt.tracks, "fret_a_mass > 500")
        0     True
        1     True
        2    False
        dtype: bool
        """
        if not len(data):
            return pd.Series([], dtype=bool)

        old_columns = data.columns
        try:
            data.columns = helper.flatten_multiindex(old_columns, mi_sep)
            e = data.eval(expr)
        except Exception:
            raise
        finally:
            data.columns = old_columns

        return e

    def query(self, expr: str, mi_sep: str = "_", reason: str = "query"):
        """Filter features according to column values

        Flatten the column MultiIndex and filter the resulting DataFrame's
        `eval` method.

        Parameters
        ----------
        expr
            Filter expression. See :py:meth:`pandas.DataFrame.eval` for
            details.
        mi_sep
            Separate multi-index levels by this character / string.
        reason
            Filtering reason / column name to use.

        Examples
        --------
        Remove lines where ("fret", "a_mass") <= 500 from :py:attr:`tracks`

        >>> filt.query("fret_a_mass > 500")
        """
        reason_cnt = self._increment_reason_counter(reason)

        # iterate datasets
        for dset in itertools.chain(self.sm_data.values(),
                                    self.special_sm_data.values()):
            # iterate files
            for d in dset.values():
                for sub in d.values():
                    filtered = self._eval(sub, expr, mi_sep)
                    filtered = np.asarray(~filtered, dtype=np.intp)
                    self._update_filter(sub, filtered, reason, reason_cnt)

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

    def _image_mask_single(self, tracks: Mapping[Literal["donor", "acceptor"],
                                                 pd.DataFrame],
                           masks: Iterable[Mapping],
                           channel: Literal["donor", "acceptor"],
                           reason: str, reason_cnt: int):
        cols = {"coords": [(channel, c) for c in ("x", "y")]}

        for sub in tracks.values():
            flt = pd.Series(-1, dtype=np.intp, index=sub.index)
            for m in masks:
                r = roi.MaskROI(m["mask"])

                t_mask = np.ones(len(sub), dtype=bool)
                start = m.get("start")
                if start is not None:
                    t_mask &= sub["fret", "frame"] >= start
                end = m.get("end")
                if end is not None:
                    t_mask &= sub["fret", "frame"] < end

                sub_t = sub[t_mask]
                flt[sub_t.index] = np.maximum(
                    ~r.dataframe_mask(sub_t, columns=cols),
                    flt[sub_t.index])
            self._update_filter(sub, flt, reason, reason_cnt)

    def image_mask(self,
                   mask: Union[np.ndarray,
                               Dict[str, Mapping[int, Iterable[Mapping]]]],
                   channel: str, reason: str = "image_mask"):
        """Filter using a boolean mask image

        Remove all lines where coordinates lie in a region where `mask` is
        `False`.

        Parameters
        ----------
        mask
            Mask image(s). If this is a single array, apply it to the whole
            all data in :py:attr:`sm_data` and :py:attr:`special_sm_data`.

            To apply different masks to different data (e.g. from image
            segmentation), create dicts mapping `dataset key` -> `file id` ->
            list of dicts describing the mask. The latter shall contain a
            ``"mask"`` entry holding a boolean array representing the mask, and
            optionally ``"start"`` and ``"end"`` keys holding describing to
            which frames the mask applies (``"start"`` is inclusive, ``"end"``
            is exclusive)..
        channel
            Channel to use for the filtering
        reason
            Filtering reason / column name to use.

        Examples
        --------
        Create a 2D boolean mask to remove any features that do not have
        x and y coordinates between 50 and 100 in the donor channel.

        >>> mask = numpy.zeros((200, 200), dtype=bool)
        >>> mask[50:100, 50:100] = True
        >>> filt.image_mask(mask, "donor")

        If :py:attr:`tracks` has a MultiIndex index, where e.g. the first
        level is "file1", "file2", … and different masks should be applied
        for each file, this is possible by passing a list of
        dicts. Furthermore, we can apply one mask to all frames up to 100 and
        another to the rest:

        >>> masks = {"dataset1": {0: [{"mask": np.array(...), "end": 10},
        ...                           {"mask": np.array(...), "start": 10,
        ...                            "end": 20},
        ...                           {"mask": np.array(...), "start": 20}],
        ...                       1: [{"mask": np.array(...)}]},
        ...          "dataset1": {0: [{"mask": np.array(...)}]}}
        >>> filt.image_mask(masks, "donor")
        """
        reason_cnt = self._increment_reason_counter(reason)

        if isinstance(mask, dict):
            datasets = self.sm_data.items()
            mask_dict = mask
        else:
            datasets = itertools.chain(self.sm_data.items(),
                                       self.special_sm_data.items())
            mask_dict = defaultdict(
                lambda: defaultdict(lambda: [{"mask": mask}]))

        # iterate datasets
        for dname, dset in datasets:
            try:
                dmask = mask_dict[dname]
            except KeyError:
                dmask = {}
            # iterate files
            for fid, d in dset.items():
                try:
                    dm = dmask[fid]
                except KeyError:
                    dm = []
                self._image_mask_single(d, dm, channel, reason, reason_cnt)

    def _flatfield_correction_single(self, tracks):
        dest_cols = list(itertools.product(
            ("donor", "acceptor"), ("mass", "signal")))
        src_cols = [(c[0], f"{c[1]}_pre_flat") for c in dest_cols]

        for chan in "donor", "acceptor":
            sdc = tracks[chan]

            for src, dest in zip(src_cols, dest_cols):
                # If source columns (i.e., "{col}_pre_flat") do not exist,
                # create them
                if src not in sdc:
                    sdc[src] = sdc[dest]

            coord_cols = [(chan, "x"), (chan, "y")]
            c = self.flatfield[chan](
                sdc, columns={"coords": coord_cols, "corr": src_cols})

            for src, dest in zip(src_cols, dest_cols):
                sdc[dest] = c[src]

    def flatfield_correction(self):
        """Apply flatfield correction to donor and acceptor localization data

        THis make use of :py:attr:`flatfield`.

        If present, donor and acceptor ``"mass_pre_flat"`` and
        ``"signal_pre_flat"`` columns are used as inputs for flatfield
        correction, results are written to ``"mass"`` and `"signal"`` columns.
        Otherwise, ``"mass"`` and `"signal"`` columns are copied to
        ``"mass_pre_flat"`` and ``"signal_pre_flat"`` first.

        Any values derived from those (e.g., apparent FRET efficiencies) need
        to be recalculated manually.
        """
        # iterate datasets
        for dset in itertools.chain(self.sm_data.values(),
                                    self.special_sm_data.values()):
            # iterate files
            for d in dset.values():
                self._flatfield_correction_single(d)

    def calc_leakage(self):
        r"""Calculate donor leakage (bleed-through) into the acceptor channel

        For this to work, :py:attr:`tracks` must be a dataset of donor-only
        molecules. In this case, the leakage :math:`alpha` can be
        computed using the formula [Hell2018]_

        .. math:: \alpha = \frac{\langle E_\text{app}\rangle}{1 -
            \langle E_\text{app}\rangle},

        where :math:`\langle E_\text{app}\rangle` is the mean apparent FRET
        efficiency of a donor-only population.

        The leakage :math:`\alpha` together with the direct acceptor excitation
        :math:`\delta` can be used to calculate the real fluorescence due to
        FRET,

        .. math:: F_\text{DA} = I_\text{DA} - \alpha I_\text{DD} - \delta
            I_\text{AA}.

        This sets the :py:attr:`leakage` attribute.
        See :py:meth:`fret_correction` for how use this to calculate corrected
        FRET values.
        """
        eff = []
        for d in self.special_sm_data["donor-only"].values():
            m = self._apply_filters(d["donor"], type="mask")
            eff.append(d["donor"].loc[m, ("fret", "eff_app")].to_numpy())
        m_eff = np.nanmean(np.concatenate(eff))
        self.leakage = m_eff / (1 - m_eff)

    def calc_direct_excitation(self):
        r"""Calculate direct acceptor excitation by the donor laser

        For this to work, :py:attr:`tracks` must be a dataset of acceptor-only
        molecules. In this case, the direct acceptor excitation :math:`delta`
        can be computed using the formula [Hell2018]_

        .. math:: \alpha = \frac{\langle S_\text{app}\rangle}{1 -
            \langle S_\text{app}\rangle},

        where :math:`\langle ES\text{app}\rangle` is the mean apparent FRET
        stoichiometry of an acceptor-only population.

        The leakage :math:`\alpha` together with the direct acceptor excitation
        :math:`\delta` can be used to calculate the real fluorescence due to
        FRET,

        .. math:: F_\text{DA} = I_\text{DA} - \alpha I_\text{DD} - \delta
            I_\text{AA}.

        This sets the :py:attr:`direct_excitation` attribute.
        See :py:meth:`fret_correction` for how use this to calculate corrected
        FRET values.
        """
        stoi = []
        for d in self.special_sm_data["acceptor-only"].values():
            m = self._apply_filters(d["donor"], type="mask")
            stoi.append(d["donor"].loc[m, ("fret", "stoi_app")].to_numpy())
        m_stoi = np.nanmean(np.concatenate(stoi))
        self.direct_excitation = m_stoi / (1 - m_stoi)

    def _calc_detection_eff_single(self, tracks, min_seg_len, stat):
        gammas = pd.Series(np.NaN,
                           index=tracks["donor"]["fret", "particle"].unique())

        trc = self._apply_filters(tracks["donor"]).sort_values(
            [("fret", "particle"), ("fret", "frame")])
        trc_split = helper.split_dataframe(
            trc, ("fret", "particle"),
            [("donor", "mass"), ("acceptor", "mass"), ("fret", "a_seg")],
            type="array_list", sort=False)
        for p, t in trc_split:
            fin_mask = np.isfinite(t[0]) & np.isfinite(t[1])
            pre_mask = (t[2] == 0) & fin_mask
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

    def calc_detection_eff(self,
                           aggregate: Literal["all", "dataset",
                                              "individual"] = "all",
                           aggregate_stat: Callable[[np.array], float
                                                    ] = np.nanmedian,
                           seg_stat: Callable[[np.array], float] = np.median,
                           min_seg_len: int = 5,
                           datasets: Optional[Union[Iterable[str], str]
                                              ] = None):
        r"""Calculate detection efficiency ratio of dyes

        The detection efficiency ratio is the ratio of decrease in acceptor
        brightness to the increase in donor brightness upon acceptor
        photobleaching [MacC2010]_:

        .. math:: \gamma = \frac{\langle I_\text{DA}^\text{pre}\rangle -
            \langle I_\text{DA}^\text{post}\rangle}{
            \langle I_\text{DD}^\text{post}\rangle -
            \langle I_\text{DD}^\text{pre}\rangle}

        This needs molecules with exactly one donor and one acceptor
        fluorophore to work. Tracks need to be segmented already (see
        :py:meth:`segment_a_mass`).

        The correction can be calculated for each track individually or some
        statistic (e.g. the median) of the indivdual :math:`gamma` values can
        be used as a global correction factor for all tracks.

        The detection efficiency :math:`\gamma` can be used to calculate the
        real fluorescence of the donor fluorophore,

        .. math:: F_\text{DD} = \gamma I_\text{DD}.

        This sets the :py:attr:`detection_eff` attribute.
        See :py:meth:`fret_correction` for how use this to calculate corrected
        FRET values.

        Parameters
        ----------
        min_seg_len
            How many data points need to be present before and after the
            bleach step to ensure a reliable calculation of the mean
            intensities. If there are fewer data points, a value of NaN will be
            assigned.
        aggregate
            If ``"individual"``, the :math:`\gamma` value for each track is
            stored and used to correct the values individually when calling
            :py:meth:`fret_correction`. If ``"dataset"`` or ``"all"``,
            calculate one correction factor per dataset or for all data,
            respectively by calling `aggregate_stat` on per-track values.

            If a function, apply this function
            to the :math:`\gamma` array and its return value as a global
            correction factor. A sensible example for such a function would be
            :py:func:`numpy.nanmean`. Beware that some :math:`\gamma` may be
            NaN.
        seg_stat
            Statistic to use to determine fluorescence intensity before and
            after acceptor photobleaching.
        aggregate_stat
            If `aggregate` is not ``"individual"``, apply this function to
            per-track :math:`\gamma` values to determine the per-dataset or
            global value. The function should be able to deal with NaNs.
        datasets
            If `aggregate` is not ``"individual"``, specify  datasets to
            compute correction factors for all datasets. If `None`, use all
            datasets.
        """
        if isinstance(datasets, str):
            datasets = (datasets,)
        if aggregate == "all" and datasets is not None:
            datasets = ((k, self.sm_data[k]) for k in datasets)
        else:
            datasets = self.sm_data.items()

        gammas = {}
        for k, dset in datasets:
            g = {}
            for fid, d in dset.items():
                g[fid] = self._calc_detection_eff_single(
                    d, min_seg_len, seg_stat)
            gammas[k] = g

        if aggregate == "individual":
            self.detection_eff = gammas
        elif aggregate == "dataset":
            self.detection_eff = {k: aggregate_stat(np.concat(v.values()))
                                  for k, v in gammas.items()}
        else:
            g = [v2 for v in gammas.values() for v2 in v.values()]
            self.detection_eff = aggregate_stat(pd.concat(g))

    def calc_excitation_eff(self, dataset: str, n_components: int = 1,
                            component: int = 0, random_seed: int = 0):
        r"""Calculate excitation efficiency ratio of dyes

        This is a measure of how efficient the direct acceptor excitation is
        compared to the donor excitation. It depends on the fluorophores and
        also on the excitation laser intensities.

        It can be calculated using the formula [Lee2005]_

        .. math:: \beta = \frac{1 - \langle S_\gamma \rangle}{
            \langle S_\gamma\rangle},

        where :math:`S_\gamma` is calculated like the apparent stoichiometry,
        but with the donor and acceptor fluorescence upon donor excitation
        already corrected using the leakage, direct excitation, and
        detection efficiency factors.

        This needs molecules with exactly one donor and one acceptor
        fluorophore to work. Tracks need to be segmented already (see
        :py:meth:`segment_a_mass`). The :py:attr:`leakage`,
        :py:attr:`direct_excitation`, and :py:attr:`detection_eff` attributes
        need to be set correctly.

        The excitation efficiency :math:`\beta` can be used to correct the
        acceptor fluorescence upon acceptor excitation,

        .. math:: F_\text{AA} = I_\text{AA} / \beta.

        This sets the :py:attr:`excitation_eff` attribute.
        See :py:meth:`fret_correction` for how use this to calculate corrected
        FRET values.

        Parameters
        ----------
        dataset
            Identifier of dataset to use for correction factor calculation
        n_components
            If > 1, perform a Gaussian mixture fit on the 2D apparent
            efficiency-vs.-stoichiomtry dataset. This helps to choose only the
            correct component with one donor and one acceptor. Defaults to 1.
        component
            If n_components > 1, use this to choos the component number.
            Components are ordered according to decreasing mean apparent FRET
            efficiency. :py:func:`gaussian_mixture_split` can be used to
            check which component is the desired one. Defaults to 0.
        random_seed
            Seed for the random number generator used to initialize the
            Gaussian mixture model fit.
        """
        trc = {}
        for fid, d in self.sm_data[dataset].items():
            t = d["donor"]
            t = t[(t["fret", "a_seg"] == 0) & (t["fret", "d_seg"] == 0)]
            trc[fid] = {"donor": self._apply_filters(t)}

        if n_components > 1:
            s = pd.concat([t["donor"] for t in trc.values()],
                          keys=list(trc.keys()))
            split = gaussian_mixture_split(s, n_components,
                                           random_seed=random_seed)[0]
            for fid, comp in pd.Series(split, index=s.index).groupby(level=0):
                trc[fid]["donor"] = trc[fid]["donor"][
                    comp.to_numpy() == component]

        tmp_ana = self.__class__()
        tmp_ana.sm_data["detection_eff"] = trc
        tmp_ana.leakage = self.leakage
        tmp_ana.direct_excitation = self.direct_excitation
        tmp_ana.detection_eff = self.detection_eff
        tmp_ana.fret_correction()

        s_gamma = pd.concat(t["donor"]["fret", "stoi"] for t in trc.values()
                            ).mean()
        self.excitation_eff = (1 - s_gamma) / s_gamma

    def calc_detection_excitation_effs(
            self, n_components: int,
            components: Optional[Sequence[int]] = None,
            dataset: Optional[str] = None, random_seed: int = 0):
        r"""Get detection and excitation efficiency from multi-state sample

        States are found in efficiency-vs.-stoichiometry space using a
        Gaussian mixture fit. Detection efficiency factor :math:`\gamma` and
        excitation efficiency factor :math:`\delta` are found performing a
        linear fit to the equation

        .. math:: S^{-1} = 1 + \beta\gamma + (1 - \gamma)\beta E

        to the Gaussian mixture fit results, where :math:`S` are the
        components' mean stoichiometries (corrected for leakage and direct
        excitation) and :math:`E` are the corresponding FRET efficiencies
        (also corrected for leakage and direct excitation) [Hell2018]_.

        Parameters
        ----------
        n_components
            Number of components for Gaussian mixture model
        components
            List of indices of components to use for the linear fit. If `None`,
            use all.
        random_seed
            Seed for the random number generator used to initialize the
            Gaussian mixture model fit.
        """
        if dataset is None:
            a = self.special_sm_data["multi-state"]
        else:
            a = self.sm_data[dataset]

        trc = {}
        for fid, d in a.items():
            t = d["donor"]
            t = t[(t["fret", "a_seg"] == 0) & (t["fret", "d_seg"] == 0)]
            trc[fid] = {"donor": self._apply_filters(t)}

        tmp_ana = self.__class__()
        tmp_ana.sm_data["det_exc_eff"] = trc
        tmp_ana.leakage = self.leakage
        tmp_ana.direct_excitation = self.direct_excitation
        tmp_ana.fret_correction()

        trc = pd.concat((d["donor"][[("fret", "eff"), ("fret", "stoi")]]
                         for d in trc.values()),
                        ignore_index=True)

        split = gaussian_mixture_split(
            trc, n_components, columns=[("fret", "eff"), ("fret", "stoi")],
            random_seed=random_seed)[1]
        if components is None:
            components = slice(None)
        b, a = np.polyfit(split[components, 0], 1 / split[components, 1],
                          deg=1)
        self.detection_eff = (a - 1) / (a + b - 1)
        self.excitation_eff = a + b - 1

    def _fret_correction_single(self, tracks, gamma):
        t = tracks["donor"]

        if isinstance(gamma, pd.Series):
            gamma = gamma.reindex(t["fret", "particle"]).to_numpy()

        i_da = t["acceptor", "mass"]
        i_dd = t["donor", "mass"]
        i_aa = t["fret", "a_mass"]

        f_da = i_da - self.leakage * i_dd - self.direct_excitation * i_aa
        f_dd = i_dd * gamma
        f_aa = i_aa / self.excitation_eff

        t["fret", "f_da"] = f_da
        t["fret", "f_dd"] = f_dd
        t["fret", "f_aa"] = f_aa

        t["fret", "eff"] = f_da / (f_dd + f_da)
        t["fret", "stoi"] = (f_dd + f_da) / (f_dd + f_da + f_aa)

    def fret_correction(self):
        r"""Apply corrections to calculate real FRET-related values

        By correcting the measured acceptor and donor intensities upon
        donor excitation (:math:`I_\text{DA}` and :math:`I_\text{DD}`) and
        acceptor intensity upon acceptor excitation (:math:`I_\text{AA}`) for
        donor leakage into the acceptor channel :math:`\alpha`, acceptor
        excitation by the donor laser :math:`\delta`, detection efficiencies
        :math:`\gamma`, and excitation efficiencies :math:`\beta`
        using [Hell2018]_

        .. math:: F_\text{DA} &= I_\text{DA} - \alpha I_\text{DD} - \delta
            I_\text{AA} \\
            F_\text{DD} &= \gamma I_\text{DD} \\
            F_\text{AA} &= I_\text{AA} / \beta

        the real FRET efficiency and stoichiometry values can be calculated:

        .. math:: E &= \frac{F_\text{DA}}{F_\text{DA} + F_\text{DD}} \\
            S &=  \frac{F_\text{DA} + F_\text{DD}}{F_\text{DA} + F_\text{DD} +
            F_\text{AA}}

        :math:`F_\text{DA}` will be appended to :py:attr:`tracks` as the
        ``("fret", "f_da")`` column; :math:`F_\text{DD}` as
        ``("fret", "f_dd")``; :math:`F_\text{DA}` as ``("fret", "f_aa")``;
        :math:`E` as ``("fret", "eff")``; and :math:`S` as ``("fret",
        "stoi")``.
        """
        # iterate datasets
        for k, dset in self.sm_data.items():
            if isinstance(self.detection_eff, numbers.Number):
                gammas = self.detection_eff
            else:
                gammas = self.detection_eff[k]

            # iterate files
            for fid, d in dset.items():
                if isinstance(gammas, numbers.Number):
                    g = gammas
                else:
                    g = gammas[fid]
                self._fret_correction_single(d, g)

    @classmethod
    def load(cls, file_prefix: str = "tracking", reset_filters: bool = True
             ) -> "Analyzer":
        """Load data from disk

        Parameters
        ----------
        file_prefix
            Prefix of save files to load. Same as `file_prefix`` argument
            passed to :py:meth:`save`.
        reset_filters
            If `True`, reset filters in tracking data.

        Returns
        -------
        Class instance with data from save files
        """
        ds = DataStore.load(file_prefix)

        ret = cls()
        for key in ("sm_data", "special_sm_data"):
            with contextlib.suppress(AttributeError):
                setattr(ret, key, getattr(ds, key))
        if reset_filters:
            ret.reset_filters()
        return ret
