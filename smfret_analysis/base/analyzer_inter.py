# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base class for analyzing and filtering intermolecular smFRET data"""
from typing import Sequence, Union

import numpy as np
import pandas as pd
from sdt import helper

from .analyzer_base import BaseAnalyzer


class IntermolecularAnalyzer(BaseAnalyzer):
    _app_vals_columns = [("fret", "d_particle"), ("fret", "a_particle")]

    def calc_leakage_from_unbound(
            self, datasets: Union[str, Sequence[str], None] = None,
            print_summary: bool = False):
        """Calculate leakage correction from non-codiffusing donor traces

        Parameters
        ----------
        datasets
            dataset(s) to use. If `None`, use all..
        print_summary
            Print number of datapoints and result.

        See also
        --------
        calc_leakage
        """
        def selector(trc):
            # search for donor excitation with no codiffusing acceptor
            return ((trc["fret", "has_neighbor"] == 0) &
                    (trc["fret", "d_particle"] >= 0) &
                    (trc["fret", "a_particle"] < 0))
        self._calc_leakage_from_data(datasets, selector, print_summary)

    def calc_direct_excitation_from_unbound(
            self, datasets: Union[str, Sequence[str], None] = None,
            print_summary: bool = False):
        """Calculate dir. exc. corr. factor from non-codiffusing acc. traces

        Parameters
        ----------
        datasets
            dataset(s) to use. If `None`, use all.
        print_summary
            Print number of datapoints and result.

        See also
        --------
        calc_direct_excitation
        """
        def selector(trc):
            # search for acceptor excitation with no codiffusing donor
            return ((trc["fret", "has_neighbor"] == 0) &
                    (trc["fret", "d_particle"] < 0) &
                    (trc["fret", "a_particle"] >= 0))
        self._calc_direct_excitation_from_data(datasets, selector,
                                               print_summary)

    def _calc_detection_eff_single(self, tracks, eff_app_thresh, min_seg_len,
                                   stat):
        gammas = pd.Series(
            np.NaN, index=tracks["donor"]["fret", "d_particle"].unique())

        trc = self._apply_filters(tracks["donor"])
        # When donor and acceptor dissociate, they will still be in close
        # proximity, therefore discard events with near neighbors
        # Also, needs the donor present
        trc = trc[(trc["fret", "has_neighbor"] == 0) &
                  np.isfinite(trc["donor", "mass"]) &
                  np.isfinite(trc["acceptor", "mass"]) &
                  (trc["fret", "d_particle"] >= 0)].sort_values(
            [("fret", "d_particle"), ("fret", "frame")])
        trc_split = helper.split_dataframe(
            trc, ("fret", "d_particle"),
            [("donor", "mass"), ("acceptor", "mass"), ("fret", "eff_app"),
             ("fret", "particle")],
            type="array_list", sort=False)
        for p, t in trc_split:
            is_assoc = t[3] >= 0
            eff_mask = t[2] >= eff_app_thresh

            dis_mask = ~is_assoc
            i_dd_dis = t[0][dis_mask]
            i_da_dis = t[1][dis_mask]
            as_mask = is_assoc & eff_mask
            i_dd_as = t[0][as_mask]
            i_da_as = t[1][as_mask]

            if len(i_dd_dis) < min_seg_len or len(i_dd_as) < min_seg_len:
                continue

            gammas[p] = ((stat(i_da_as) - stat(i_da_dis)) /
                         (stat(i_dd_dis) - stat(i_dd_as)))

        return gammas

    _detection_eff_particle_column = ("fret", "d_particle")

    def _excitation_eff_filter(self, d: pd.DataFrame) -> pd.Series:
        return (d["fret", "particle"] >= 0) & (d["fret", "has_neighbor"] == 0)
