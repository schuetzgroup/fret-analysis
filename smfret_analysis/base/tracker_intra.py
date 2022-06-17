# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Tuple, Union

import pandas as pd
from sdt import brightness, spatial
import trackpy

from .tracker_base import BaseTracker


class IntramolecularTracker(BaseTracker):
    def track_video(self, loc_data: Dict[str, pd.DataFrame]):
        """Track smFRET data

        Localization data for both the donor and the acceptor channel is
        merged (since a FRET construct has to be visible in at least one
        channel). The merged data is than linked into trajectories using
        py:func:`trackpy.link_df`. For this the :py:mod:`trackpy` package needs
        to be installed.
        Trajectories are identified by unique IDs in the ``("fret",
        "particle")`` column.

        Parameters
        ----------
        loc_data
            ``"donor"`` and ``"acceptor"`` keys map to localization data upon
            donor and acceptor excitation, respectively. This is typically
            ``self.sm_data[dataset_key][file_id]`` for some ``dateset_key``
            and ``file_id``.
        """
        # Create DataFrame for tracking
        columns = [("acceptor", "x"), ("acceptor", "y"), ("fret", "frame")]
        da_loc = pd.concat([loc_data[c][columns].droplevel(0, axis=1)
                            for c in ("donor", "acceptor")],
                           ignore_index=True)
        # Preserve indices so that new data can be assigned to original
        # DataFrames later
        da_loc["d_index"] = -1
        da_loc.iloc[:len(loc_data["donor"]), -1] = loc_data["donor"].index
        da_loc["a_index"] = -1
        da_loc.iloc[len(loc_data["donor"]):, -1] = loc_data["acceptor"].index

        da_loc["frame"] = self.frame_selector.renumber_frames(
            da_loc["frame"], "da")

        trc = trackpy.link(da_loc, **self.link_options)

        # Append new columns to localization data
        for chan in "donor", "acceptor":
            idx_col = f"{chan[0]}_index"
            t = trc[trc[idx_col] >= 0]
            loc_data[chan]["fret", "particle"] = pd.Series(
                t["particle"].to_numpy(), index=t[idx_col].to_numpy())

    def interpolate_missing_video(self, source: Union[str, Tuple[str, str]],
                                  loc_data: Dict[str, pd.DataFrame]):
        cols = [("acceptor", "x"), ("acceptor", "y"), ("fret", "frame"),
                ("fret", "particle")]
        da_loc = pd.concat(
            [loc_data[ch][cols] for ch in ("donor", "acceptor")],
            ignore_index=True).droplevel(0, axis=1)
        da_loc = spatial.interpolate_coords(da_loc)
        interp_mask = da_loc["interp"] > 0
        if not interp_mask.any():
            return
        nd = self._get_neighbor_distance()
        if nd > 0:
            spatial.has_near_neighbor(da_loc, nd)
        else:
            da_loc["has_neighbor"] = 0

        da_loc_interp = da_loc[interp_mask]

        d_loc = self.registrator(da_loc_interp, channel=2)
        a_loc = da_loc_interp.copy()

        opened = []
        try:
            im_seq, opened = self._open_image_sequence(source)
            b_opts, b_filt = self._get_brightness_options_filter()
            brightness.from_raw_image(d_loc, b_filt(im_seq["donor"]), **b_opts)
            brightness.from_raw_image(a_loc, b_filt(im_seq["acceptor"]),
                                      **b_opts)
        finally:
            for o in opened:
                o.close()

        ex_cols = ["x", "y", "mass", "signal", "bg", "bg_dev"]
        ex_loc = pd.concat({"donor": d_loc[ex_cols],
                            "acceptor": a_loc[ex_cols],
                            "fret": d_loc[["frame", "particle",
                                           "has_neighbor", "interp"]]},
                           axis=1)
        for ch in "donor", "acceptor":
            loc_data[ch]["fret", "interp"] = 0
            ld = pd.concat(
                [loc_data[ch], self.frame_selector.select(
                    ex_loc, ch[0], columns={"time": ("fret", "frame")})],
                ignore_index=True)
            ld.sort_values([("fret", "particle"), ("fret", "frame")],
                           ignore_index=True, inplace=True)
            loc_data[ch] = ld
