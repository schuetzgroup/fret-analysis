# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import itertools
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sdt import brightness, multicolor, spatial
import trackpy
import traitlets

from .tracker_base import BaseTracker


class IntermolecularTracker(BaseTracker):
    codiffusion_options: Dict[str, Any] = traitlets.Dict(
        default_value={"abs_threshold": 2, "rel_threshold": 0.0,
                       "max_dist": 2.0})

    def track_video(self, loc_data: Dict[str, pd.DataFrame]):
        acc_loc = loc_data["acceptor"][[("acceptor", "x"), ("acceptor", "y"),
                                        ("fret", "frame")]
                                       ].droplevel(0, axis=1)
        acc_loc.reset_index(inplace=True)
        acc_loc = self.frame_selector.select(acc_loc, "a", renumber=True)
        don_loc = loc_data["donor"][[("acceptor", "x"), ("acceptor", "y"),
                                     ("fret", "frame")]
                                    ].droplevel(0, axis=1)
        don_loc = self.frame_selector.select(don_loc, "d", renumber=True)
        don_loc.reset_index(inplace=True)

        acc_tr = trackpy.link(acc_loc, **self.link_options)
        don_tr = trackpy.link(don_loc, **self.link_options)

        codiff = multicolor.find_codiffusion(
            don_tr, acc_tr, **self.codiffusion_options,
            channel_names=["donor", "acceptor"], keep_unmatched="all")

        for ch in "donor", "acceptor":
            idx = codiff[ch, "index"]
            mask = np.isfinite(idx)  # lines with NaN as index have no match
            m_idx = idx[mask].astype(np.intp)
            for src, dest in [("donor", "d_particle"),
                              ("acceptor", "a_particle"),
                              ("codiff", "particle")]:
                p = codiff.loc[mask, (src, "particle")].copy()
                p.index = m_idx
                loc_data[ch]["fret", dest] = \
                    p.where(np.isfinite(p), -1).astype(np.intp)

    def interpolate_missing_video(self, source: Union[str, Tuple[str, str]],
                                  loc_data: Dict[str, pd.DataFrame]):
        cols = [("acceptor", "x"), ("acceptor", "y"), ("fret", "frame"),
                ("fret", "particle"), ("fret", "d_particle"),
                ("fret", "a_particle")]
        neigh_dist = self._get_neighbor_distance()

        da_loc = {}
        for ch, other in itertools.permutations(("donor", "acceptor")):
            # Interpolate gaps in each channel's traces
            loc = loc_data[ch][cols].droplevel(0, axis=1)
            loc = spatial.interpolate_coords(
                loc, columns={"particle": f"{ch[0]}_particle"})
            loc_ch = self.frame_selector.select(loc, ch[0]).copy()
            # Replace NaNs in particle columns
            loc_ch.fillna(method="pad", inplace=True, downcast="infer")
            # Check whether a change in particle number coincides with an
            # interpolated position, in which case the interpolated position
            # will not be assigned to a "particle"
            interp_idx = loc_ch.index[loc_ch["interp"] > 0]
            change_idx = np.nonzero(np.diff(loc_ch["particle"]))[0]
            loc_ch.loc[np.intersect1d(interp_idx, change_idx),
                       ["particle", f"{other[0]}_particle"]] = -1

            # Create tracks in the other channel where there was no
            # co-diffusion
            loc_o = self.frame_selector.select(loc, other[0])

            da_loc[ch, ch] = loc_ch
            da_loc[other, ch] = loc_o

        for ch, other in itertools.permutations(("donor", "acceptor")):
            # Concat data from actual localizations and from interpolations
            # in the other excitation channel
            loc = pd.concat([da_loc[ch, ch], da_loc[ch, other]],
                            ignore_index=True)
            # This will get rid of interpolated data where actual data exists
            loc.drop_duplicates(["frame", f"{other[0]}_particle"],
                                inplace=True, keep="first")
            # Fill NaNs in "particle" and other channel's particle columns
            loc.fillna(-1, downcast="infer", inplace=True)

            interp_mask = loc["interp"] > 0
            if not interp_mask.any():
                break
            if neigh_dist > 0:
                # TODO: Only flag as having a neighbor if nearby feature is
                # not interpolated
                spatial.has_near_neighbor(loc, neigh_dist)
            else:
                loc["has_neighbor"] = 0

            loc_interp = loc[interp_mask]

            d_loc = self.registrator(loc_interp, channel=2)
            a_loc = loc_interp.copy()

            opened = []
            try:
                im_seq, opened = self._open_image_sequence(source)
                b_opts, b_filt = self._get_brightness_options_filter()
                brightness.from_raw_image(d_loc, b_filt(im_seq["donor"]),
                                          **b_opts)
                brightness.from_raw_image(a_loc, b_filt(im_seq["acceptor"]),
                                          **b_opts)
            finally:
                for o in opened:
                    o.close()

            ex_cols = ["x", "y", "mass", "signal", "bg", "bg_dev"]
            ex_loc = pd.concat({"donor": d_loc[ex_cols],
                                "acceptor": a_loc[ex_cols],
                                "fret": d_loc[["frame", "particle",
                                               "d_particle", "a_particle",
                                               "has_neighbor", "interp"]]},
                               axis=1)
            loc_data[ch]["fret", "interp"] = 0
            ld = pd.concat([loc_data[ch], ex_loc], ignore_index=True)
            ld.sort_values([("fret", "particle"), ("fret", "frame")],
                           ignore_index=True, inplace=True)
            loc_data[ch] = ld
