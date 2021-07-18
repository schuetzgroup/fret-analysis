# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from sdt import fret

from .data_store import DataStore


class Plotter:
    def __init__(self, file_prefix="tracking"):
        ds = DataStore.load(file_prefix, loc=False, tracks=True,
                            segment_images=False, flatfield=False)
        self.track_data = ds.tracks

    def scatter(self, *args, **kwargs):
        return fret.smfret_scatter(self.track_data, *args, **kwargs)

    def hist(self, *args, **kwargs):
        return fret.smfret_hist(self.track_data, *args, **kwargs)
