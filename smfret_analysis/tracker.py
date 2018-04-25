import re
import os
import tempfile
import subprocess
import collections
from contextlib import suppress
from pathlib import Path

import pandas as pd
import numpy as np
import ipywidgets
import pims

from sdt import roi, chromatic, io, image
from sdt.fret import SmFretTracker, FretImageSelector
from sdt.loc import daostorm_3d
from sdt.nbui import Locator

from .version import output_version


class Tracker:
    def __init__(self, don_o, acc_o, roi_shape, exc_scheme="da",
                 data_dir=""):
        self.rois = dict(
            donor=roi.ROI(don_o, shape=roi_shape),
            acceptor=roi.ROI(acc_o, shape=roi_shape))
        self.tracker = SmFretTracker(exc_scheme)
        self.exc_img_filter = FretImageSelector(exc_scheme)
        self.img = collections.OrderedDict()
        self.loc_data = collections.OrderedDict()
        self.track_data = collections.OrderedDict()
        self.data_dir = Path(data_dir)

        self.bead_files = []
        self.bead_locator = None
        self.bead_loc_options = None

        self.donor_locator = None
        self.donor_loc_options = None

        self.acceptor_locator = None
        self.acceptor_loc_options = None

    def set_bead_loc_opts(self, files_re=None):
        if files_re is not None:
            self.bead_files = io.get_files(files_re, self.data_dir)[0]
        self.bead_locator = Locator([str(self.data_dir / f)
                                     for f in self.bead_files])
        if isinstance(self.bead_loc_options, dict):
            self.bead_locator.set_options(**self.bead_loc_options)

    def make_chromatic(self, plot=True, max_frame=None, params={}):
        if self.bead_locator is not None:
            self.bead_loc_options = self.bead_locator.get_options()
        if self.bead_loc_options is None:
            raise RuntimeError("Localization options not set. Either set the"
                               "`bead_loc_options` dict or use the "
                               "`set_bead_loc_opts` method.")

        label = ipywidgets.Label(value="Starting…")
        display(label)

        bead_loc = []
        for n, f in enumerate(self.bead_files):
            label.value = f"Locating beads (file {n+1}/{len(self.bead_files)})"
            with pims.open(str(self.data_dir / f)) as i:
                bead_loc.append(daostorm_3d.batch(
                    i[:max_frame], **self.bead_loc_options))
        label.layout = ipywidgets.Layout(display="none")

        acc_beads = [self.rois["acceptor"](l) for l in bead_loc]
        don_beads = [self.rois["donor"](l) for l in bead_loc]
        cc = chromatic.Corrector(don_beads, acc_beads)
        cc.determine_parameters(**params)

        if plot:
            cc.test()

        self.tracker.chromatic_corr = cc

    def add_dataset(self, key, files_re):
        self.img[key] = io.get_files(files_re, self.data_dir)[0]

    def donor_sum(self, fr):
        fr = self.exc_img_filter(fr, "d")
        fr_d = self.rois["donor"](fr)
        fr_a = self.rois["acceptor"](fr)
        return [a + self.tracker.chromatic_corr(d, channel=1, cval=d.mean())
                for d, a in zip(fr_d, fr_a)]

    def set_don_loc_opts(self, key, idx):
        i_name = self.img[key][idx]
        with pims.open(str(self.data_dir / i_name)) as fr:
            lo = {i_name: self.donor_sum(fr)}
        self.donor_locator = Locator(lo)
        if isinstance(self.donor_loc_options, dict):
            self.donor_locator.set_options(**self.donor_loc_options)

    def set_acc_loc_opts(self, key, idx):
        i_name = self.img[key][idx]
        with pims.open(str(self.data_dir / i_name)) as fr:
            lo = {i_name: list(self.rois["acceptor"](
                self.exc_img_filter(fr, "a")))}
        self.acceptor_locator = Locator(lo)
        if isinstance(self.acceptor_loc_options, dict):
            self.acceptor_locator.set_options(**self.acceptor_loc_options)

    def locate(self):
        num_files = sum(len(i) for i in self.img.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        if self.donor_locator is not None:
            self.donor_loc_options = self.donor_locator.get_options()
        if self.acceptor_locator is not None:
            self.acceptor_loc_options = self.acceptor_locator.get_options()
        if None in (self.donor_loc_options, self.acceptor_loc_options):
            raise RuntimeError("Localization options not set. Either set "
                               "{donor,acceptor}_loc_options or use the "
                               "`set_{don,acc}_loc_opts` methods.")

        for key, files in self.img.items():
            ret = []
            for i, f in enumerate(files):
                label.value = f"Locating {f} ({cnt}/{num_files})"
                cnt += 1

                with pims.open(str(self.data_dir / f)) as fr:
                    overlay = self.donor_sum(fr)
                    for o in overlay:
                        o[o < 1] = 1
                    lo_d = daostorm_3d.batch(overlay, **self.donor_loc_options)
                    acc_fr = list(self.rois["acceptor"](
                        self.exc_img_filter(fr, "a")))
                    for a in acc_fr:
                        a[a < 1] = 1
                    lo_a = daostorm_3d.batch(acc_fr,
                                             **self.acceptor_loc_options)
                    lo = pd.concat([lo_d, lo_a]).sort_values("frame")
                    lo = lo.reset_index(drop=True)

                    # correct for the fact that locating happend in the
                    # acceptor ROI
                    lo[["x", "y"]] += self.rois["acceptor"].top_left
                    ret.append(lo)
            self.loc_data[key] = pd.concat(ret, keys=files)

    def track(self, feat_radius=4, bg_frame=3, link_radius=1, link_mem=1,
              min_length=4, bg_estimator="mean",
              image_filter=lambda i: image.gaussian_filter(i, 1)):
        num_files = sum(len(i) for i in self.img.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        self.tracker.link_options.update({"search_range": link_radius,
                                          "memory": link_mem})
        self.tracker.brightness_options.update({"radius": feat_radius,
                                                "bg_frame": bg_frame,
                                                "bg_estimator": bg_estimator})
        self.tracker.min_length = min_length

        for key in self.img:
            ret = []
            ret_keys = []
            new_p = 0  # Particle ID unique across files
            for f in self.loc_data[key].index.levels[0].unique():
                loc = self.loc_data[key].loc[f].copy()
                label.value = f"Tracking {f} ({cnt}/{num_files})"
                cnt += 1

                with pims.open(str(self.data_dir / f)) as img:
                    don_loc = self.rois["donor"](loc)
                    acc_loc = self.rois["acceptor"](loc)

                    if image_filter is not None:
                        img = image_filter(img)

                    d = self.tracker.track(
                        self.rois["donor"](img), self.rois["acceptor"](img),
                        don_loc, acc_loc)
                ps = d["fret", "particle"].copy().values
                for p in np.unique(ps):
                    d.loc[ps == p, ("fret", "particle")] = new_p
                    new_p += 1
                ret.append(d)
                ret_keys.append(f)

            self.track_data[key] = pd.concat(ret, keys=ret_keys)

    def analyze(self):
        for t in self.track_data.values():
            self.tracker.analyze(t)

    def save_data(self, file_prefix="tracking"):
        loc_options = collections.OrderedDict([
            ("donor", self.donor_loc_options),
            ("acceptor", self.acceptor_loc_options),
            ("beads", self.bead_loc_options)])
        top = collections.OrderedDict(
            tracker=self.tracker, rois=self.rois, loc_options=loc_options,
            files=self.img, bead_files=self.bead_files)
        outfile = self.data_dir / f"{file_prefix}-v{output_version:03}"
        with outfile.with_suffix(".yaml").open("w") as f:
            io.yaml.safe_dump(top, f)

        with pd.HDFStore(outfile.with_suffix(".h5")) as s:
            for key, loc in self.loc_data.items():
                s["{}_loc".format(key)] = loc
            for key, trc in self.track_data.items():
                s["{}_trc".format(key)] = trc

    @classmethod
    def load(cls, file_prefix="tracking", data_dir="", loc=True, tracks=True):
        data_dir = Path(data_dir)
        infile = data_dir / f"{file_prefix}-v{output_version:03}"
        with infile.with_suffix(".yaml").open() as f:
            cfg = io.yaml.safe_load(f)
        ret = cls([0, 0], [0, 0], [0, 0], "")
        ret.rois = cfg["rois"]
        ret.img = cfg["files"]
        ret.bead_files = cfg["bead_files"]
        ret.bead_loc_options = cfg["loc_options"]["beads"]
        ret.donor_loc_options = cfg["loc_options"]["donor"]
        ret.acceptor_loc_options = cfg["loc_options"]["acceptor"]
        ret.tracker = cfg["tracker"]
        ret.data_dir = data_dir

        do_load = []
        if loc:
            do_load.append((ret.loc_data, "_loc"))
        if tracks:
            do_load.append((ret.track_data, "_trc"))
        with pd.HDFStore(infile.with_suffix(".h5"), "r") as s:
            for sink, suffix in do_load:
                keys = (k for k in s.keys() if k.endswith(suffix))
                for k in keys:
                    new_key = k[1:-len(suffix)]
                    sink[new_key] = s[k]

        return ret

