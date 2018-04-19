import re
import os
import tempfile
import subprocess
import collections
from contextlib import suppress

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
    def __init__(self, don_o, acc_o, roi_size, exc_scheme="da"):
        self.rois = dict(
            donor=roi.ROI(don_o, (don_o[0] + roi_size[0],
                                  don_o[1] + roi_size[1])),
            acceptor=roi.ROI(acc_o, (acc_o[0] + roi_size[0],
                                     acc_o[1] + roi_size[1])))
        self.exc_scheme = exc_scheme
        self.exc_img_filter = FretImageSelector(exc_scheme)
        self.cc = None
        self.img = collections.OrderedDict()
        self.loc_data = collections.OrderedDict()
        self.track_data = collections.OrderedDict()

        self.bead_files = []
        self.bead_locator = None
        self.bead_loc_options = None

        self.donor_locator = None
        self.donor_loc_options = None

        self.acceptor_locator = None
        self.acceptor_loc_options = None

    def set_bead_loc_opts(self, files_re=None):
        if files_re is not None:
            self.bead_files = io.get_files(files_re)[0]
        self.bead_locator = Locator(self.bead_files)
        if isinstance(self.bead_loc_options, dict):
            self.bead_locator.set_options(**self.bead_loc_options)

    def make_chromatic(self, loc=True, plot=True, max_frame=None, params={}):
        self.bead_loc_options = self.bead_locator.get_options()

        bead_loc = []
        for f in self.bead_files:
            with pims.open(f) as i:
                bead_loc.append(daostorm_3d.batch(
                    i[:max_frame], **self.bead_loc_options))

        acc_beads = [self.rois["acceptor"](l) for l in bead_loc]
        don_beads = [self.rois["donor"](l) for l in bead_loc]
        # things below assume that first channel is donor, second is acceptor
        cc = chromatic.Corrector(don_beads, acc_beads)
        cc.determine_parameters(**params)

        if plot:
            cc.test()

        self.cc = cc

    def add_dataset(self, key, files_re):
        self.img[key] = io.get_files(files_re)[0]

    def donor_sum(self, fr):
        fr = self.exc_img_filter(fr, "d")
        fr_d = self.rois["donor"](fr)
        fr_a = self.rois["acceptor"](fr)
        return [a + self.cc(d, channel=1, cval=d.mean())
                for d, a in zip(fr_d, fr_a)]

    def set_don_loc_opts(self, key, idx):
        i_name = self.img[key][idx]
        with pims.open(i_name) as fr:
            lo = {i_name: self.donor_sum(fr)}
        self.donor_locator = Locator(lo)
        if isinstance(self.donor_loc_options, dict):
            self.donor_locator.set_options(**self.donor_loc_options)

    def set_acc_loc_opts(self, key, idx):
        i_name = self.img[key][idx]
        with pims.open(i_name) as fr:
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

        self.donor_loc_options = self.donor_locator.get_options()
        self.acceptor_loc_options = self.acceptor_locator.get_options()

        for key, files in self.img.items():
            ret = []
            for i, f in enumerate(files):
                label.value = "Locating {} ({}/{})".format(f, cnt, num_files)
                cnt += 1

                with pims.open(f) as fr:
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

        self.tracker = SmFretTracker(self.exc_scheme, self.cc, link_radius,
                                     link_mem, min_length, feat_radius,
                                     bg_frame, bg_estimator, interpolate=True)

        for key in self.img:
            ret = []
            ret_keys = []
            new_p = 0  # Particle ID unique across files
            for f in self.loc_data[key].index.levels[0].unique():
                loc = self.loc_data[key].loc[f].copy()
                label.value = "Tracking {} ({}/{})".format(f, cnt, num_files)
                cnt += 1

                with pims.open(f) as img:
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
            self.tracker.analyze(t, aa_interp="linear")

    def save_data(self, file_prefix="tracking"):
        loc_options = collections.OrderedDict([
            ("donor", self.donor_loc_options),
            ("acceptor", self.acceptor_loc_options),
            ("beads", self.bead_loc_options)])
        top = collections.OrderedDict(
            excitation_scheme=self.exc_scheme, rois=self.rois,
            loc_options=loc_options, files=self.img,
            bead_files=self.bead_files)
        with open("{}-v{:03}.yaml".format(file_prefix, output_version),
                  "w") as f:
            io.yaml.safe_dump(top, f, default_flow_style=False)

        with suppress(AttributeError):
            fn = "{}-v{:03}_chromatic.npz".format(file_prefix, output_version)
            self.cc.save(fn)

        with pd.HDFStore("{}-v{:03}.h5".format(file_prefix,
                                               output_version)) as s:
            for key, loc in self.loc_data.items():
                s["{}_loc".format(key)] = loc
            for key, trc in self.track_data.items():
                s["{}_trc".format(key)] = trc

    @classmethod
    def load(cls, file_prefix="tracking", loc=True, tracks=True):
        with open("{}-v{:03}.yaml".format(file_prefix, output_version)) as f:
            cfg = io.yaml.safe_load(f)
        ret = cls([0, 0], [0, 0], [0, 0], cfg["excitation_scheme"])
        ret.rois = cfg["rois"]
        ret.img = cfg["files"]
        ret.bead_files = cfg["bead_files"]
        ret.bead_loc_options = cfg["loc_options"]["beads"]
        ret.donor_loc_options = cfg["loc_options"]["donor"]
        ret.acceptor_loc_options = cfg["loc_options"]["acceptor"]

        try:
            fn = "{}-v{:03}_chromatic.npz".format(file_prefix, output_version)
            ret.cc = chromatic.Corrector.load(fn)
        except FileNotFoundError:
            ret.cc = None

        do_load = []
        if loc:
            do_load.append((ret.loc_data, "_loc"))
        if tracks:
            do_load.append((ret.track_data, "_trc"))
        with pd.HDFStore("{}-v{:03}.h5".format(file_prefix,
                                               output_version), "r") as s:
            for sink, suffix in do_load:
                keys = (k for k in s.keys() if k.endswith(suffix))
                for k in keys:
                    new_key = k[1:-len(suffix)]
                    sink[new_key] = s[k]

        return ret

