import re
import os
import tempfile
import subprocess
import collections

import pandas as pd
import numpy as np
import ipywidgets
import pims

from sdt import roi, fret, chromatic, io, image
from sdt.loc import daostorm_3d

from .version import output_version

 
def get_files(subdir, pattern):
    r = re.compile(pattern)
    flist = []
    idlist = []
    for dp, dn, fn in os.walk(subdir):
        for f in fn:
            relp = os.path.relpath(os.path.join(dp, f), subdir)
            m = r.fullmatch(relp)
            if m is None:
                continue
            flist.append(os.path.join(dp, f))
            ids = []
            for i in m.groups():
                try:
                    ids.append(int(i))
                except ValueError:
                    try:
                        ids.append(float(i))
                    except ValueError:
                        ids.append(i)
            idlist.append(ids)
    slist = sorted(zip(flist, idlist), key=lambda x: x[0])
    return [s[0] for s in slist], [s[1] for s in slist]


class Tracker:
    def __init__(self, don_o, acc_o, size, ex_scheme="da"):
        self.don_roi = roi.ROI(don_o, (don_o[0] + size[0], don_o[1] + size[1]))
        self.acc_roi = roi.ROI(acc_o, (acc_o[0] + size[0], acc_o[1] + size[1]))
        self.ana = fret.SmFretAnalyzer(ex_scheme)
        self.img = {}
        self.loc_data = {}
        self.track_data = {}

    def make_cc(self, path, files_re, loc=True, plot=True, params={}):
        bead_files, _ = get_files(path, files_re)

        if loc:
            subprocess.run(["python", "-m", "sdt.gui.locator"] + bead_files)

        bead_loc = [io.load(os.path.splitext(f)[0]+".h5") for f in bead_files]
        acc_beads = [self.acc_roi(l) for l in bead_loc]
        don_beads = [self.don_roi(l) for l in bead_loc]
        # things below assume that first channel is donor, second is acceptor
        cc = chromatic.Corrector(don_beads, acc_beads)
        cc.determine_parameters(**params)
        cc.save("chromatic.npz")

        if plot:
            cc.test()

        self.cc = cc

    def add_dataset(self, key, path, files_re):
        self.img[key] = get_files(path, files_re)[0]

    def donor_sum(self, fr):
        fr = self.ana.get_excitation_type(fr, "d")
        fr_d = self.don_roi(fr)
        fr_a = self.acc_roi(fr)
        return [a + self.cc(d, channel=1, cval=d.mean())
                for d, a in zip(fr_d, fr_a)]

    def set_loc_options(self, key, idx):
        i_name = self.img[key][idx]
        with pims.open(i_name) as fr:
            fr = fr[1:-1]
            with tempfile.NamedTemporaryFile(suffix=".tif") as t:
                io.save_as_tiff(self.donor_sum(fr), t.name)
                subprocess.run(["python", "-m", "sdt.gui.locator", t.name])
            with tempfile.NamedTemporaryFile(suffix=".tif") as t:
                i = self.ana.get_excitation_type(fr, "a")
                io.save_as_tiff(self.acc_roi(i), t.name)
                subprocess.run(["python", "-m", "sdt.gui.locator", t.name])

    def locate(self):
        num_files = sum(len(i) for i in self.img.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        for key, files in self.img.items():
            ret = []
            for i, f in enumerate(files):
                label.value = f"Locating {f} ({cnt}/{num_files})"
                cnt += 1

                subdir = os.path.dirname(f)
                with open(os.path.join(subdir, "loc-options-a.yaml")) as lo:
                    opts_a = io.yaml.load(lo)["options"]
                with open(os.path.join(subdir, "loc-options-d.yaml")) as lo:
                    opts_d = io.yaml.load(lo)["options"]

                with pims.open(f) as fr:
                    fr = fr[1:-1]
                    overlay = self.donor_sum(fr)
                    for o in overlay:
                        o[o < 1] = 1
                    lo_d = daostorm_3d.batch(overlay, **opts_d)
                    acc_fr = list(self.acc_roi(
                        self.ana.get_excitation_type(fr, "a")))
                    for a in acc_fr:
                        a[a < 1] = 1
                    lo_a = daostorm_3d.batch(acc_fr, **opts_a)
                    lo = pd.concat([lo_d, lo_a]).sort_values("frame")
                    lo = lo.reset_index(drop=True)

                    # correct for the fact that locating happend in the
                    # acceptor ROI
                    lo[["x", "y"]] += self.acc_roi.top_left
                    ret.append(lo)
            self.loc_data[key] = ret

    def track(self, feat_radius=4, bg_frame=3, link_radius=1, link_mem=1,
              min_length=4, bg_estimator="mean",
              image_filter=lambda i: image.gaussian_filter(i, 1)):
        num_files = sum(len(i) for i in self.img.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        for key in self.img:
            ret = []
            for f, loc in zip(self.img[key], self.loc_data[key]):
                label.value = f"Tracking {f} ({cnt}/{num_files})"
                cnt += 1

                with pims.open(f) as img:
                    # First and last images are Fura, remove them
                    loc = loc[(loc["frame"] > 0) &
                              (loc["frame"] < len(img) - 1)]
                    img = img[1:-1]
                    # Correct frame number
                    loc["frame"] -= 1

                    don_loc = self.don_roi(loc)
                    acc_loc = self.acc_roi(loc)

                    if image_filter is not None:
                        img = image_filter(img)

                    # Track
                    d = fret.SmFretData.track(
                        self.ana, self.don_roi(img),
                        self.acc_roi(img), don_loc, acc_loc, self.cc,
                        link_radius=link_radius, link_mem=link_mem,
                        min_length=min_length, bg_estimator=bg_estimator,
                        feat_radius=feat_radius, bg_frame=bg_frame,
                        interpolate=True)
                ret.append(d.tracks)
            self.track_data[key] = ret

    def analyze_fret(self):
        num_files = sum(len(i) for i in self.track_data.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        for key in self.img:
            for f, t in zip(self.img[key], self.track_data[key]):
                label.value = (f"Calculating FRET values {f} "
                               f"({cnt}/{num_files})")
                cnt += 1
                self.ana.quantify_fret(t, aa_interp="linear")

    def save_data(self, file_prefix="tracking"):
        with pd.HDFStore(f"{file_prefix}-v{output_version:03}.h5") as s:
            for key in self.img:
                loc = pd.concat(self.loc_data[key], keys=self.img[key])
                s[f"{key}_loc"] = loc

                # Give each particle a unique ID (across files)
                new_p = 0
                for t in self.track_data[key]:
                    ps = t["fret", "particle"].copy().values
                    for p in np.unique(ps):
                        t.loc[ps == p, ("fret", "particle")] = new_p
                        new_p += 1
                trc = pd.concat(self.track_data[key], keys=self.img[key])
                s[f"{key}_trc"] = trc

        rois = dict(donor=self.don_roi, acceptor=self.acc_roi)
        top = collections.OrderedDict(rois=rois,
                                      excitation_scheme="".join(self.ana.desc),
                                      files=self.img)
        with open(f"{file_prefix}-v{output_version:03}.yaml", "w") as f:
            io.yaml.safe_dump(top, f, default_flow_style=False)
