# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import os
import tempfile
import subprocess
import collections
from contextlib import suppress
from pathlib import Path
import warnings

import pandas as pd
import numpy as np
import ipywidgets
import pims
import matplotlib.pyplot as plt

from sdt import roi, chromatic, io, image, fret, changepoint, helper
from sdt import flatfield as _flatfield  # avoid name clashes
from sdt.fret import SmFretTracker, FrameSelector
from sdt.loc import daostorm_3d
from sdt.nbui import Locator

from .version import output_version


@helper.pipeline(ancestor_count=2)
def _img_sum(d, a):
    return d + a


class Tracker:
    def __init__(self, don_o, acc_o, roi_size, excitation_seq="da",
                 data_dir=""):
        self.rois = dict(
            donor=roi.ROI(don_o, size=roi_size),
            acceptor=roi.ROI(acc_o, size=roi_size))
        self.tracker = SmFretTracker(excitation_seq)
        self.sources = collections.OrderedDict()

        self.loc_data = collections.OrderedDict()
        self.track_data = collections.OrderedDict()
        self.cell_images = collections.OrderedDict()
        self.flatfield = collections.OrderedDict()
        self.data_dir = Path(data_dir)

        self.locators = collections.OrderedDict([
            ("donor", Locator()), ("acceptor", Locator()),
            ("beads", Locator())])

    def _make_dataset_selector(self, state, callback):
        d_sel = ipywidgets.Dropdown(options=list(self.track_data.keys()),
                                    description="dataset")

        def change_dataset(key=None):
            tr = self.track_data[d_sel.value]
            state["tr"] = tr
            state["pnos"] = sorted(tr["fret", "particle"].unique())
            state["files"] = tr.index.remove_unused_levels().levels[0].unique()

            callback()

        d_sel.observe(change_dataset, names="value")
        change_dataset()

        return d_sel

    def add_dataset(self, key, files_re, cells=False):
        files = io.get_files(files_re, self.data_dir)[0]
        if not files:
            warnings.warn(f"Empty dataset added: {key}")
        s = collections.OrderedDict(
            [("files", self._open_image_sequences(files)), ("cells", cells)])
        self.sources[key] = s

    def set_bead_loc_opts(self, files_re):
        files = io.get_files(files_re, self.data_dir)[0]
        self.locators["beads"].files = \
            {f: pims.open(str(self.data_dir / f)) for f in files}
        self.locators["beads"].auto_scale()
        return self.locators["beads"]

    def make_chromatic(self, plot=True, max_frame=None, params={}):
        label = ipywidgets.Label(value="Starting…")
        display(label)

        bead_loc = []
        lcr = self.locators["beads"]
        n_files = len(lcr.files)
        for n, (f, i) in enumerate(self.locators["beads"].files.items()):
            label.value = f"Locating beads (file {n+1}/{n_files})"
            bead_loc.append(lcr.batch_func(
                i[:max_frame], **lcr.options))
        label.layout = ipywidgets.Layout(display="none")

        acc_beads = [self.rois["acceptor"](l) for l in bead_loc]
        don_beads = [self.rois["donor"](l) for l in bead_loc]
        cc = chromatic.Corrector(don_beads, acc_beads)
        cc.determine_parameters(**params)
        self.tracker.chromatic_corr = cc

        if plot:
            fig, ax = plt.subplots(1, 2)
            cc.test(ax=ax)
            return fig.canvas

    def _open_image_sequences(self, files):
        ret = collections.OrderedDict()
        for f in files:
            pth = self.data_dir / f
            if pth.suffix.lower() == ".spe":
                # Disable warnings about file size being wrong which is caused
                # by SDT-control not dumping the whole file
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ret[f] = pims.open(str(pth))
            else:
                ret[f] = pims.open(str(pth))
        return ret

    def donor_sum(self, fr):
        fr = self.tracker.frame_selector(fr, "d")
        fr_d = self.tracker.chromatic_corr(self.rois["donor"](fr), channel=1,
                                           cval=np.mean)
        fr_a = self.rois["acceptor"](fr)
        return _img_sum(fr_d, fr_a)

    def set_loc_opts(self, exc_type):
        d_sel = ipywidgets.Dropdown(
            options=list(self.sources.keys()), description="dataset")

        if exc_type.startswith("d"):
            loc = self.locators["donor"]
        elif exc_type.startswith("a"):
            loc = self.locators["acceptor"]
        else:
            raise ValueError("`exc_type` has to be \"donor\" or \"acceptor\".")

        def set_files(change=None):
            if exc_type.startswith("d"):
                loc.files = \
                    {k: self.donor_sum(v)
                     for k, v in self.sources[d_sel.value]["files"].items()}
            elif exc_type.startswith("a"):
                loc.files = \
                    {k: self.rois["acceptor"](
                         self.tracker.frame_selector(v, "a"))
                     for k, v in self.sources[d_sel.value]["files"].items()}

        set_files()
        loc.auto_scale()
        d_sel.observe(set_files, "value")

        return ipywidgets.VBox([d_sel, loc])

    def locate(self):
        num_files = sum(len(s["files"]) for s in self.sources.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        for key, src in self.sources.items():
            ret = []
            files = src["files"]
            for i, (f, fr) in enumerate(files.items()):
                label.value = f"Locating {f} ({cnt}/{num_files})"
                cnt += 1

                don_fr = self.donor_sum(fr)
                lo = self.locators["donor"].batch_func(
                    don_fr, **self.locators["donor"].options)

                acc_fr = self.rois["acceptor"](
                    self.tracker.frame_selector(fr, "a"))
                if len(acc_fr):
                    lo_a = self.locators["acceptor"].batch_func(
                        acc_fr, **self.locators["acceptor"].options)
                    lo = pd.concat([lo, lo_a]).sort_values("frame")
                    lo = lo.reset_index(drop=True)

                # correct for the fact that locating happend in the
                # acceptor ROI
                self.rois["acceptor"].reset_origin(lo)
                ret.append(lo)
            self.loc_data[key] = pd.concat(ret, keys=files)

    def track(self, feat_radius=4, bg_frame=3, link_radius=1, link_mem=1,
              min_length=4, bg_estimator="mean",
              image_filter=lambda i: image.gaussian_filter(i, 1),
              neighbor_radius=None):
        num_files = sum(len(s["files"]) for s in self.sources.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        self.tracker.link_options.update({"search_range": link_radius,
                                          "memory": link_mem})
        self.tracker.brightness_options.update({"radius": feat_radius,
                                                "bg_frame": bg_frame,
                                                "bg_estimator": bg_estimator})
        self.tracker.min_length = min_length
        if neighbor_radius is not None:
            self.tracker.neighbor_radius = neighbor_radius
        else:
            self.tracker.neighbor_radius = 2 * feat_radius + 1

        for key, src in self.sources.items():
            ret = []
            ret_keys = []
            new_p = 0  # Particle ID unique across files
            for f, img in src["files"].items():
                label.value = f"Tracking {f} ({cnt}/{num_files})"
                cnt += 1

                loc = self.loc_data[key]
                try:
                    loc = loc.loc[f].copy()
                except KeyError:
                    # No localizations in this file
                    continue

                don_loc = self.rois["donor"](loc)
                acc_loc = self.rois["acceptor"](loc)

                if image_filter is not None:
                    img = image_filter(img)

                try:
                    d = self.tracker.track(
                            self.rois["donor"](img),
                            self.rois["acceptor"](img),
                            don_loc, acc_loc)
                except Exception as e:
                    warnings.warn(f"Tracking failed for {f}. Reason: {e}")

                    fake_fret_df = pd.DataFrame(
                        columns=["particle", "interp", "has_neighbor"],
                        dtype=int)
                    d = pd.concat([don_loc.iloc[:0], acc_loc.iloc[:0],
                                   fake_fret_df],
                                  keys=["donor", "acceptor", "fret"], axis=1)

                ps = d["fret", "particle"].copy().values
                for p in np.unique(ps):
                    d.loc[ps == p, ("fret", "particle")] = new_p
                    new_p += 1
                ret.append(d)
                ret_keys.append(f)

            self.track_data[key] = pd.concat(ret, keys=ret_keys)

    def extract_cell_images(self, key="c"):
        for k, v in self.sources.items():
            if not v["cells"]:
                # no cells
                continue
            for f, fr in v["files"].items():
                don_fr = self.rois["donor"](fr)
                cell_fr = np.array(self.tracker.frame_selector(don_fr, key))
                self.cell_images[f] = cell_fr

    def make_flatfield(self, dest, files_re, src=None, frame=0, bg=200,
                       smooth_sigma=3., gaussian_fit=False):
        files = io.get_files(files_re, self.data_dir)[0]
        imgs = []
        if src is None:
            src = dest

        for f in files:
            with pims.open(str(self.data_dir / f)) as fr:
                imgs.append(self.rois[src](fr[frame].astype(float)) - bg)

        if src == "acceptor" and dest == "donor":
            imgs = [self.tracker.chromatic_corr(i, channel=2) for i in imgs]
        elif src != dest:
            raise ValueError ("src != dest and (src != \"acceptor\" and dest "
                              "!= \"donor\)")

        self.flatfield[dest] = _flatfield.Corrector(
            imgs, smooth_sigma=smooth_sigma, gaussian_fit=gaussian_fit)

    def make_flatfield_sm(self, dest, keys="all", weighted=False, frame=None):
        if not len(keys):
            return
        if keys == "all":
            keys = self.track_data.keys()
        elif keys == "no-cells":
            keys = [k for k in self.track_data.keys() if self.sources[k]["cells"]]

        if frame is None:
            frame = self.tracker.excitation_frames[dest[0]][0]

        data = []
        for k in keys:
            d = self.track_data[k]
            d = d[(d[dest, "frame"] == frame) &
                  (d["fret", "interp"] == 0) &
                  (d["fret", "has_neighbor"] == 0)]
            if dest.startswith("d"):
                mass = d["donor", "mass"] + d["acceptor", "mass"]
            else:
                mass = d[dest, "mass"]
            d = d[[(dest, "x"), (dest, "y")]].copy()
            d.columns = ["x", "y"]
            d["mass"] = mass
            data.append(d)

        r = self.rois[dest]
        img_shape = (r.bottom_right[1] - r.top_left[1],
                     r.bottom_right[0] - r.top_left[0])
        self.flatfield[dest] = _flatfield.Corrector(*data, shape=img_shape,
                                                    density_weight=weighted)

    def save(self, file_prefix="tracking"):
        loc_options = collections.OrderedDict(
            [(k, v.get_settings()) for k, v in self.locators.items()])

        src = collections.OrderedDict()
        for k, v in self.sources.items():
            v_new = v.copy()
            v_new["files"] = list(v["files"].keys())
            src[k] = v_new

        top = collections.OrderedDict(
            tracker=self.tracker, rois=self.rois, loc_options=loc_options,
            data_dir=str(self.data_dir), sources=src)
        outfile = Path(f"{file_prefix}-v{output_version:03}")
        with outfile.with_suffix(".yaml").open("w") as f:
            io.yaml.safe_dump(top, f)

        with warnings.catch_warnings():
            import tables
            warnings.simplefilter("ignore", tables.NaturalNameWarning)

            with pd.HDFStore(outfile.with_suffix(".h5")) as s:
                for key, loc in self.loc_data.items():
                    s["{}_loc".format(key)] = loc
                for key, trc in self.track_data.items():
                    s["{}_trc".format(key)] = trc

        np.savez_compressed(outfile.with_suffix(".cell_img.npz"),
                            **self.cell_images)
        for k, ff in self.flatfield.items():
            ff.save(outfile.with_suffix(f".flat_{k}.npz"))

    @staticmethod
    def load_data(file_prefix="tracking", loc=True, tracks=True,
                  cell_images=True, flatfield=True, images=True):
        infile = Path(f"{file_prefix}-v{output_version:03}")
        with infile.with_suffix(".yaml").open() as f:
            cfg = io.yaml.safe_load(f)

        ret = {"rois": cfg["rois"], "data_dir": Path(cfg.get("data_dir", "")),
               "sources": cfg["sources"], "loc_options": cfg["loc_options"],
               "tracker": cfg["tracker"],
               "loc_data": collections.OrderedDict(),
               "track_data": collections.OrderedDict(),
               "cell_images": collections.OrderedDict(),
               "flatfield": collections.OrderedDict()}

        do_load = []
        if loc:
            do_load.append((ret["loc_data"], "_loc"))
        if tracks:
            do_load.append((ret["track_data"], "_trc"))
        if len(do_load):
            with pd.HDFStore(infile.with_suffix(".h5"), "r") as s:
                for sink, suffix in do_load:
                    keys = (k for k in s.keys() if k.endswith(suffix))
                    for k in keys:
                        new_key = k[1:-len(suffix)]
                        sink[new_key] = s[k]

        if cell_images:
            cell_img_file = infile.with_suffix(".cell_img.npz")
            try:
                with np.load(cell_img_file) as data:
                    ret["cell_images"] = collections.OrderedDict(data)
            except Exception:
                warnings.warn("Could not load cell images from file "
                              f"\"{str(cell_img_file)}\".")
        if flatfield:
            flatfield_glob = str(infile.with_suffix(".flat_*.npz"))
            key_re = re.compile(r"^\.flat_([\w\s-]+)")
            for p in Path().glob(flatfield_glob):
                m = key_re.match(p.suffixes[-2])
                if m is None:
                    warnings.warn("Could not load flatfield corrector from "
                                  f"{str(p)}.")
                else:
                    ret["flatfield"][m.group(1)] = _flatfield.Corrector.load(p)

        return ret

    @classmethod
    def load(cls, file_prefix="tracking", loc=True, tracks=True,
             cell_images=True, profile_images=True):
        cfg = cls.load_data(file_prefix, loc, tracks, cell_images,
                            profile_images)

        ret = cls([0, 0], [0, 0], [0, 0], "")

        for key in ("rois", "data_dir", "tracker", "loc_data", "track_data",
                    "cell_images", "flatfield"):
            setattr(ret, key, cfg[key])

        src = cfg["sources"]
        for k, s in src.items():
            src[k]["files"] = ret._open_image_sequences(s["files"])
        ret.sources = src

        for n, lo in cfg["loc_options"].items():
            ret.locators[n].set_settings(lo)

        return ret
