import contextlib
import functools
import warnings
import collections
from pathlib import Path

import numpy as np
from scipy import ndimage
import pandas as pd
import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pims
import cv2

from sdt import io, roi, fret, chromatic, flatfield, helper, nbui, image

from .version import output_version
from .tracker import Tracker


class StatItem:
    def __init__(self, op, n_before, n_after):
        self.op = op
        self.n_before = n_before
        self.n_after = n_after


class Filter:
    def __init__(self, file_prefix="tracking"):
        cfg = Tracker.load_data(file_prefix, loc=False)

        self.rois = cfg["rois"]
        self.cc = cfg["tracker"].chromatic_corr
        self.track_filters = {k: fret.SmFretFilter(v)
                              for k, v in cfg["track_data"].items()}
        self.exc_scheme = "".join(cfg["tracker"].excitation_seq)
        self.data_dir = cfg["data_dir"]
        self.sources = cfg["sources"]
        self.cell_images = cfg["cell_images"]
        self.profile_images = cfg["profile_images"]

        self.beam_shapes = {"donor": None, "acceptor": None}

        self._brightness_fig = None
        self._thresholder = None
        self._beam_shape_fig = None
        self._beam_shape_artists = [None, None]

        self.statistics = {k: [] for k in self.track_filters.keys()}

    def calc_beam_shape_sm(self, keys="all", channel="donor", weighted=False,
                           frame=None):
        if not len(keys):
            return

        if channel.startswith("d"):
            k = "donor"
            mk = "d_mass"
            f = self.exc_scheme.find("d") if frame is None else frame
        elif channel.startswith("a"):
            k = "acceptor"
            mk = "a_mass"
            f = self.exc_scheme.find("a") if frame is None else frame
        else:
            raise ValueError("Channel must be \"donor\" or \"acceptor\".")

        r = self.rois[k]

        if keys == "all":
            keys = self.track_filters.keys()

        img_shape = (r.bottom_right[1] - r.top_left[1],
                     r.bottom_right[0] - r.top_left[0])
        data = [v.tracks_orig for k, v in self.track_filters.items()
                if k in keys]
        data = [d[(d[k, "frame"] == f) & (d["fret", "interp"] == 0) &
                  (d["fret", "has_neighbor"] == 0)]
                for d in data]
        bs = flatfield.Corrector(
            *data, columns={"coords": [(k, "x"), (k, "y")],
                            "mass": ("fret", mk)},
            shape=img_shape, density_weight=weighted)

        self.beam_shapes[k] = bs

    def calc_beam_shape_bulk(self, channel="donor", gaussian_fit=True):
        self.beam_shapes[channel] = flatfield.Corrector(
            self.profile_images[channel], gaussian_fit=gaussian_fit)

    def filter_acc_bleach(self, brightness_thresh):
        for k, f in self.track_filters.items():
            b = len(f.tracks)
            f.acceptor_bleach_step(brightness_thresh, truncate=True)
            a = len(f.tracks)
            self.statistics[k].append(StatItem("acc bleach", b, a))

    def _make_dataset_selector(self, state, callback):
        d_sel = ipywidgets.Dropdown(options=list(self.track_filters.keys()),
                                    description="dataset")

        def change_dataset(key=None):
            trc = self.track_filters[d_sel.value].tracks
            state["trc"] = trc
            state["pnos"] = sorted(trc["fret", "particle"].unique())
            state["files"] = \
                trc.index.remove_unused_levels().levels[0].unique()

            callback()

        d_sel.observe(change_dataset, names="value")
        change_dataset()

        return d_sel

    def find_brightness_params(self, frame=None):
        state = {"picked": []}

        if self._brightness_fig is None:
            fig = plt.figure()
            gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
            fig.add_subplot(gs[0, 0])
            fig.add_subplot(gs[0, 1])
            fig.add_subplot(gs[1, :])

            def on_pick(event):
                state["picked"] = event.ind
                plot_time_trace()

            fig.canvas.mpl_connect("pick_event", on_pick)
            self._brightness_fig = fig

        es_ax, br_ax, t_ax = self._brightness_fig.axes

        f_sel = ipywidgets.IntText(
            value=self.exc_scheme.find("d") if frame is None else frame,
            description="frame")

        def set_frame(change=None):
            trc = state["trc"]
            state["cur_trc"] = trc[(trc["donor", "frame"] == f_sel.value) &
                                   (trc["fret", "has_neighbor"] == 0)]

            plot_scatter()
            plot_time_trace()

        def plot_scatter(change=None):
            t = state["cur_trc"]
            es_ax.cla()
            es_ax.scatter(t["fret", "eff"], t["fret", "stoi"], picker=True,
                          marker=".")
            es_ax.set_xlim([-0.5, 1.5])
            es_ax.set_ylim([0., 1.25])

            br_ax.cla()
            br_ax.scatter(t["fret", "d_mass"], t["fret", "a_mass"],
                          picker=True, marker=".")

        def plot_time_trace(change=None):
            t_ax.cla()

            if len(state["picked"]) == 1:
                pno = state["cur_trc"].iloc[state["picked"][0]]
                pno = pno["fret", "particle"]

                t = state["trc"]
                t = t[(t["fret", "particle"] == pno) &
                      (t["fret", "exc_type"] == 0)]

                t_ax.plot(t["donor", "frame"], t["donor", "mass"], "g")
                t_ax.plot(t["acceptor", "frame"], t["acceptor", "mass"], "r")

            self._brightness_fig.tight_layout()
            self._brightness_fig.canvas.draw()

        d_sel = self._make_dataset_selector(state, set_frame)
        f_sel.observe(set_frame, "value")

        return ipywidgets.VBox([d_sel, f_sel, self._brightness_fig.canvas])

    def query(self, expr, mi_sep="_"):
        for k, f in self.track_filters.items():
            b = len(f.tracks)
            f.query(expr, mi_sep)
            a = len(f.tracks)
            self.statistics[k].append(StatItem("query", b, a))

    def present_at_start(self, frame=None):
        frame = self.exc_scheme.find("d") if frame is None else frame
        for k, f in self.track_filters.items():
            b = len(f.tracks)
            f.filter_particles(f"donor_frame == {frame}")
            a = len(f.tracks)
            self.statistics[k].append(StatItem("at start", b, a))

    def find_cell_mask_params(self):
        if self._thresholder is None:
            self._thresholder = nbui.Thresholder()

        self._thresholder.images = collections.OrderedDict(
            [(k, v[0]) for k, v in self.cell_images.items()])

        return self._thresholder

    def apply_cell_masks(self, method="adaptive", **kwargs):
        for k, v in self.sources.items():
            filt = self.track_filters[k]
            b = len(filt.tracks)

            if isinstance(method, str):
                method = getattr(image, method + "_thresh")

            if v["cells"]:
                trc = filt.tracks

                files = np.unique(trc.index.levels[0])
                mask = [(f, method(self.cell_images[f][0], **kwargs))
                        for f in files]
                filt.image_mask(mask, channel="donor")

            a = len(filt.tracks)
            self.statistics[k].append(StatItem("cell mask", b, a))

    def find_beam_shape_thresh(self):
        if self._beam_shape_fig is None:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            fig.add_subplot(1, 2, 2)

            self._beam_shape_fig = fig

        ax = self._beam_shape_fig.axes

        channel_sel = ipywidgets.Dropdown(options=list(self.beam_shapes),
                                          description="channel")
        thresh_sel = ipywidgets.BoundedIntText(value=75, min=0, max=100,
                                               description="threshold")

        bs = self.beam_shapes[channel_sel.value]

        def channel_changed(change=None):
            bs = self.beam_shapes[channel_sel.value]
            if self._beam_shape_artists[0] is not None:
                self._beam_shape_artists[0].remove()
            self._beam_shape_artists[0] = ax[0].imshow(bs.corr_img)
            update()

        def update(change=None):
            bs = self.beam_shapes[channel_sel.value]
            if self._beam_shape_artists[1] is not None:
                self._beam_shape_artists[1].remove()
            self._beam_shape_artists[1] = ax[1].imshow(
                bs.corr_img * 100 > thresh_sel.value)
            self._beam_shape_fig.tight_layout()
            self._beam_shape_fig.canvas.draw()

        channel_sel.observe(channel_changed, "value")
        thresh_sel.observe(update, "value")

        channel_changed()

        return ipywidgets.VBox([channel_sel, thresh_sel,
                                self._beam_shape_fig.canvas])

    def filter_beam_shape_region(self, channel, thresh):
        mask = self.beam_shapes[channel].corr_img * 100 > thresh
        for k, f in self.track_filters.items():
            b = len(f.tracks)
            f.image_mask(mask, channel)
            a = len(f.tracks)
            self.statistics[k].append(StatItem("beam shape", b, a))

    def save(self, file_prefix="filtered"):
        outfile = Path(f"{file_prefix}-v{output_version:03}")

        with warnings.catch_warnings():
            import tables
            warnings.simplefilter("ignore", tables.NaturalNameWarning)

            with pd.HDFStore(outfile.with_suffix(".h5")) as s:
                for key, filt in self.track_filters.items():
                    s["{}_trc".format(key)] = filt.tracks

        yadict = {}
        for k, v in self.beam_shapes.items():
            if v is None or v.fit_result is None:
                continue
            yadict[k] = v.fit_result

        with outfile.with_suffix(".yaml").open("w") as f:
            io.yaml.safe_dump(dict(beam_shapes=yadict), f,
                              default_flow_style=False)

    def show_statistics(self, key):
        fig, ax = plt.subplots()
        data = self.statistics[key]

        x = [d.op for d in data]
        y = [(d.n_before - d.n_after) / d.n_before for d in data]
        c = ["C0"] * len(x)

        x.append("total")
        y.append(1 - np.prod(1 - np.array(y)))
        c.append("C1")

        for i, v in enumerate(y):
            if not v:
                continue
            ax.text(i, v - 0.02, f"{v:.3f}", ha="center", va="top",
                    color="white", weight="bold")


        ax.bar(x, y, color=c)
        ax.set_xlabel("operation")
        ax.set_ylabel("discarded fraction")
        fig.autofmt_xdate()
