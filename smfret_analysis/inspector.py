# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import suppress

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets
from IPython.display import display
import pims

from sdt import fret, helper

from .tracker import Tracker


class Inspector:
    def __init__(self, file_prefix="tracking"):
        cfg = Tracker.load_data(file_prefix, loc=False)

        self.track_data = cfg["track_data"]
        self.excitation_seq = cfg["tracker"].excitation_seq
        self.rois = cfg["rois"]
        self.data_dir = cfg["data_dir"]

    # Copied from Tracker
    def _pims_open_no_warn(self, f):
        pth = self.data_dir / f
        if pth.suffix.lower() == ".spe":
            # Disable warnings about file size being wrong which is caused
            # by SDT-control not dumping the whole file
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return pims.open(str(pth))
        return pims.open(str(pth))

    def _open_image_sequence(self, f):
        if isinstance(f, tuple):
            don = self._pims_open_no_warn(f[0])
            acc = self._pims_open_no_warn(f[1])
            to_close = [don, acc]
        else:
            don = acc = self._pims_open_no_warn(f)
            to_close = [don]

        ret = {}
        for ims, chan in [(don, "donor"), (acc, "acceptor")]:
            if self.rois[chan] is not None:
                ims = self.rois[chan](ims)
            ret[chan] = ims
        return ret, to_close
    # End of copy

    def make_dataset_selector(self, state, callback):
        d_sel = ipywidgets.Dropdown(options=list(self.track_data.keys()),
                                    description="dataset")

        def change_dataset(key=None):
            trc = self.track_data[d_sel.value]
            state["id"] = d_sel.value
            state["trc"] = trc
            state["pnos"] = trc["fret", "particle"].unique()
            state["files"] = \
                trc.index.remove_unused_levels().levels[0].unique()

            callback()

        d_sel.observe(change_dataset, names="value")
        change_dataset()

        return d_sel

    def mark_track(self):
        state = {}

        fig, (ax_d, ax_a, ax_aa) = plt.subplots(1, 3, figsize=(9, 4))
        p_sel = ipywidgets.IntText(description="particle")
        f_sel = ipywidgets.IntText(description="frame")
        fname_label = ipywidgets.Label()

        def show_track(arg=None):
            particle = state["pnos"][p_sel.value]
            frame = f_sel.value
            df = state["trc"]
            df = df[df["fret", "particle"] == particle]
            fnames = df.iloc[0].name[0]
            fname_label.value = str(fnames)

            d_frames = np.nonzero(self.excitation_seq == "d")[0]
            a_frames = np.nonzero(self.excitation_seq == "a")[0]

            reps = frame // len(d_frames)
            res = frame - reps * len(d_frames)
            fno_d = reps * len(self.excitation_seq) + d_frames[res]
            fno_a = (reps * len(self.excitation_seq) +
                     a_frames[np.nonzero(a_frames > d_frames[res])[0][0]])

            imgs, to_close = self._open_image_sequence(fnames)
            img_dd = imgs["donor"][fno_d]
            img_da = imgs["acceptor"][fno_d]
            img_aa = imgs["acceptor"][fno_a]
            for c in to_close:
                c.close()

            for a in (ax_d, ax_a, ax_aa):
                a.cla()
                a.axis("off")

            ax_d.set_title("donor")
            ax_a.set_title("acceptor")
            ax_aa.set_title("acceptor (direct)")

            if self.rois["donor"] is not None:
                img_dd = self.rois["donor"](img_dd)
            if self.rois["acceptor"] is not None:
                img_da = self.rois["acceptor"](img_da)
                img_aa = self.rois["acceptor"](img_aa)

            ax_d.imshow(img_dd, vmin=img_dd.min())
            ax_a.imshow(img_da, vmin=img_da.min())
            ax_aa.imshow(img_aa, vmin=img_aa.min())

            df_d = df[df["acceptor", "frame"] == fno_d]
            df_a = df[df["acceptor", "frame"] == fno_a]
            mask_d = np.ones(len(df_d), dtype=bool)
            mask_a = np.ones(len(df_a), dtype=bool)
            df_dd = df_d["donor"]
            df_da = df_d["acceptor"]
            df_aa = df_a["acceptor"]

            ax_d.scatter(df_dd.loc[mask_d, "x"],
                         df_dd.loc[mask_d, "y"],
                         s=100, facecolor="none", edgecolor="yellow")
            ax_a.scatter(df_da.loc[mask_d, "x"],
                         df_da.loc[mask_d, "y"],
                         s=100, facecolor="none", edgecolor="yellow")
            ax_aa.scatter(df_aa.loc[mask_a, "x"], df_aa.loc[mask_a, "y"],
                          s=100, facecolor="none", edgecolor="yellow")

            fig.tight_layout()
            fig.canvas.draw()

        p_sel.observe(show_track, "value")
        f_sel.observe(show_track, "value")

        d_sel = self.make_dataset_selector(state, show_track)
        box = ipywidgets.VBox([d_sel, p_sel, f_sel, fig.canvas, fname_label])
        display(box)

    def show_track(self):
        state = {}

        fig, ax = plt.subplots()
        p_sel = ipywidgets.IntText(description="particle")

        def show_track(particle=None):
            ax.cla()
            p = state["pnos"][p_sel.value]
            tr = state["trc"]
            t = tr[tr["fret", "particle"] == p]
            c = ax.plot(t["donor", "x"], t["donor", "y"], marker=".")
            fig.canvas.draw()

        p_sel.observe(show_track, names="value")

        d_sel = self.make_dataset_selector(state, show_track)
        box = ipywidgets.VBox([d_sel, p_sel, fig.canvas])
        display(box)

    def raw_features(self, figsize=(8, 8), n_cols=8, img_size=3):
        state = {}

        fig = plt.figure(figsize=(8, 8))
        p_sel = ipywidgets.IntText(description="particle")

        def draw(particle=None):
            fig.clf()
            tr = state["trc"]
            t0 = tr[tr["fret", "particle"] == state["pnos"][p_sel.value]]
            fnames = t0.index[0][0]

            imgs, to_close = self._open_image_sequence(fnames)
            don_img = imgs["donor"]
            acc_img = imgs["acceptor"]
            if self.rois["donor"] is not None:
                don_img = self.rois["donor"](don_img)
            if self.rois["acceptor"] is not None:
                acc_img = self.rois["acceptor"](acc_img)
            fret.draw_track(t0, p_sel.value, don_img, acc_img, img_size,
                            n_cols=n_cols, figure=fig)
            for c in to_close:
                c.close()

            fig.canvas.draw()

        p_sel.observe(draw, names="value")

        d_sel = self.make_dataset_selector(state, draw)
        box = ipywidgets.VBox([d_sel, p_sel, fig.canvas])
        display(box)

    def show_all_tracks(self):
        state = {}
        fig, ax = plt.subplots(figsize=(8, 8))
        f_sel = ipywidgets.Dropdown(description="file")

        def fill_f_sel():
            f_sel.options = [(str(f), f) for f in state["files"]]

        def plot(file=None):
            ax.cla()
            d = state["trc"].loc[f_sel.value]
            for p, trc in helper.split_dataframe(
                    d, ("fret", "particle"),
                    [("donor", "x"), ("donor", "y")]):
                ax.plot(trc[:, 0], trc[:, 1])
                ax.text(*trc[0], str(p))
            fig.canvas.draw()

        f_sel.observe(plot, "value")

        d_sel = self.make_dataset_selector(state, fill_f_sel)
        box = ipywidgets.VBox([d_sel, f_sel, fig.canvas])
        display(box)

    def plot_tracks(self, *axes):
        state = {}

        p_sel = ipywidgets.Text(description="particle")
        fig, ax = plt.subplots(1, len(axes), figsize=(8, 4))
        info_label = ipywidgets.HTML()

        def plot_data(particle=None):
            for a in ax:
                a.cla()
                a.grid()

            particles = []
            for i in p_sel.value.split():
                with suppress(ValueError):
                    particles.append(int(i))

            info = []

            data = state["trc"]
            for p in particles:
                d = data[(data["fret", "particle"] == p)]
                for a, (x, y) in zip(ax, axes):
                    a.plot(d[x], d[y], ".-")
                    a.set_xlabel(" ".join(x))
                    a.set_ylabel(" ".join(y))
                info.append("particle {}: start {}, end {}".format(
                    p, d["donor", "frame"].min(), d["donor", "frame"].max()))

            info_label.value = "<br />".join(info)

            fig.tight_layout()
            fig.canvas.draw()

        p_sel.observe(plot_data, names="value")

        d_sel = self.make_dataset_selector(state, plot_data)
        box = ipywidgets.VBox([d_sel, p_sel, fig.canvas, info_label])
        display(box)
