import contextlib
import functools
import warnings
import collections
from pathlib import Path
import itertools

import numpy as np
from scipy import ndimage
import pandas as pd
import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sdt import io, roi, fret, helper, nbui, image, plot, changepoint

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
        self.excitation_seq = cfg["excitation_seq"]
        self.analyzers = {k: fret.SmFretAnalyzer(v, self.excitation_seq)
                          for k, v in cfg["track_data"].items()}
        self.data_dir = cfg["data_dir"]
        self.sources = cfg["sources"]
        self.cell_images = cfg["cell_images"]
        self.flatfield = cfg["flatfield"]

        self._segment_fig = None
        self._brightness_fig = None
        self._thresholder = None
        self._beam_shape_fig = None
        self._beam_shape_artists = [None, None]
        self._beta_population_fig = None
        self._eff_stoi_fig = None

    def flag_excitation_type(self):
        for a in self.analyzers.values():
            a.flag_excitation_type()

    def present_at_start(self, frame=None):
        frame = self.excitation_seq.find("d") if frame is None else frame
        for a in self.analyzers.values():
            a.query_particles(f"donor_frame == {frame}")

    def find_beam_shape_thresh(self):
        if self._beam_shape_fig is None:
            fig = plt.figure()
            fig.add_subplot(1, 2, 1)
            fig.add_subplot(1, 2, 2)

            self._beam_shape_fig = fig

        ax = self._beam_shape_fig.axes

        channel_sel = ipywidgets.Dropdown(options=list(self.flatfield),
                                          description="channel")
        thresh_sel = ipywidgets.BoundedIntText(value=50, min=0, max=100,
                                               description="threshold")

        def channel_changed(change=None):
            bs = self.flatfield[channel_sel.value]
            if self._beam_shape_artists[0] is not None:
                self._beam_shape_artists[0].remove()
            self._beam_shape_artists[0] = ax[0].imshow(bs.corr_img)
            update()

        def update(change=None):
            bs = self.flatfield[channel_sel.value]
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
        mask = self.flatfield[channel].corr_img * 100 > thresh
        for a in self.analyzers.values():
            a.image_mask(mask, channel)

    def flatfield_correction(self):
        for a in self.analyzers.values():
            a.flatfield_correction(self.flatfield["donor"],
                                   self.flatfield["acceptor"])

    def calc_fret_values(self, *args, **kwargs):
        for a in self.analyzers.values():
            a.calc_fret_values(*args, **kwargs)

    def _make_dataset_selector(self, state, callback):
        d_sel = ipywidgets.Dropdown(options=list(self.analyzers.keys()),
                                    description="dataset")

        def change_dataset(key=None):
            trc = self.analyzers[d_sel.value].tracks
            state["id"] = d_sel.value
            state["trc"] = trc
            state["pnos"] = sorted(trc["fret", "particle"].unique())
            state["files"] = \
                trc.index.remove_unused_levels().levels[0].unique()

            callback()

        d_sel.observe(change_dataset, names="value")
        change_dataset()

        return d_sel

    def find_segment_a_mass_options(self):
        state = {}

        p_sel = ipywidgets.IntText(value=0, description="particle")
        p_label = ipywidgets.Label()
        pen_sel = ipywidgets.FloatText(value=2e7, description="penalty")

        if self._segment_fig is None:
            self._segment_fig, ax = plt.subplots(1, 3, figsize=(8, 4))
            ax[1].twinx()

        ax_dm, ax_am, ax_eff, ax_hn = self._segment_fig.axes

        def show_track(change=None):
            for a in self._segment_fig.axes:
                a.cla()

            pi = state["pnos"][p_sel.value]
            p_label.value = f"Real particle number: {pi}"
            trc = state["trc"]
            t = trc[trc["fret", "particle"] == pi]
            td = t[t["fret", "exc_type"] == "d"]
            ta = t[t["fret", "exc_type"] == "a"]

            fd = td["donor", "frame"].values
            dm = td["fret", "d_mass"].values
            fa = ta["donor", "frame"].values
            am = ta["fret", "a_mass"].values
            ef = td["fret", "eff_app"].values
            hn = td["fret", "has_neighbor"].values

            cp_a = self.analyzers[state["id"]].cp_detector.find_changepoints(
                am, pen_sel.value) - 1
            cp_d = np.searchsorted(fd, fa[cp_a]) - 1
            if len(fd):
                changepoint.plot_changepoints(dm, cp_d, time=fd, ax=ax_dm)
                changepoint.plot_changepoints(ef, cp_d, time=fd, ax=ax_eff)
            if len(fa):
                changepoint.plot_changepoints(am, cp_a, time=fa, ax=ax_am)
            ax_hn.plot(fd, hn, c="C2", alpha=0.2)

            ax_dm.relim(visible_only=True)
            ax_am.relim(visible_only=True)
            ax_eff.set_ylim(-0.05, 1.05)
            ax_hn.set_ylim(-0.05, 1.05)

            ax_dm.set_title("d_mass")
            ax_am.set_title("a_mass")
            ax_eff.set_title("app. eff")

            self._segment_fig.tight_layout()
            self._segment_fig.canvas.draw()

        p_sel.observe(show_track, "value")
        pen_sel.observe(show_track, "value")

        d_sel = self._make_dataset_selector(state, show_track)

        return ipywidgets.VBox([d_sel, p_sel, pen_sel,
                                self._segment_fig.canvas, p_label])

    def segment_a_mass(self, **kwargs):
        for a in self.analyzers.values():
            a.segment_a_mass(**kwargs)

    def filter_acc_bleach(self, brightness_thresh):
        for a in self.analyzers.values():
            a.acceptor_bleach_step(brightness_thresh, truncate=False)

    def plot_eff_vs_stoi(self):
        if self._eff_stoi_fig is None:
            self._eff_stoi_fig, _ = plt.subplots()

        state = {}

        def update(change=None):
            data = state["trc"].loc[:, [("fret", "eff"), ("fret", "stoi")]]
            data = data.values
            data = data[np.all(np.isfinite(data), axis=1)]

            ax = self._eff_stoi_fig.axes[0]
            ax.cla()
            plot.density_scatter(*data.T, ax=ax)
            ax.set_xlabel("apparent eff.")
            ax.set_ylabel("apparent stoi.")

            self._eff_stoi_fig.canvas.draw()

        d_sel = self._make_dataset_selector(state, update)

        return ipywidgets.VBox([d_sel, self._eff_stoi_fig.canvas])

    def find_excitation_eff_component(self):
        if self._beta_population_fig is None:
            self._beta_population_fig, _ = plt.subplots()

        state = {}

        n_comp_sel = ipywidgets.IntText(description="components", value=1)

        cmaps = ["viridis", "gray", "plasma", "jet"]
        def update(change=None):
            d = state["trc"]
            d = d[d["fret", "a_seg"] == 0].copy()
            split = fret.gaussian_mixture_split(d, n_comp_sel.value)

            ax = self._beta_population_fig.axes[0]
            ax.cla()
            for v, cm in zip(split, itertools.cycle(cmaps)):
                sel = (d["fret", "particle"].isin(v) &
                       np.isfinite(d["fret", "eff_app"]) &
                       np.isfinite(d["fret", "stoi_app"]))
                plot.density_scatter(d.loc[sel, ("fret", "eff_app")],
                                     d.loc[sel, ("fret", "stoi_app")],
                                    marker=".", label=f"{len(v)}", ax=ax,
                                    cmap=cm)
            ax.legend(loc=0)
            ax.set_xlabel("apparent eff.")
            ax.set_ylabel("apparent stoi.")
            self._beta_population_fig.canvas.draw()

        d_sel = self._make_dataset_selector(state, update)
        n_comp_sel.observe(update, "value")

        return ipywidgets.VBox([d_sel, n_comp_sel,
                                self._beta_population_fig.canvas])

    def set_leakage(self, leakage):
        for a in self.analyzers.values():
            a.leakage = leakage

    def calc_leakage(self, dataset):
        a = self.analyzers[dataset]
        a.calc_leakage()
        self.set_leakage(a.leakage)

    def set_direct_excitation(self, dir_exc):
        for a in self.analyzers.values():
            a.direct_excitation = dir_exc

    def calc_direct_excitation(self, dataset):
        a = self.analyzers[dataset]
        a.calc_direct_excitation()
        self.set_leakage(a.direct_excitation)

    def set_detection_eff(self, eff):
        for a in self.analyzers.values():
            a.detection_eff = eff

    def calc_detection_eff(self, min_part_len=5, how="individual",
                           aggregate="dataset", dataset=None):
        if callable(how):
            if dataset is not None:
                a = self.analyzers[dataset]
                a.calc_detection_eff(min_part_len, how)
                self.set_detection_eff(a.detection_eff)
            elif aggregate == "dataset":
                for a in self.analyzers.values():
                    a.calc_detection_eff(min_part_len, how)
            elif aggregate == "all":
                effs = []
                for a in self.analyzers.values():
                    a.calc_detection_eff(min_part_len, "individual")
                    effs.append(a.detection_eff)
                self.set_detection_eff(how(pd.concat(effs)))
            else:
                raise ValueError(
                    "`aggregate` must be in {\"dataset\", \"all\"}.")
        elif how == "individual":
            for a in self.analyzers.values():
                a.calc_detection_eff(min_part_len, how)
        else:
            raise ValueError(
                "`how` must be either callable or \"individual\".")

    def set_excitation_eff(self, eff):
        for a in self.analyzers.values():
            a.excitation_eff = eff

    def calc_excitation_eff(self, dataset, n_components=1, component=0):
        a = self.analyzers[dataset]
        a.calc_excitation_eff(n_components, component)
        self.set_excitation_eff(a.excitation_eff)

    def fret_correction(self, *args, **kwargs):
        for a in self.analyzers.values():
            a.fret_correction(*args, **kwargs)

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
            value=self.excitation_seq.find("d") if frame is None else frame,
            description="frame")

        def set_frame(change=None):
            trc = state["trc"]
            sel = trc["fret", "has_neighbor"] == 0
            if f_sel.value >= 0:
                sel &= trc["donor", "frame"] == f_sel.value
            state["cur_trc"] = trc[sel]

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
