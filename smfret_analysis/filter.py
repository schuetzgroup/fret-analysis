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
import pims
import cv2

from sdt import io, roi, fret, chromatic, flatfield, changepoint, helper

from .version import output_version
from .tracker import Tracker


_bokeh_js_loaded = False


def cell_mask_a_thresh(img, block_size, c, smoothing=(5, 5)):
    scaled = img - img.min()
    scaled = scaled / scaled.max() * 255
    blur = cv2.GaussianBlur(scaled.astype(np.uint8), smoothing, 0)
    mask = cv2.adaptiveThreshold(blur, 1, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 2*block_size+1, c)
    return mask.astype(bool)


def cell_mask_otsu(img, factor, smoothing=(5, 5)):
    scaled = img - img.min()
    scaled = scaled / scaled.max() * 255
    blur = cv2.GaussianBlur(scaled.astype(np.uint8), smoothing, 0)
    thresh, mask = cv2.threshold(blur, 0, 1,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh2, mask = cv2.threshold(blur, thresh * factor, 1, cv2.THRESH_BINARY)
    return mask.astype(bool)


class StatItem:
    def __init__(self, op, n_before, n_after):
        self.op = op
        self.n_before = n_before
        self.n_after = n_after


class Filter:
    def __init__(self, file_prefix="tracking"):
        tr = Tracker.load(file_prefix, loc=False)
        self.rois = tr.rois
        self.cc = tr.tracker.chromatic_corr
        self.track_filters = {k: fret.SmFretFilter(v)
                              for k, v in tr.track_data.items()}
        self.exc_scheme = "".join(tr.tracker.excitation_seq)
        self.data_dir = tr.data_dir
        self.sources = tr.sources
        self.cell_images = tr.cell_images
        self.profile_images = tr.profile_images

        self.beam_shapes = {"donor": None, "acceptor": None}

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

    def find_acc_bleach_options(self, key):
        @ipywidgets.interact(p=ipywidgets.IntText(0),
                             pen=ipywidgets.FloatText(2e7))
        def show_track(p, pen):
            f = self.track_filters[key]
            t = f.tracks
            t = t[t["fret", "particle"] == p]
            td = t[t["fret", "exc_type"] ==
                   fret.SmFretTracker.exc_type_nums["d"]]
            ta = t[t["fret", "exc_type"] ==
                   fret.SmFretTracker.exc_type_nums["a"]]

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            fd = td["donor", "frame"].values
            dm = td["fret", "d_mass"].values
            fa = ta["donor", "frame"].values
            am = ta["fret", "a_mass"].values
            ef = td["fret", "eff"].values
            hn = td["fret", "has_neighbor"].values

            cp_a = f.cp_detector.find_changepoints(am, pen) - 1
            cp_d = np.searchsorted(fd, fa[cp_a]) - 1
            if len(fd):
                changepoint.plot_changepoints(dm, cp_d, time=fd, ax=ax[0])
                changepoint.plot_changepoints(ef, cp_d, time=fd, ax=ax[2])
            if len(fa):
                changepoint.plot_changepoints(am, cp_a, time=fa, ax=ax[1])
            axt1 = ax[1].twinx()
            axt1.plot(fd, hn, c="C2", alpha=0.2)
            axt1.set_ylim(-0.05, 1.05)

            ax[0].set_title("d_mass")
            ax[1].set_title("a_mass")
            ax[2].set_title("eff")

            for a in ax:
                a.grid()

            fig.tight_layout()
            plt.show()

    def filter_acc_bleach(self, cp_penalty, brightness_thresh):
        for k, f in self.track_filters.items():
            b = len(f.tracks)
            f.acceptor_bleach_step(brightness_thresh, truncate=True,
                                   penalty=cp_penalty)
            a = len(f.tracks)
            self.statistics[k].append(StatItem("acc bleach", b, a))

    def find_brightness_params(self, key, frame=None):
        import bokeh.plotting
        import bokeh.application as b_app
        import bokeh.application.handlers as b_hnd

        global _bokeh_js_loaded
        if not _bokeh_js_loaded:
            bokeh.plotting.output_notebook(bokeh.resources.INLINE,
                                           hide_banner=True)
            _bokeh_js_loaded = True

        frame = self.exc_scheme.find("d") if frame is None else frame
        dat = self.track_filters[key].tracks
        dat0 = dat[(dat["fret", "has_neighbor"] == 0) &
                   (dat["donor", "frame"] == frame)]

        ds = bokeh.models.ColumnDataSource(dat0)
        ds.data["file"] = [i[0] for i in ds.data["index"]]
        ds_all = bokeh.models.ColumnDataSource(
            dict(xs=[], xs_d=[], ys=[], file=[], particle=[], don_mass=[],
                 acc_mass=[], has_neighbor=[]))

        def update(attr, old, new):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",
                                      bokeh.util.warnings.BokehUserWarning)
                inds = np.array(new["1d"]["indices"])
                if len(inds) == 0:
                    return
                xs = []
                xs_d = []
                ys = []
                files = []
                particles = []
                don_mass = []
                acc_mass = []
                has_neigh = []
                for i in inds:
                    row = dat0.iloc[i]
                    file = row.name[0]
                    pno = row["fret", "particle"]
                    d = dat.loc[file]
                    d = d[d["fret", "particle"] == pno]
                    m = np.isfinite(d["fret", "eff"])
                    xs.append(d["donor", "frame"])
                    xs_d.append(d["donor", "frame"].values[m])
                    ys.append(d["fret", "eff"].values[m])
                    files.append(file)
                    particles.append(pno)
                    don_mass.append(d["donor", "mass"].values)
                    acc_mass.append(d["acceptor", "mass"].values)
                    has_neigh.append(d["fret", "has_neighbor"].values)

                ds_all.data["xs"] = xs
                ds_all.data["xs_d"] = xs_d
                ds_all.data["ys"] = ys
                ds_all.data["file"] = files
                ds_all.data["particle"] = particles
                ds_all.data["don_mass"] = don_mass
                ds_all.data["acc_mass"] = acc_mass
                ds_all.data["has_neighbor"] = has_neigh

        def modify_doc(doc):
            tools = "pan, wheel_zoom, box_select, lasso_select, reset, tap"
            fig_opts = dict(plot_width=300, plot_height=300, tools=tools)
            l_fig = bokeh.plotting.figure(**fig_opts)
            l = l_fig.scatter("fret_eff", "fret_stoi", source=ds)

            ht = bokeh.models.HoverTool(
                tooltips=[("file", "@file{%s}"),
                          ("particle", "@fret_particle")],
                formatters={"file": "printf"})
            l_fig.add_tools(ht)

            r_fig = bokeh.plotting.figure(**fig_opts)
            r = r_fig.scatter("fret_d_mass", "fret_a_mass", source=ds)
            r_fig.add_tools(ht)

            b_fig = bokeh.plotting.figure(plot_width=2*fig_opts["plot_width"],
                                          plot_height=100, tools="reset")
            b_fig.extra_y_ranges = {"eff": bokeh.models.Range1d(-0.01, 1.01)}
            b_fig.multi_line("xs_d", "ys", source=ds_all, y_range_name="eff")
            b_fig.multi_line("xs", "has_neighbor", source=ds_all,
                             y_range_name="eff", color="black")
            b_fig.multi_line("xs", "don_mass", source=ds_all, color="green")
            b_fig.multi_line("xs", "acc_mass", source=ds_all, color="red")
            ht = bokeh.models.HoverTool(
                tooltips=[("file", "@file{%s}"), ("particle", "@particle")],
                formatters={"file": "printf"})
            b_fig.add_tools(ht)
            b_fig.add_layout(bokeh.models.LinearAxis(y_range_name="eff"),
                             "right")

            layout = bokeh.layouts.column(
                [bokeh.layouts.row([l_fig, r_fig]),
                b_fig])
            doc.add_root(layout)

            ds.on_change("selected", update)

            return doc

        hnd = b_hnd.FunctionHandler(modify_doc)
        app = b_app.Application(hnd)

        bokeh.plotting.show(app)

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

    def load_cell_mask(self, file, return_img=False, method=cell_mask_a_thresh,
                       **kwargs):
        if isinstance(method, str):
            method = globals()["cell_mask_" + method]
        img = self.cell_images[file][0]  # Use first cell image
        mask = method(img, **kwargs)
        if return_img:
            return mask, img
        else:
            return mask

    def _plot_cell_thresh(self, mask, img):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        i1 = img.copy()
        i1[~mask] = 0
        i2 = img.copy()
        i2[mask] = 0
        imax = img.max() / 2
        ax1.imshow(i1, vmax=imax)
        ax2.imshow(i2, vmax=imax)

        fig.tight_layout()
        plt.show()

    def find_cell_mask_params(self, key, method="a_thresh"):
        fw = ipywidgets.Dropdown(
            options=self.track_filters[key].tracks.index.levels[0].unique())

        if method == "a_thresh":
            bs = ipywidgets.IntText(value=65)
            cw = ipywidgets.IntText(value=-5)

            @ipywidgets.interact(file=fw, block_size=bs, c=cw)
            def show_cell(file, block_size, c):
                mask, img = self.load_cell_mask(
                    file, return_img=True, method=method,
                    block_size=block_size, c=c)
                self._plot_cell_thresh(mask, img)
        elif method == "otsu":
            faw = ipywidgets.FloatText(value=1., step=0.01)

            @ipywidgets.interact(file=fw, factor=faw)
            def show_cell(file, factor):
                mask, img = self.load_cell_mask(
                    file, return_img=True, method=method, factor=factor)
                self._plot_cell_thresh(mask, img)

    def apply_cell_masks(self, method="a_thresh", **kwargs):
        for k, v in self.sources.items():
            filt = self.track_filters[k]
            b = len(filt.tracks)

            if v["cells"]:
                trc = filt.tracks

                files = np.unique(trc.index.levels[0])
                mask = [(f, self.load_cell_mask(f, method=method, **kwargs))
                        for f in files]
                filt.image_mask(mask, channel="donor")

            a = len(filt.tracks)
            self.statistics[k].append(StatItem("cell mask", b, a))

    def find_beam_shape_thresh(self, channel):
        bs = self.beam_shapes[channel]

        tw = ipywidgets.BoundedIntText(value=75, min=0, max=100)
        @ipywidgets.interact(thresh=tw)
        def show_cell(thresh):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(bs.corr_img)
            ax2.imshow(bs.corr_img * 100 > thresh)

            fig.tight_layout()
            plt.show()

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
            res = {k2: float(v2)
                   for k2, v2 in v.fit_result.best_values.items()}
            yadict[k] = res

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
