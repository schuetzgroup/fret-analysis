import contextlib
import functools
import warnings
import collections

import numpy as np
from scipy import ndimage
import pandas as pd
import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import pims

from sdt import io, roi, fret, chromatic, beam_shape, changepoint, helper

from .version import output_version
from .tracker import Tracker


_bokeh_js_loaded = False


def get_cell_region(img, percentile=85):
    r = ndimage.binary_opening(img > np.percentile(img, percentile))
    return ndimage.binary_fill_holes(r)


class Filter:
    def __init__(self, file_prefix="tracking", data_dir=""):
        tr = Tracker.load(file_prefix, loc=False, data_dir=data_dir)
        self.rois = tr.rois
        self.cc = tr.tracker.chromatic_corr
        self.track_filters = {k: fret.SmFretFilter(v)
                              for k, v in tr.track_data.items()}
        self.exc_scheme = "".join(tr.tracker.excitation_seq)
        self.data_dir = tr.data_dir

        self.beam_shapes = {"donor": None, "acceptor": None}

    def calc_beam_shape_sm(self, keys="all", channel="donor", weighted=True):
        if not len(keys):
            return

        if channel.startswith("d"):
            k = "donor"
            mk = "d_mass"
            f = self.exc_scheme.find("d")
        elif channel.startswith("a"):
            k = "acceptor"
            mk = "a_mass"
            f = self.exc_scheme.find("a")
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
        bs = beam_shape.Corrector(
            *data, pos_columns=[(k, "x"), (k, "y")],
            mass_column=("fret", mk), shape=img_shape,
            density_weight=weighted)

        self.beam_shapes[k] = bs

    def calc_beam_shape_bulk(self, files, channel="donor", gaussian_fit=True,
                             frame=0):
        imgs = []
        roi = self.rois[channel]

        for f in files:
            with pims.open(str(self.data_dir / f)) as fr:
                imgs.append(roi(f[frame]))

        self.beam_shapes[channel] = beam_shape.Corrector(
            imgs, gaussian_fit=gaussian_fit)

    def find_acc_bleach_options(self, key):
        @ipywidgets.interact(p=ipywidgets.IntText(0),
                             pen=ipywidgets.FloatText(1e6))
        def show_track(p, pen):
            f = self.track_filters[key]
            t = f.tracks
            t = t[t["fret", "particle"] == p]
            t = t[t["fret", "exc_type"] ==
                  fret.SmFretTracker.exc_type_nums["d"]]

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            fs = t["donor", "frame"].values
            dm = t["fret", "d_mass"].values
            am = t["fret", "a_mass"].values
            ef = t["fret", "eff"].values
            hn = t["fret", "has_neighbor"].values

            cp_a = f.cp_detector.find_changepoints(am, pen)
            changepoint.plot_changepoints(dm, cp_a, ax=ax[0])
            changepoint.plot_changepoints(am, cp_a, ax=ax[1])
            changepoint.plot_changepoints(ef, cp_a, ax=ax[2])
            axt1 = ax[1].twinx()
            axt1.plot(hn, c="C2", alpha=0.2)
            axt1.set_ylim(-0.05, 1.05)

            ax[0].set_title("d_mass")
            ax[1].set_title("a_mass")
            ax[2].set_title("eff")

            for a in ax:
                a.grid()

            fig.tight_layout()
            plt.show()

    def filter_acc_bleach(self, cp_penalty, brightness_thresh):
        for f in self.track_filters.values():
            f.acceptor_bleach_step(brightness_thresh, truncate=True,
                                   penalty=cp_penalty)

    def find_brightness_params(self, key):
        import bokeh.plotting
        import bokeh.application as b_app
        import bokeh.application.handlers as b_hnd

        global _bokeh_js_loaded
        if not _bokeh_js_loaded:
            bokeh.plotting.output_notebook(bokeh.resources.INLINE,
                                           hide_banner=True)
            _bokeh_js_loaded = True

        dat = self.track_filters[key].tracks
        dat0 = dat[(dat["fret", "has_neighbor"] == 0) &
                   (dat["donor", "frame"] == self.exc_scheme.find("d"))]

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
        for f in self.track_filters.values():
            f.query(expr, mi_sep)

    def load_cell_mask(self, file, percentile, return_img=False):
        frame_no = self.exc_scheme.find("o")
        with pims.open(str(self.data_dir / file)) as fr:
            img = self.rois["donor"](fr[frame_no])
        mask = get_cell_region(img, percentile)
        if return_img:
            return mask, img
        else:
            return mask

    def find_cell_thresh(self, key):
        fw = ipywidgets.Dropdown(
            options=self.track_filters[key].tracks.index.levels[0].unique())
        pw = ipywidgets.BoundedIntText(value=85, min=0, max=100)

        @ipywidgets.interact(file=fw, percentile=pw)
        def show_cell(file, percentile):
            mask, img = self.load_cell_mask(file, percentile, return_img=True)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img)
            ax2.imshow(mask)

            fig.tight_layout()
            plt.show()

    def filter_cell_region(self, key, percentile):
        if percentile <= 0:
            return

        f = self.track_filters[key]
        trc = f.tracks

        files = np.unique(trc.index.levels[0])
        mask = [(v, self.load_cell_mask(v, percentile)) for v in files]
        f.image_mask(mask, channel="donor")

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
        for v in self.track_filters.values():
            v.image_mask(mask, channel)

    def save_data(self, file_prefix="filtered"):
        outfile = self.data_dir / f"{file_prefix}-v{output_version:03}"
        with pd.HDFStore(outfile.with_suffix(".h5")) as s:
            for key, filt in self.track_filters.items():
                s["{}_trc".format(key)] = filt.tracks

        yadict = {}
        for k, v in self.beam_shapes.items():
            if v is None:
                continue
            res = {k2: float(v2)
                   for k2, v2 in v.fit_result.best_values.items()}
            yadict[k] = res

        with outfile.with_suffix(".yaml").open("w") as f:
            io.yaml.safe_dump(dict(beam_shapes=yadict), f,
                              default_flow_style=False)
