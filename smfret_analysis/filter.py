import contextlib
import functools

import numpy as np
from scipy import ndimage
import pandas as pd
import ruptures
import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import pims

import bokeh.plotting
import bokeh.application as b_app
import bokeh.application.handlers as b_hnd
bokeh.plotting.output_notebook(bokeh.resources.INLINE)

from sdt import io, roi, fret, chromatic, beam_shape

from .version import output_version


def get_cell_region(img, percentile=85):
    r = ndimage.binary_opening(img > np.percentile(img, percentile))
    return ndimage.binary_fill_holes(r)


def select_donor_region(fret_data, region):
    xy = fret_data.loc[:, [("donor", "x"), ("donor", "y")]].values
    x, y = xy.astype(int).T
    in_bounds = ((x >= 0) & (y >= 0) &
                 (x < region.shape[1]) & (y < region.shape[0]))
    fret_data = fret_data[in_bounds]
    return fret_data[region[y[in_bounds], x[in_bounds]]]


class Filter:
    def __init__(self, file_prefix="tracking"):
        with open(f"{file_prefix}-v{output_version:03}.yaml") as f:
            tracking_meta = io.yaml.safe_load(f)

        self.don_roi = tracking_meta["rois"]["donor"]
        self.acc_roi = tracking_meta["rois"]["acceptor"]

        self.img = tracking_meta["files"]
        self.cc = chromatic.Corrector.load("chromatic.npz")
        self.ana = fret.SmFretAnalyzer(tracking_meta["excitation_scheme"])

        with pd.HDFStore(f"{file_prefix}-v{output_version:03}.h5", "r") as s:
            self.track_data = {k: s[f"{k}_trc"] for k in self.img}

        self.cp_det = ruptures.Pelt("l2", min_size=1, jump=1)
        self.cp = {}

        self.calc_donor_beamshape()

    def calc_donor_beamshape(self):
        img_shape = (self.don_roi.bottom_right[1] - self.don_roi.top_left[1],
                     self.don_roi.bottom_right[0] - self.don_roi.top_left[0])
        data = list(self.track_data.values())
        data = [d[d["donor", "frame"] == 0] for d in data]
        self.beam_shape = beam_shape.Corrector(
            *data, pos_columns=[("donor", "x"), ("donor", "y")],
            mass_column=("fret", "d_mass"), shape=img_shape)

    def find_acc_bleach_options(self, key):
        @ipywidgets.interact(p=ipywidgets.IntText(0),
                             pen=ipywidgets.FloatText(1e6))
        def show_track(p, pen):
            t = self.track_data[key]
            t = t[t["fret", "particle"] == p]
            t = self.ana.get_excitation_type(t, "d")

            fig, ax = plt.subplots(1, 3, figsize=(12, 4))
            fs = t["donor", "frame"].values
            dm = t["fret", "d_mass"].values
            am = t["fret", "a_mass"].values
            ef = t["fret", "eff"].values
            hn = t["fret", "has_neighbor"].values

            with contextlib.suppress(Exception):
                cp_d = self.cp_det.fit_predict(dm, pen)
                # FIXME: This depends on custom modifications to ruptures
                ruptures.show.display(dm, [], cp_d, ax=ax[0], marker="x")
                axt0 = ax[0].twinx()
                axt0.plot(hn, c="C2", alpha=0.2)
                axt0.set_ylim(-0.05, 1.05)

            with contextlib.suppress(Exception):
                cp_a = self.cp_det.fit_predict(am, pen)
                # FIXME: This depends on custom modifications to ruptures
                ruptures.show.display(am, [], cp_a, ax=ax[1], marker="x")
                ruptures.show.display(ef, [], cp_a, ax=ax[2], marker="x")
                axt1 = ax[1].twinx()
                axt1.plot(hn, c="C2", alpha=0.2)
                axt1.set_ylim(-0.05, 1.05)

            #c, s = steps[p]
            #c = fs[c]
            #c = [fs.min()] + c.tolist() + [fs.max()]
            #s = [s[0]] + s
            #ax[1].plot(fs, am)
            #ax[1].step(c, s, c="C3")
            #axt1 = ax[1].twinx()
            #axt1.set_ylim(-0.05, 1.05)
            #axt1.plot(fs, hn, c="C2", alpha=0.2)
            ax[0].set_title("d_mass")
            ax[1].set_title("a_mass")
            ax[2].set_title("eff")

            for a in ax:
                a.grid()

            fig.tight_layout()
            plt.show()

    def filter_acc_bleach(self, cp_penalty, brightness_thresh):
        for key in self.track_data:
            trc = self.track_data[key]
            ps = trc["fret", "particle"].unique()
            n = len(ps)
            prog = 1
            prog_text = ipywidgets.Label("Starting…")
            display(prog_text)

            res = []
            for p in ps:
                prog_text.value = f"Filtering particle {p} ({prog}/{n})"
                prog += 1

                trc_p = trc[trc["fret", "particle"] == p]
                trc_p = trc_p.sort_values(("donor", "frame"))

                m = self.ana.get_excitation_type(trc_p, "d")
                m_a = m["fret", "a_mass"].values
                f_a = m["donor", "frame"].values

                # Find changepoints
                try:
                    cp = self.cp_det.fit_predict(m_a, cp_penalty)
                except Exception as e:
                    continue

                if len(cp) and cp[-1] == len(m):
                    cp.pop(-1)
                if not len(cp):
                    continue

                # Make step function
                s = np.split(m_a, cp)
                s = [np.median(s_) for s_ in s]

                # See if only the first step is above brightness_thresh
                if not all(s_ < brightness_thresh for s_ in s[1:]):
                    continue

                # Remove data after bleach step
                res.append(
                    trc_p[trc_p["donor", "frame"] < f_a[max(0, cp[0] - 1)]])

            if len(res):
                self.track_data[key] = pd.concat(res)
            else:
                self.track_data[key] = trc.iloc[:0].copy()

    def find_brightness_params(self, key):
        dat = self.track_data[key]
        dat0 = dat[(dat["fret", "has_neighbor"] == 0) &
                   (dat["donor", "frame"] == 0)]

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

    def filter_brightness(self, d_mass=(-np.inf, np.inf),
                          a_mass=(-np.inf, np.inf), stoi=(-np.inf, np.inf),
                          filter_neighbors=True):
        for key in self.track_data:
            trc = self.track_data[key]

            if not len(trc):
                continue

            mask = [
                trc["fret", "d_mass"] > d_mass[0],
                trc["fret", "d_mass"] < d_mass[1],
                trc["fret", "a_mass"] > a_mass[0],
                trc["fret", "a_mass"] < a_mass[1],
                trc["fret", "stoi"] > stoi[0],
                trc["fret", "stoi"] < stoi[1]
            ]
            mask = functools.reduce(np.bitwise_and, mask)

            if filter_neighbors:
                mask &= (trc["fret", "has_neighbor"] == 0)

            trc = trc[mask]
            good_p = trc.loc[trc["donor", "frame"] == 0,
                             ("fret", "particle")].unique()

            self.track_data[key] = trc[trc["fret", "particle"].isin(good_p)]

    def find_cell_thresh(self, key):
        fw = ipywidgets.Dropdown(
            options=self.track_data[key].index.levels[0].unique())
        pw = ipywidgets.BoundedIntText(value=85, min=0, max=100)

        @ipywidgets.interact(file=fw, percentile=pw)
        def show_cell(file, percentile):
            with pims.open(file) as fr:
                img = self.don_roi(fr[0])
            mask = get_cell_region(img, percentile)

            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img)
            ax2.imshow(mask)

            fig.tight_layout()
            plt.show()

    def filter_cell_region(self, key, thresh):
        trc = self.track_data[key]
        if thresh > 0:
            trc_filtered = []
            for f in np.unique(trc.index.levels[0]):
                with pims.open(f) as fr:
                    r = get_cell_region(self.don_roi(fr[0]), thresh)
                trc_filtered.append(select_donor_region(trc, r))

            self.track_data[key] = pd.concat(trc_filtered)

    def find_beam_shape_thresh(self):
        tw = ipywidgets.BoundedIntText(value=75, min=0, max=100)
        @ipywidgets.interact(thresh=tw)
        def show_cell(thresh):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(self.beam_shape.corr_img)
            ax2.imshow(self.beam_shape.corr_img > thresh / 100)

            fig.tight_layout()
            plt.show()

    def filter_beam_shape_region(self, thresh):
        mask = self.beam_shape.corr_img > thresh / 100
        for key in self.track_data:
            trc = self.track_data[key]
            self.track_data[key] = select_donor_region(trc, mask)

    def save_data(self, file_prefix="filtered"):
        with pd.HDFStore(f"{file_prefix}-v{output_version:03}.h5") as s:
            for key, trc in self.track_data.items():
                s[f"{key}_trc"] = trc

        with open(f"{file_prefix}-v{output_version:03}.yaml", "w") as f:
            v = {k: float(v)
                 for k, v in self.beam_shape.fit_result.best_values.items()}
            io.yaml.safe_dump(dict(beam_shape_fit=v), f,
                              default_flow_style=False)

    def show_raw_track(self, key):
        @ipywidgets.interact(
            particle=ipywidgets.BoundedIntText(min=0, value=0),
            eff_thresh=ipywidgets.BoundedFloatText(min=0, max=1, value=0.6),
            frame=ipywidgets.IntText(value=0))
        def show_track(particle, frame, eff_thresh):
            df = self.track_data[key]
            df = df[df["fret", "particle"] == particle]
            fname = df.iloc[0].name[0]
            print(fname)

            # FIXME: This works only for "da" excitation scheme
            fno_d = 2 * frame + 1
            fno_a = 2 * frame + 2

            with pims.open(fname) as fr:
                img_c = self.don_roi(fr[0])
                img_d = fr[fno_d]
                img_a = fr[fno_a]

            fig, (ax_d, ax_a, ax_aa) = plt.subplots(1, 3, figsize=(15, 10))

            for a in (ax_d, ax_a, ax_aa):
                a.axis("off")

            ax_d.set_title("donor")
            ax_a.set_title("acceptor")
            ax_aa.set_title("acceptor (direct)")

            img_dd = self.don_roi(img_d)
            img_da = self.acc_roi(img_d)
            img_aa = self.acc_roi(img_a)

            ax_d.imshow(img_dd, vmin=img_dd.min())
            ax_a.imshow(img_da, vmin=img_da.min())
            ax_aa.imshow(img_aa, vmin=img_aa.min())

            df = self.track_data[key].loc[fname]
            df_d = df[df["acceptor", "frame"] == fno_d-1]
            df_a = df[df["acceptor", "frame"] == fno_a-1]
            mask_d = np.ones(len(df_d), dtype=bool)
            mask_a = np.ones(len(df_a), dtype=bool)
            eff_mask = df_d["fret", "eff"] < eff_thresh
            df_dd = df_d["donor"]
            df_da = df_d["acceptor"]
            df_aa = df_a["acceptor"]

            if particle >= 0:
                mask_d &= df_d["fret", "particle"] == particle
                mask_a &= df_a["fret", "particle"] == particle
            ax_d.scatter(df_dd.loc[mask_d & eff_mask, "x"],
                        df_dd.loc[mask_d & eff_mask, "y"],
                        s=100, facecolor="none", edgecolor="red")
            ax_a.scatter(df_da.loc[mask_d & eff_mask, "x"],
                        df_da.loc[mask_d & eff_mask, "y"],
                        s=100, facecolor="none", edgecolor="red")
            ax_d.scatter(df_dd.loc[mask_d & ~eff_mask, "x"],
                        df_dd.loc[mask_d & ~eff_mask, "y"],
                        s=100, facecolor="none", edgecolor="yellow")
            ax_a.scatter(df_da.loc[mask_d & ~eff_mask, "x"],
                        df_da.loc[mask_d & ~eff_mask, "y"],
                        s=100, facecolor="none", edgecolor="yellow")
            ax_aa.scatter(df_aa.loc[mask_a, "x"], df_aa.loc[mask_a, "y"],
                        s=100, facecolor="none", edgecolor="yellow")

            fig.tight_layout()
            plt.show()

    def show_trajectory(self, key):
        tr = self.track_data[key]
        pnos = tr["fret", "particle"].unique()

        @ipywidgets.interact(particle=ipywidgets.IntText(default=0))
        def show_track(particle):
            p = pnos[particle]
            fig, ax = plt.subplots(figsize=(10, 10))
            t = tr[tr["fret", "particle"] == p]
            c = ax.plot(t["donor", "x"], t["donor", "y"],
                        c=mpl.cm.jet(t["fret", "eff"].mean()), marker=".")
            print(t["fret", "eff"].mean())
            plt.show()
