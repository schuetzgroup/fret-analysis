import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets
import pims

from sdt import fret, helper

from .tracker import Tracker


class Inspector:
    def __init__(self, file_prefix="tracking", data_dir=""):
        tr = Tracker.load(file_prefix, loc=False, data_dir=data_dir)
        self.track_data = tr.track_data
        self.exc_scheme = "".join(tr.tracker.excitation_seq)
        self.rois = tr.rois
        self.data_dir = tr.data_dir

    def mark_track(self, key):
        @ipywidgets.interact(
            particle=ipywidgets.IntText(value=0),
            eff_thresh=ipywidgets.BoundedFloatText(min=0, max=1, value=0.6),
            frame=ipywidgets.IntText(value=0))
        def show_track(particle, frame, eff_thresh):
            df = self.track_data[key]
            df = df[df["fret", "particle"] == particle]
            fname = df.iloc[0].name[0]
            print(fname)

            exc = np.array(list(self.exc_scheme))
            d_frames = np.nonzero(exc == "d")[0]
            a_frames = np.nonzero(exc == "a")[0]

            reps = frame // len(d_frames)
            res = frame - reps * len(d_frames)
            fno_d = reps * len(exc) + d_frames[res]
            fno_a = (reps * len(exc) +
                     a_frames[np.nonzero(a_frames > res)[0][0]])

            with pims.open(str(self.data_dir / fname)) as fr:
                img_d = fr[fno_d]
                img_a = fr[fno_a]

            fig, (ax_d, ax_a, ax_aa) = plt.subplots(1, 3, figsize=(15, 10))

            for a in (ax_d, ax_a, ax_aa):
                a.axis("off")

            ax_d.set_title("donor")
            ax_a.set_title("acceptor")
            ax_aa.set_title("acceptor (direct)")

            img_dd = self.rois["donor"](img_d)
            img_da = self.rois["acceptor"](img_d)
            img_aa = self.rois["acceptor"](img_a)

            ax_d.imshow(img_dd, vmin=img_dd.min())
            ax_a.imshow(img_da, vmin=img_da.min())
            ax_aa.imshow(img_aa, vmin=img_aa.min())

            df_d = df[df["acceptor", "frame"] == fno_d]
            df_a = df[df["acceptor", "frame"] == fno_a]
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

    def show_track(self, key):
        tr = self.track_data[key]
        pnos = tr["fret", "particle"].unique()

        @ipywidgets.interact(particle=ipywidgets.IntText(value=0))
        def show_track(particle):
            p = pnos[particle]
            fig, ax = plt.subplots(figsize=(10, 10))
            t = tr[tr["fret", "particle"] == p]
            c = ax.plot(t["donor", "x"], t["donor", "y"],
                        c=mpl.cm.jet(t["fret", "eff"].mean()), marker=".")
            print(t["fret", "eff"].mean())
            plt.show()

    def raw_features(self, key, particle, figsize=None, colums=8,
                     img_size=3):
        t = self.track_data[key]
        t0 = t[t["fret", "particle"] == particle]
        fname = t0.index[0][0]

        fig = plt.figure(figsize=figsize)

        with pims.open(str(self.data_dir / fname)) as img:
            don_img = self.rois["donor"](img)
            acc_img = self.rois["acceptor"](img)

            fret.draw_track(t0, particle, don_img, acc_img, img_size,
                            columns=colums, figure=fig)

    def show_all_tracks(self, key):
        dat = self.track_data[key]
        files = dat.index.levels[0].unique()

        @ipywidgets.interact(file=ipywidgets.Dropdown(options=files))
        def plot(file):
            fig, ax = plt.subplots(figsize=(8, 8))

            d = dat.loc[file]
            for p, trc in helper.split_dataframe(
                    d, ("fret", "particle"),
                    [("donor", "x"), ("donor", "y")]):
                ax.plot(trc[:, 0], trc[:, 1])
                ax.text(*trc[0], str(p))
            plt.show()

    def plot_tracks(self, key, particles, axes):
        data = self.track_data[key]
        fig, ax = plt.subplots(1, len(axes), figsize=(10, 5))

        info = []
        for p in particles:
            d = data[(data["fret", "particle"] == p) &
                     (data["fret", "exc_type"] == 0)]
            for a, (x, y) in zip(ax, axes):
                a.plot(d[x], d[y], ".-")
                a.set_xlabel(" ".join(x))
                a.set_ylabel(" ".join(y))
            info.append("particle {}: start {}, end {}".format(
                p, d["donor", "frame"].min(), d["donor", "frame"].max()))

        for a in ax.flatten():
            a.grid()

        fig.tight_layout()
        plt.show()
        print("\n".join(info))
