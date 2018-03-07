import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sdt.plot import density_scatter

from .version import output_version


class Plotter:
    def __init__(self, file_prefix="filtered"):
        self.track_data = {}
        with pd.HDFStore(f"{file_prefix}-v{output_version:03}.h5", "r") as s:
            for k in s.keys():
                if not k.endswith("_trc"):
                    continue
                key = k.lstrip("/")[:-4]
                self.track_data[key] = s[k]

    def scatter(self, xdata=("fret", "eff"), ydata=("fret", "stoi"),
                frame=None, columns=2, size=5):
        rows = math.ceil(len(self.track_data) / columns)
        fig, ax = plt.subplots(rows, columns, figsize=(columns*size,
                                                       rows*size),
                               sharex=True, sharey=True)

        for (k, f), a in zip(self.track_data.items(), ax.T.flatten()):
            if frame is not None:
                f = f[f["donor", "frame"] == frame]
            x = f[xdata].values.astype(float)
            y = f[ydata].values.astype(float)
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]
            try:
                density_scatter(x, y, ax=a)
            except Exception:
                a.scatter(x, y)
            a.set_title(k)

        for a in ax.T.flatten()[len(self.track_data):]:
            a.axis("off")

        for a in ax.flatten():
            a.set_xlabel(" ".join(xdata))
            a.set_ylabel(" ".join(ydata))
            a.grid()

    def hist(self, data=("fret", "eff"), frame=None, columns=2, size=5):
        rows = math.ceil(len(self.track_data) / columns)
        fig, ax = plt.subplots(rows, columns, figsize=(columns*size,
                                                       rows*size),
                               sharex=True, sharey=True)
        b = np.linspace(-0.5, 1.5, 50)
        for (k, f), a in zip(self.track_data.items(), ax.T.flatten()):
            if frame is not None:
                f = f[f["donor", "frame"] == frame]
            x = f[data].values.astype(float)
            m = np.isfinite(x)
            x = x[m]

            a.hist(x, b, density=False)
            a.set_title(k)

        for a in ax.T.flatten()[len(self.track_data):]:
            a.axis("off")

        for a in ax.flatten():
            a.set_xlabel("FRET eff")
            a.set_ylabel("# events")
            a.grid()

