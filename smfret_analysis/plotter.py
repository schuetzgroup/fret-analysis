import math
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sdt import fret

from .version import output_version


class Plotter:
    def __init__(self, file_prefix="filtered", data_dir=""):
        self.track_data = OrderedDict()
        self.data_dir = Path(data_dir)
        infile = Path(f"{file_prefix}-v{output_version:03}.h5")
        with pd.HDFStore(infile, "r") as s:
            for k in s.keys():
                if not k.endswith("_trc"):
                    continue
                key = k.lstrip("/")[:-4]
                self.track_data[key] = s[k]

    def scatter(self, *args, **kwargs):
        return fret.smfret_scatter(self.track_data, *args, **kwargs)

    def hist(self, *args, **kwargs):
        return fret.smfret_hist(self.track_data, *args, **kwargs)

