import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sdt import fret

from .version import output_version
from .analyzer import Analyzer


class Plotter:
    def __init__(self, file_prefix="filtered"):
        self.track_data = Analyzer.load_data(file_prefix)

    def scatter(self, *args, **kwargs):
        return fret.smfret_scatter(self.track_data, *args, **kwargs)

    def hist(self, *args, **kwargs):
        return fret.smfret_hist(self.track_data, *args, **kwargs)

