#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from pathlib import Path

import numpy as np

from sdt import flatfield


def load_old(file):
    """Load old format save file from disk

    Parameters
    ----------
    file : str or file-like
        Where to load from

    Returns
    -------
    sdt.flatfield.Corrector
        Loaded instance
    """
    with np.load(file, allow_pickle=True) as ld:
        ret = flatfield.Corrector([ld["avg_img"]], gaussian_fit=False)
        ret.corr_img = ld["corr_img"]
        ret.fit_result = np.asscalar(ld["fit_result"])
        bg = ld["bg"]
        ret.bg = bg if bg.size > 1 else np.asscalar(bg)
        ret._make_interp()
    return ret


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert sdt.flatfield.Corrector savefiles created with "
                    "sdt-python before commit "
                    "e57a3cefa3ca3fb2aea0758f30ca1d265966349b.")
    ap.add_argument("file", help="Save file to convert")
    args = ap.parse_args()

    cc = load_old(args.file)
    Path(args.file).rename(args.file + ".old")
    cc.save(args.file)