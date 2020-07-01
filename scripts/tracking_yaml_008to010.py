#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from collections import OrderedDict
from pathlib import Path

from sdt import io, fret, chromatic
import yaml


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert v008 tracking yaml to v010")
    ap.add_argument("file", type=Path, help="YAML file to convert")
    args = ap.parse_args()

    io.yaml.SafeLoader.add_constructor(fret.SmFretTracker.yaml_tag,
                                       io.yaml.odict_constructor)

    with args.file.open() as f:
        data = io.yaml.safe_load(f)

    out = OrderedDict()
    out["excitation_seq"] = data["tracker"].pop("excitation_seq")

    t = fret.SmFretTracker()
    for k, v in data.pop("tracker").items():
        setattr(t, k, v)
    out["tracker"] = t

    lo = data["loc_options"]
    for k, v in lo.items():
        lo[k] = OrderedDict([("algorithm", "3D-DAOSTORM"), ("options", v)])

    data.pop("bead_files")

    out.update(data)

    with open(args.file.name.replace("v008", "v010"), "w") as f:
        io.yaml.safe_dump(out, f)
