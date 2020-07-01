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
        description="Convert v011 tracking yaml to v012")
    ap.add_argument("file", type=Path, help="YAML file to convert")
    args = ap.parse_args()

    with args.file.open() as f:
        data = io.yaml.safe_load(f)

    for v in data["sources"].values():
        c = v.pop("cells")
        v["special"] = "cells" if c else "none"

    with open(args.file.name.replace("v011", "v012"), "w") as f:
        io.yaml.safe_dump(data, f)
