#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from collections import OrderedDict

from sdt import io, fret, chromatic


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert v007 tracking yaml to v008")
    ap.add_argument("file", help="YAML file to convert")
    args = ap.parse_args()

    with open(args.file) as f:
        data = io.yaml.safe_load(f)

    files = data.pop("files")
    sources = OrderedDict()
    for key, f in files.items():
        sources[key] = {"files": f, "cells": False}
    data["sources"] = sources
    data["data_dir"] = ""

    with open(args.file.replace("v007", "v008"), "w") as f:
        io.yaml.safe_dump(data, f)
