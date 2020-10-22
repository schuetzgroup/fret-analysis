#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import re
import shutil
import warnings

import pandas as pd
from sdt import channel_reg, fret, io  # noqa: F401
import tables


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Convert v012 tracking yaml to v013")
    ap.add_argument("--track-prefix", help="Tracking files prefix",
                    default="tracking")
    ap.add_argument("--filtered-prefix", help="Filtered files prefix",
                    default="filtered")
    ap.add_argument("--reg-files", nargs="*",
                    help="Image files for channel registration")
    ap.add_argument("--don-profile-files", nargs="*",
                    help="Image files for determination of donor excitation "
                         "profile")
    ap.add_argument("--acc-profile-files", nargs="*",
                    help="Image files for determination of acceptor "
                         "excitation profile")
    args = ap.parse_args()

    # Convert YAML
    with open(f"{args.track_prefix}-v012.yaml") as f:
        yaml_text = f.read()

    yaml_text = re.sub(r"\!SmFretTracker", r"!SmFRETTracker", yaml_text)
    yaml_text = re.sub(r"chromatic_corr:\s*\!ChromaticCorrector",
                       r"registrator: !Registrator", yaml_text)

    yaml_data = io.yaml.safe_load(yaml_text)
    loc_opts = yaml_data["loc_options"]
    loc_opts["reg_donor"] = loc_opts["reg_acceptor"] = loc_opts.pop("beads")

    new_src = {}
    for k, v in yaml_data["sources"].items():
        new_src[k] = v["files"]
    yaml_data["sources"] = new_src

    spec_src = {}
    if args.reg_files:
        spec_src["registration"] = args.reg_files
    if args.don_profile_files:
        spec_src["donor-profile"] = args.don_profile_files
    if args.acc_profile_files:
        spec_src["acceptor-profile"] = args.acc_profile_files
    yaml_data["special_sources"] = spec_src

    with open(f"{args.track_prefix}-v013.yaml", "w") as f:
        io.yaml.safe_dump(yaml_data, f)

    # Convert tracking data
    tr = yaml_data["tracker"]
    with pd.HDFStore(f"{args.track_prefix}-v012.h5") as src, \
            pd.HDFStore(f"{args.track_prefix}-v013.h5") as dest:
        for k in src.keys():
            df = src[k]
            if k.endswith("trc"):
                tr.flag_excitation_type(df)
                df = df.astype({("fret", "exc_type"): str})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", tables.NaturalNameWarning)
                dest.put(k, df)

    # Copy the rest
    prefix = f"{args.track_prefix}-v{{:03}}"
    for suffix in ".cell_img.npz", ".flat_acceptor.npz", ".flat_donor.npz":
        t_pat = prefix + suffix
        shutil.copy(t_pat.format(12), t_pat.format(13))
    f_pat = f"{args.filtered_prefix}-v{{:03}}.h5"
    shutil.copy(f_pat.format(12), f_pat.format(13))
