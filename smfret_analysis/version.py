# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import subprocess

import sdt

__version__ = "3.1.0"


def print_info():
    try:
        # capture output so that git errors are not printed in notebooks
        git_rev = subprocess.check_output(
            ["git", "describe", "--always"], capture_output=True
        )
        git_rev = git_rev.decode().strip()
    except Exception:
        git_rev = "unknown"

    print(f"""smFRET analysis software version {__version__}
(git revision {git_rev})
Using sdt-python version {sdt.__version__}""")
