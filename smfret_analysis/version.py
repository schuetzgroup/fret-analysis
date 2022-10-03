# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

__version__ = "4.0.dev"


def print_info():
    import sdt

    print(f"""smFRET analysis software version {__version__}
Using sdt-python version {sdt.__version__}""")
