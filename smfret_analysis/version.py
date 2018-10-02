import subprocess

import sdt


__version__ = "2.1"
output_version = 9


def print_info():
    try:
        git_rev = subprocess.check_output(["git", "describe", "--always"])
        git_rev = git_rev.decode().strip()
    except Exception:
        git_rev = "unknown"

    print(f"""smFRET analysis software version {__version__}
(git revision {git_rev})
Output version {output_version}
Using sdt-python version {sdt.__version__}""")
