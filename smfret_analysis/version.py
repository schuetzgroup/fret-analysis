import subprocess

import sdt


try:
    git_desc = subprocess.check_output(["git", "describe", "--always"])
    git_desc = git_desc.decode().strip()
except Exception:
    git_desc = "unknown"

__version__ = "2.1"
output_version = 8


def print_info():
    try:
        git_rev = subprocess.check_output(["git", "describe", "--always"])
    except Exception:
        git_rev = "unknown"

    print(f"""smFRET analysis software version {__version__}
(git revision {git_desc})
Output version {output_version}
Using sdt-python version {sdt.__version__}""")
