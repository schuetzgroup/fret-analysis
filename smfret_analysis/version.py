import sdt


__version__ = "1.0.dev"
output_version = 6


def print_info():
    print(f"""Skript version {__version__}
Output version {output_version}
Using sdt-python version {sdt.__version__}""")
