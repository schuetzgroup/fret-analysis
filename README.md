<!--
SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>

SPDX-License-Identifier: CC-BY-4.0
-->

# Single-molecule FRET analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4604566.svg)](https://doi.org/10.5281/zenodo.4604566)

This repository contains a Python package and several Jupyter notebooks to analyze single molecule FRET data.
The procedure is discribed in our article published in the [Journal of Visualized Experiments](https://doi.org/10.3791/63124).
The [supplemental information](https://www.jove.com/files/ftp_upload/63124/si.pdf) contains screen shots with step-by-step instructions.
Note that over time, some function names may have changed slightly, but with the current notebooks from this repository it should still be easily possible to follow the manual.

If you use this software in a project resulting in a scientific publication, please [cite](https://doi.org/10.5281/zenodo.4604566) the software.


## Suggested workflow:

- Install the [uv](https://docs.astral.sh/uv/) Python package manager.
  Linux users can use their distribution's package management system to install `uv`.
- Create a new folder.
- Download [Jupyter notebooks for the analysis](https://github.com/schuetzgroup/fret-analysis/tree/master/notebooks) into the folder.
- Navigate into the folder it using a command line prompt.
- Initialize `uv` in this folder by executing

  ```
  uv init --bare
  ```

- Install the FRET analysis python package, either from PyPI,

  ```
  uv add fret-analysis
  ```

  or from Github,

  ```
  uv add git+https://github.com/schuetzgroup/fret-analysis.git
  ```

- Start Jupyter Lab,
  ```
  uv run --with jupyter jupyter lab
  ```

- Open `01. Tracking` notebook. Run each cell and adjust file paths and parameters as needed.
- When finished, do the same using the `02. Filter` notebook.
- Summary plots can be created with the `03. Plots` notebook.
