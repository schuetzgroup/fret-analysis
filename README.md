<!--
SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>

SPDX-License-Identifier: CC-BY-4.0
-->

# Single molecule FRET analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4604567.svg)](https://doi.org/10.5281/zenodo.4604567)

This repository contains a Python package and several Jupyter notebooks to analyze single molecule FRET data.
The procedure is discribed in our article published in the [Journal of Visualized Experiments](https://doi.org/10.3791/63124).
The [supplemental information](https://www.jove.com/files/ftp_upload/63124/si.pdf) contains screen shots with step-by-step instructions.
Note that over time, some function names may have changed slightly, but with the current notebooks from this repository it should still be easily possible to follow the manual.

If you use this software in a project resulting in a scientific publication, please [cite](https://doi.org/10.5281/zenodo.4604567) the software.


## Suggested workflow:

- Clone repository
- Copy Jupyter notebooks from the `notebooks` folder to the root (this) folder.
- Copy your data into the `data` folder.
- Open `01. Tracking` notebook. Run each cell and adjust file paths and
  parameters as needed.
- When finished, do the same using the `02. Filter` notebook.
- Summary plots can be created with the `03. Plots` notebook.


## Requirements

- Python >= 3.10
- Jupyter Notebook or Lab
- sdt-python >= 17.0
- numpy
- scipy
- pandas
- matplotlib
- pims
- ipympl
- OpenCV
