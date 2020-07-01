<!--
SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>

SPDX-License-Identifier: CC-BY-4.0
-->

# Single molecule FRET analysis

This repository contains a Python package and several Jupyter notebooks to
analyze single molecule FRET data.


## Suggested workflow:

- Clone repository
- Copy Jupyter notebooks from the `notebooks` folder to the root (this) folder.
- Copy your data into the `data` folder.
- Open `01. Tracking` notebook. Run each cell and adjust file paths and
  parameters as needed.
- When finished, do the same using the `02. Filter` notebook.
- Summary plots can be created with the `03. Plots` notebook.
- If you want to inspect the tracking data, use the `Inspect` notebook.


## Requirements

- Python >= 3.6
- Jupyter Notebook or Lab
- sdt-python >= 14.4
- numpy
- scipy
- pandas
- matplotlib
- pims
- ipympl
- OpenCV
