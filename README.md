# Single molecule FRET analysis

This repository contains a Python package and several Jupyter notebooks to
analyze single molecule FRET data.


## Suggested workflow:

- Copy Jupyter notebooks from the `notebooks` folder to the root (this) folder.
- Copy your data into the `data` folder.
- Open `01. Tracking` notebook. Run each cell and adjust file paths and
  parameters as needed.
- When finished, do the same using the `02. Filter` notebook.
- Summary plots can be created with the `03. Plots` notebook.
- If you want to inspect the tracking data, use the `Inspect` notebook.


## Requirements

- Python >= 3.6
- sdt-python >= 13.0
- numpy
- scipy
- pandas
- matplotlib
- pims
- bokeh >= 0.12.5
