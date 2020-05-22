# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import re
import collections
from pathlib import Path
from typing import (Any, Dict, Iterable, Literal, Optional, Sequence, Tuple,
                    Union)
import warnings

from IPython.display import display
import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims

from sdt import roi, chromatic, io, image, helper
from sdt import flatfield as _flatfield  # avoid name clashes
from sdt.fret import SmFretTracker
from sdt.nbui import Locator

from .version import output_version


@helper.pipeline(ancestor_count=2)
def _img_sum(a: Union[helper.Slicerator, np.ndarray],
             b: Union[helper.Slicerator, np.ndarray]) \
        -> Union[helper.Slicerator, np.ndarray]:
    """:py:func:`helper.pipeline` sum of two images

    Parameters
    ----------
    a, b
        Image data

    Returns
    -------
        Sum image
    """
    return a + b


class Tracker:
    """Jupyter notebook UI for single molecule FRET tracking

    This allows for image registration, single molecule localization,
    tracking, and brightness determination. These are the most time-consuming
    tasks and are thus isolated from the rest of the analysis. Additionally,
    these tasks are the only ones that require access to raw image data, which
    can take up large amounts of disk space and are thus often located on
    external storage. After running the methods of this class, this storage
    can be disconnected and further filtering and analysis can be efficiently
    performed using the :py:class:`Analyzer` class.
    """
    def __init__(self, don_o: Tuple[int], acc_o: Tuple[int],
                 roi_size: Tuple[int], excitation_seq: str = "da",
                 data_dir : Union[str, Path] = ""):
        """Parameters
        ----------
        don_o, acc_o
            Origin (top left corner) of the donor and acceptor channel ROIs
        roi_size
            Width and height of the ROIs
        excitation_seq
            Excitation sequence. Use "d" for excitation, "a" for acceptor
            excitation, and "c" for a image of the cell.
            The given sequence is repeated as necessary. E.g., "da" is
            equivalent to "dadadadada"... Defaults to "da".
        data_dir
            Data directory path. All data paths (e.g. in
            :py:meth:`add_dataset`) are take relative to this. Defaults to "",
            which is the current working directory.
        """
        self.rois = dict(
            donor=roi.ROI(don_o, size=roi_size),
            acceptor=roi.ROI(acc_o, size=roi_size))
        """dict of channel name -> :py:class:`roi.ROI` instances. Contains
        "donor" and "acceptor" keys.
        """
        self.tracker = SmFretTracker(excitation_seq)
        """:py:class:`SmFretTracker` instance used for tracking"""
        self.sources = collections.OrderedDict()
        """dict of dataset name (see :py:meth:`add_dataset`) -> information
        about source (image) files.
        """

        self.loc_data = collections.OrderedDict()
        """dict of dataset name -> single molecule localization data."""
        self.track_data = collections.OrderedDict()
        """dict of dataset name -> single molecule tracking data."""
        self.cell_images = collections.OrderedDict()
        """dict of dataset name -> cell images ("c" in excitation sequence)."""
        self.flatfield = collections.OrderedDict()
        """dict of channel name -> :py:class:`flatfield.Corrector` instances.
        """
        self.data_dir = Path(data_dir)
        """data directory root path"""

        self.locators = collections.OrderedDict([
            ("donor", Locator()), ("acceptor", Locator()),
            ("beads", Locator())])
        """:py:class:`nbui.Locator` instances for locating "beads", "donor"
        emission data, "acceptor" emission data.
        """

    def add_dataset(self, key: str, files_re: str, cells: bool = False):
        """Add a dataset

        A typical experiment consists of several datasets (e.g., the actual
        sample plus controls). This method should be called for each of them
        to add it :py:attr:`sources`.

        Parameters
        ----------
        key
            Name of the dataset
        files_re
            Regular expression describing image file names relative to
            :py:attr:`data_dir`. Use forward slashes as path separators.
        cells
            If True, this is a sample with cells. Any frames corresponding to
            "c" in the excitation sequence will be extracted by
            :py:meth:`extract_cell_images`.
        """
        files = io.get_files(files_re, self.data_dir)[0]
        if not files:
            warnings.warn(f"Empty dataset added: {key}")
        s = collections.OrderedDict(
            [("files", self._open_image_sequences(files)), ("cells", cells)])
        self.sources[key] = s

    def set_bead_loc_opts(self, files_re: str) -> Locator:
        """Display a widget to set options for localization of beads

        for image registration. After finishing, call :py:meth:`make_chromatic`
        to create a :py:class:`chromatic.Corrector` object.

        Parameters
        ----------
        files_re
            Regular expression describing image file names relative to
            :py:attr:`data_dir`. Use forward slashes as path separators.

        Returns
        -------
            Widget
        """
        files = io.get_files(files_re, self.data_dir)[0]
        self.locators["beads"].files = \
            {f: pims.open(str(self.data_dir / f)) for f in files}
        self.locators["beads"].auto_scale()
        return self.locators["beads"]

    def make_chromatic(self, plot: bool = True, max_frame: Optional[int] = None,
                       params: Optional[Dict[str, Any]] = None) \
            -> Optional[mpl.figure.FigureCanvasBase]:
        """Calculate transformation between color channels

        Localize beads using the options set with :py:meth:`set_bead_loc_opts`,
        find pairs and fit transformation. Store result in
        :py:attr:`self.tracker.chromatic_corr`.

        Parameters
        ----------
        plot
            If True, plot the fit results and return the figure canvas.
        max_frame
            Maximum frame number to consider. Useful if beads defocused in
            later frames. If `None` use all frames.
        params
            Passed to :py:meth:`chromatic.Corrector.determine_parameters`.

        Returns
        -------
            If ``plot=True``, return the figure canvas which can be displayed
            in Jupyter notebooks.
        """
        label = ipywidgets.Label(value="Starting…")
        display(label)

        bead_loc = []
        lcr = self.locators["beads"]
        n_files = len(lcr.files)
        for n, (f, i) in enumerate(self.locators["beads"].files.items()):
            label.value = f"Locating beads (file {n+1}/{n_files})"
            bead_loc.append(lcr.batch_func(
                i[:max_frame], **lcr.options))
        label.layout = ipywidgets.Layout(display="none")

        acc_beads = [self.rois["acceptor"](l) for l in bead_loc]
        don_beads = [self.rois["donor"](l) for l in bead_loc]
        cc = chromatic.Corrector(don_beads, acc_beads)
        cc.determine_parameters(**params or {})
        self.tracker.chromatic_corr = cc

        if plot:
            fig, ax = plt.subplots(1, 2)
            cc.test(ax=ax)
            return fig.canvas

    def _open_image_sequences(self, files: Iterable[Union[str, Path]]) \
            -> collections.OrderedDict:
        """Use :py:func:`pims.open` to open image sequences

        Parameters
        ----------
        files
            Files to open

        Returns
        -------
            Map of file name -> pims FrameSequence
        """
        ret = collections.OrderedDict()
        for f in files:
            pth = self.data_dir / f
            if pth.suffix.lower() == ".spe":
                # Disable warnings about file size being wrong which is caused
                # by SDT-control not dumping the whole file
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    ret[f] = pims.open(str(pth))
            else:
                ret[f] = pims.open(str(pth))
        return ret

    def donor_sum(self, fr: Union[helper.Slicerator, np.ndarray]) \
            -> Union[helper.Slicerator, np.ndarray]:
        """Get sum image (sequence) of both channels upon donor excitation

        Transform donor channel image(s) and add acceptor channel image(s).

        Parameters
        ----------
        fr
            Image (sequence)

        Returns
        -------
            Sum of transformed donor image(s) and acceptor image(s)
        """
        fr = self.tracker.frame_selector(fr, "d")
        fr_d = self.tracker.chromatic_corr(self.rois["donor"](fr), channel=1,
                                           cval=np.mean)
        fr_a = self.rois["acceptor"](fr)
        return _img_sum(fr_d, fr_a)

    def set_loc_opts(self, exc_type: Literal["donor", "acceptor"]) \
            -> ipywidgets.Widget:
        """Return a widget for setting localization options

        Parameters
        ----------
        exc_type
            Whether to set the localization options for the sum of donor
            and acceptor emission upon donor excitation ("donor") or
            for acceptor emission upon acceptor excitation ("acceptor").

        Returns
        -------
            Interactive selection of localization options with visual feedback
        """
        d_sel = ipywidgets.Dropdown(
            options=list(self.sources.keys()), description="dataset")

        if exc_type.startswith("d"):
            loc = self.locators["donor"]
        elif exc_type.startswith("a"):
            loc = self.locators["acceptor"]
        else:
            raise ValueError("`exc_type` has to be \"donor\" or \"acceptor\".")

        def set_files(change=None):
            if exc_type.startswith("d"):
                loc.files = \
                    {k: self.donor_sum(v)
                     for k, v in self.sources[d_sel.value]["files"].items()}
            elif exc_type.startswith("a"):
                loc.files = \
                    {k: self.rois["acceptor"](
                         self.tracker.frame_selector(v, "a"))
                     for k, v in self.sources[d_sel.value]["files"].items()}

        set_files()
        loc.auto_scale()
        d_sel.observe(set_files, "value")

        return ipywidgets.VBox([d_sel, loc])

    def locate(self):
        """Locate single-molecule signals

        For this, use the settings selected with help of the widgets
        returned by :py:meth:`set_loc_opts` for donor and acceptor excitation
        or loaded with :py:meth:`load`.

        Localization data for each dataset is collected in the
        :py:attr:`loc_data` dictionary.
        """
        num_files = sum(len(s["files"]) for s in self.sources.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        for key, src in self.sources.items():
            ret = []
            files = src["files"]
            for i, (f, fr) in enumerate(files.items()):
                label.value = f"Locating {f} ({cnt}/{num_files})"
                cnt += 1

                don_fr = self.donor_sum(fr)
                lo = self.locators["donor"].batch_func(
                    don_fr, **self.locators["donor"].options)

                acc_fr = self.rois["acceptor"](
                    self.tracker.frame_selector(fr, "a"))
                if len(acc_fr):
                    lo_a = self.locators["acceptor"].batch_func(
                        acc_fr, **self.locators["acceptor"].options)
                    lo = pd.concat([lo, lo_a]).sort_values("frame")
                    lo = lo.reset_index(drop=True)

                # correct for the fact that locating happend in the
                # acceptor ROI
                self.rois["acceptor"].reset_origin(lo)
                ret.append(lo)
            self.loc_data[key] = pd.concat(ret, keys=files)

    def track(self, feat_radius: int = 4, bg_frame: int = 3,
              link_radius: float = 1.0, link_mem: int = 1, min_length: int = 4,
              bg_estimator: Union[Literal["mean", "median"], np.ufunc] = "mean",
              image_filter: Optional[helper.Pipeline] =
                  lambda i: image.gaussian_filter(i, 1),
              neighbor_radius: Optional[float] = None):
        """Link single-molecule localizations into trajectories

        and measure signal intensities.
        This is to be called after :py:meth:`locate`. Results are collected in
        the :py:attr:`track_data` dictionary.

        Parameters
        ----------
        feat_radius
            For intensity measurement, sum all pixel values in a circle with
            this radius around each signal's position; see
            :py:func:`sdt.brightness.from_raw_image`. Defaults to 4.
        bg_frame
            Determine local background from pixels in a ring of this width
            around each signal circle; see
            :py:func:`sdt.brightness.from_raw_image`. Defaults to 3.
        link_radius
            Maximum distance a particle is expected to move between frames.
            See :py:func:`trackpy.link`. Defaults to 1.
        link_mem
            Maximum number of consecutive frames a particle is can be missed.
            See :py:func:`trackpy.link`. It is recommended to set this >= 1
            as this allows for tracking particles even if one of the
            FRET pair fluorophores is bleached. Defaults to 1.
        min_length
            Remove any tracks with fewer localizations. Defaults to 4.
        bg_estimator
            How to determine the background from the pixels in the background
            ring; see :py:func:`sdt.brightness.from_raw_image`. Defaults to
            "mean".
        image_filter
            Apply this filter before intensity determination. Defaults to
            ``lambda i: sdt.image.gaussian_filter(i, 1)``, i.e., a slight
            Gaussian blur which has been found to decrease noise in low-SNR
            scenarios while hardly affecting the intensity.
        neighbor_radius
            Any particles that are closer than ``neighbor_radius`` are
            considered overlapping and may be filtered later. If `None`,
            use ``2 * feat_radius + 1``. Defaults to `None`.
        """
        num_files = sum(len(s["files"]) for s in self.sources.values())
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        self.tracker.link_options.update({"search_range": link_radius,
                                          "memory": link_mem})
        self.tracker.brightness_options.update({"radius": feat_radius,
                                                "bg_frame": bg_frame,
                                                "bg_estimator": bg_estimator})
        self.tracker.min_length = min_length
        if neighbor_radius is not None:
            self.tracker.neighbor_radius = neighbor_radius
        else:
            self.tracker.neighbor_radius = 2 * feat_radius + 1

        for key, src in self.sources.items():
            ret = []
            ret_keys = []
            new_p = 0  # Particle ID unique across files
            for f, img in src["files"].items():
                label.value = f"Tracking {f} ({cnt}/{num_files})"
                cnt += 1

                loc = self.loc_data[key]
                try:
                    loc = loc.loc[f].copy()
                except KeyError:
                    # No localizations in this file
                    continue

                don_loc = self.rois["donor"](loc)
                acc_loc = self.rois["acceptor"](loc)

                if image_filter is not None:
                    img = image_filter(img)

                try:
                    d = self.tracker.track(
                            self.rois["donor"](img),
                            self.rois["acceptor"](img),
                            don_loc, acc_loc)
                except Exception as e:
                    warnings.warn(f"Tracking failed for {f}. Reason: {e}")

                    fake_fret_df = pd.DataFrame(
                        columns=["particle", "interp", "has_neighbor"],
                        dtype=int)
                    d = pd.concat([don_loc.iloc[:0], acc_loc.iloc[:0],
                                   fake_fret_df],
                                  keys=["donor", "acceptor", "fret"], axis=1)

                ps = d["fret", "particle"].copy().values
                for p in np.unique(ps):
                    d.loc[ps == p, ("fret", "particle")] = new_p
                    new_p += 1
                ret.append(d)
                ret_keys.append(f)

            self.track_data[key] = pd.concat(ret, keys=ret_keys)

    def extract_cell_images(self, key: str = "c"):
        """Get cell images for thresholding

        This extracts the images of cell contours for use with
        :py:class:`Analyzer`. Images are copied to :py:attr:`cell_images`.
        This is only applied to datasets where ``cells=True`` was set when
        adding them using :py:meth:`add_dataset`.

        Parameters
        ----------
        key
            Character describing cell images in the excitation sequence
            ``excitation_seq`` parameter to :py:meth:`__init__`. Defaults to
            "c".
        """
        for k, v in self.sources.items():
            if not v["cells"]:
                # no cells
                continue
            for f, fr in v["files"].items():
                don_fr = self.rois["donor"](fr)
                cell_fr = np.array(self.tracker.frame_selector(don_fr, key))
                self.cell_images[f] = cell_fr

    def make_flatfield(self, dest: Literal["donor", "acceptor"], files_re: str,
                       src: Optional[Literal["donor", "acceptor"]] = None,
                       frame: int = 0, bg: Union[float, np.ndarray] = 200,
                       smooth_sigma: float = 3., gaussian_fit: bool = False):
        """Calculate flatfield correction from separate bulk images

        If images the excitation laser profile were recorded using
        homogeneously labeled surfaces, this method can be used to calculate
        the flatfield corrections for donor and acceptor excitation.
        Results are saved to :py:attr:`flatfield`.

        Parameters
        ----------
        dest
            Whether to calculate the flatfield correction for the donor
            (``"donor"``) or acceptor (``"acceptor"``) excitation.
        files_re
            Regular expression describing image file names relative to
            :py:attr:`data_dir`. Use forward slashes as path separators.
        src
            Whether to use the donor or acceptor emission channel. If `None`,
            use the same as `dest`. Defaults to `None`.
        frame
            Which frame in the image sequences described by `files_re` to use.
            Defaults to `None`.
        bg
            Camera background. Either a value or an image data array.
            Defaults to 200.
        smooth_sigma
            Sigma for Gaussian blur to smoothe images. Defaults to 3.
        gaussian_fit
            If `True`, perform a Gaussian fit to the excitation profiles.
            Otherwise, use the mean of the `frame`-th images from the files
            described by `files_re`. Defaults to `False`.
        """
        files = io.get_files(files_re, self.data_dir)[0]
        imgs = []
        if src is None:
            src = dest

        for f in files:
            with pims.open(str(self.data_dir / f)) as fr:
                imgs.append(self.rois[src](fr[frame].astype(float)) - bg)

        if src == "acceptor" and dest == "donor":
            imgs = [self.tracker.chromatic_corr(i, channel=2) for i in imgs]
        elif src != dest:
            raise ValueError ("src != dest and (src != \"acceptor\" and dest "
                              "!= \"donor\")")

        self.flatfield[dest] = _flatfield.Corrector(
            imgs, smooth_sigma=smooth_sigma, gaussian_fit=gaussian_fit)

    def make_flatfield_sm(self, dest: Literal["donor", "acceptor"],
                          keys: Union[Sequence[str],
                                      Literal["all", "no-cells"]] = "all",
                          weighted: bool = False, frame: Optional[int] = None):
        """Calculate flatfield correction from single-molecule data

        If images the excitation laser profile were NOT recorded using
        homogeneously labeled surfaces, this method can be used to calculate
        the flatfield corrections for donor and acceptor excitation from
        single-molecule data. To this end, a 2D Gaussian is fit to the
        single-molecule intensities. Results are saved to :py:attr:`flatfield`.

        Parameters
        ----------
        dest
            Whether to calculate the flatfield correction for the donor or
            acceptor excitation.
        keys
            Names of datasets to use for calculation. "all" will use all
            datasets. "no-cells" will use only datasets where ``cells=False``
            was set when adding them via :py:meth:`add_dataset`. Defaults to
            "all".
        weighted
            Weigh data inversely to density when performing a Gaussian fit.
            Not very robust, thus the default is `False`.
        frame
            Select only data from this frame. If `None`, use the first frame
            corresponding to `dest`. Defaults to `None`.
        """
        if not len(keys):
            return
        if keys == "all":
            keys = self.track_data.keys()
        elif keys == "no-cells":
            keys = [k for k in self.track_data.keys()
                    if self.sources[k]["cells"]]

        if frame is None:
            frame = self.tracker.excitation_frames[dest[0]][0]

        data = []
        for k in keys:
            d = self.track_data[k]
            d = d[(d[dest, "frame"] == frame) &
                  (d["fret", "interp"] == 0) &
                  (d["fret", "has_neighbor"] == 0)]
            if dest.startswith("d"):
                mass = d["donor", "mass"] + d["acceptor", "mass"]
            else:
                mass = d[dest, "mass"]
            d = d[[(dest, "x"), (dest, "y")]].copy()
            d.columns = ["x", "y"]
            d["mass"] = mass
            data.append(d)

        r = self.rois[dest]
        img_shape = (r.bottom_right[1] - r.top_left[1],
                     r.bottom_right[0] - r.top_left[0])
        self.flatfield[dest] = _flatfield.Corrector(*data, shape=img_shape,
                                                    density_weight=weighted)

    def save(self, file_prefix: str = "tracking"):
        """Save results to disk

        This will save localization settings and data, tracking data, channel
        overlay transforms, cell images, and flatfield corrections to disk.

        Parameters
        ----------
        file_prefix
            Common prefix for all files written by this method. It will be
            suffixed by the output format version (v{output_version}) and
            file extensions corresponding to what is saved in each file.
            Defaults to "tracking".
        """
        loc_options = collections.OrderedDict(
            [(k, v.get_settings()) for k, v in self.locators.items()])

        src = collections.OrderedDict()
        for k, v in self.sources.items():
            v_new = v.copy()
            v_new["files"] = list(v["files"].keys())
            src[k] = v_new

        top = collections.OrderedDict(
            tracker=self.tracker, rois=self.rois, loc_options=loc_options,
            data_dir=str(self.data_dir), sources=src)
        outfile = Path(f"{file_prefix}-v{output_version:03}")
        with outfile.with_suffix(".yaml").open("w") as f:
            io.yaml.safe_dump(top, f)

        with warnings.catch_warnings():
            import tables
            warnings.simplefilter("ignore", tables.NaturalNameWarning)

            with pd.HDFStore(outfile.with_suffix(".h5")) as s:
                for key, loc in self.loc_data.items():
                    s["{}_loc".format(key)] = loc
                for key, trc in self.track_data.items():
                    s["{}_trc".format(key)] = trc

        np.savez_compressed(outfile.with_suffix(".cell_img.npz"),
                            **self.cell_images)
        for k, ff in self.flatfield.items():
            ff.save(outfile.with_suffix(f".flat_{k}.npz"))

    @staticmethod
    def load_data(file_prefix: str = "tracking", loc: bool = True,
                  tracks: bool = True, cell_images: bool = True,
                  flatfield: bool = True) -> Dict:
        """Load data to a dictionary

        Parameters
        ----------
        file_prefix
            Prefix used for saving via :py:meth:`save`. Defaults to "tracking".
        loc
            Whether to load localization data. Defaults to `True`.
        tracks
            Whether to load tracking data. Defaults to `True`.
        cell_images
            Whether to load cell images. Defaults to `True`.
        flatfield
            Whether to load flatfield corrections. Defaults to `True`.

        Returns
        -------
            Dictionary of loaded data and settings.
        """
        infile = Path(f"{file_prefix}-v{output_version:03}")
        with infile.with_suffix(".yaml").open() as f:
            cfg = io.yaml.safe_load(f)

        ret = {"rois": cfg["rois"], "data_dir": Path(cfg.get("data_dir", "")),
               "sources": cfg["sources"], "loc_options": cfg["loc_options"],
               "tracker": cfg["tracker"],
               "loc_data": collections.OrderedDict(),
               "track_data": collections.OrderedDict(),
               "cell_images": collections.OrderedDict(),
               "flatfield": collections.OrderedDict()}

        do_load = []
        if loc:
            do_load.append((ret["loc_data"], "_loc"))
        if tracks:
            do_load.append((ret["track_data"], "_trc"))
        if len(do_load):
            with pd.HDFStore(infile.with_suffix(".h5"), "r") as s:
                for sink, suffix in do_load:
                    keys = (k for k in s.keys() if k.endswith(suffix))
                    for k in keys:
                        new_key = k[1:-len(suffix)]
                        sink[new_key] = s[k]

        if cell_images:
            cell_img_file = infile.with_suffix(".cell_img.npz")
            try:
                with np.load(cell_img_file) as data:
                    ret["cell_images"] = collections.OrderedDict(data)
            except Exception:
                warnings.warn("Could not load cell images from file "
                              f"\"{str(cell_img_file)}\".")
        if flatfield:
            flatfield_glob = str(infile.with_suffix(".flat_*.npz"))
            key_re = re.compile(r"^\.flat_([\w\s-]+)")
            for p in Path().glob(flatfield_glob):
                m = key_re.match(p.suffixes[-2])
                if m is None:
                    warnings.warn("Could not load flatfield corrector from "
                                  f"{str(p)}.")
                else:
                    ret["flatfield"][m.group(1)] = _flatfield.Corrector.load(p)

        return ret

    @classmethod
    def load(cls, file_prefix: str = "tracking", loc: bool = True,
             tracks: bool = True, cell_images: bool = True,
             flatfield: bool = True) -> "Tracker":
        """Construct class instance from saved data

        Raw data needs to be accessible for this.

        Parameters
        ----------
        file_prefix
            Prefix used for saving via :py:meth:`save`. Defaults to "tracking".
        loc
            Whether to load localization data. Defaults to `True`.
        tracks
            Whether to load tracking data. Defaults to `True`.
        cell_images
            Whether to load cell images. Defaults to `True`.
        flatfield
            Whether to load flatfield corrections. Defaults to `True`.

        Returns
        -------
            Attributes reflect saved settings and data.
        """
        cfg = cls.load_data(file_prefix, loc, tracks, cell_images, flatfield)

        ret = cls([0, 0], [0, 0], [0, 0], "")

        for key in ("rois", "data_dir", "tracker", "loc_data", "track_data",
                    "cell_images", "flatfield"):
            setattr(ret, key, cfg[key])

        src = cfg["sources"]
        for k, s in src.items():
            src[k]["files"] = ret._open_image_sequences(s["files"])
        ret.sources = src

        for n, lo in cfg["loc_options"].items():
            ret.locators[n].set_settings(lo)

        return ret
