# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provide :py:class:`Tracker` as a Jupyter notebook UI for smFRET tracking"""
import collections
import contextlib
import itertools
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union
import warnings

from IPython.display import display
import ipywidgets
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims

from sdt import fret, helper, io, image, multicolor, nbui, roi
from sdt import flatfield as _flatfield  # avoid name clashes

from .data_store import DataStore


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


class Locator(ipywidgets.VBox):
    def __init__(self, tracker, locator):
        self._tracker = tracker
        self._locator = locator
        self._files = []

        self._file_selector = ipywidgets.Dropdown(description="file")
        self._frame_selector = ipywidgets.BoundedIntText(
            description="frame", min=0, max=0)
        self._selector_box = ipywidgets.HBox([self._file_selector,
                                              self._frame_selector])

        super().__init__([self._selector_box, self._locator])

        self._opened = []

        self._file_selector.observe(self._file_changed, "value")
        self._frame_selector.observe(self._frame_changed, "value")

    def _make_image_sequence(self, don, acc):
        raise NotImplementedError("_make_image_sequence() needs to be "
                                  "implemented by sub-class.")

    def _set_files(self, files):
        self._files = files
        opt_list = []
        for k, f in files.items():
            if isinstance(f, dict):
                opt_list.append((f"{f['donor']}, {f['acceptor']}", k))
            else:
                opt_list.append((f, k))
        self._file_selector.options = opt_list
        if opt_list and self._file_selector.index is None:
            self._file_selector.index = 0

    def _file_changed(self, change=None):
        for f in self._opened:
            f.close()
        self._opened = []

        cf = self._files[self._file_selector.index]
        im_seq, self._opened = self._tracker._open_image_sequence(cf)

        self._cur_seq = self._make_image_sequence(im_seq["donor"],
                                                  im_seq["acceptor"])
        self._frame_selector.max = len(self._cur_seq)

        self._frame_changed()

    def _frame_changed(self, change=None):
        self._locator.input = self._cur_seq[self._frame_selector.value]


class RegistrationLocator(Locator):
    def __init__(self, tracker, channel):
        super().__init__(tracker, tracker.locators[f"reg_{channel}"])
        self._channel = channel
        self._set_files(tracker.special_sources["registration"])
        self._locator.image_display.auto_scale()

    def _make_image_sequence(self, don, acc):
        if self._channel == "donor":
            return don
        return acc


class FRETLocator(Locator):
    def __init__(self, tracker, channel):
        super().__init__(tracker, tracker.locators[channel])

        self._dataset_selector = ipywidgets.Dropdown(description="dataset")
        self._selector_box.children = [self._dataset_selector,
                                       *self._selector_box.children]

        self._dataset_selector.observe(self._dataset_changed, "value")

        self._dataset_selector.options = list(tracker.sources)
        if tracker.sources and self._dataset_selector.index is None:
            self._dataset_selector.index = 0
        self._locator.image_display.auto_scale()

    def _dataset_changed(self, change=None):
        src = self._tracker.sources[self._dataset_selector.value]
        self._set_files(src)


class DonorLocator(FRETLocator):
    def __init__(self, tracker):
        super().__init__(tracker, "donor")

    def _make_image_sequence(self, don, acc):
        return self._tracker.donor_sum(don, acc)


class AcceptorLocator(FRETLocator):
    def __init__(self, tracker):
        super().__init__(tracker, "acceptor")

    def _make_image_sequence(self, don, acc):
        return self._tracker.tracker.frame_selector.select(acc, "a")


class _ChannelSplitter(ipywidgets.VBox):
    def __init__(self, tracker):
        self.image_selector = nbui.ImageSelector()

        # The following fails due to output and input being arrays and `link`
        # trying to compary using a simple !=
        # traitlets.link((self.image_selector, "output"),
        #                (tracker.channel_splitter, "input"))
        # Therefore, use `observe`
        self._tracker = tracker
        self.image_selector.observe(self._image_selected, "output")

        super().__init__([self.image_selector, tracker.channel_splitter])

    def _image_selected(self, change=None):
        self._tracker.channel_splitter.input = self.image_selector.output


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
    tracker: fret.SmFRETAnalyzer
    """Provides tracking functionality"""
    data_dir: Path
    """All paths to source image files are relative to this"""
    rois: Dict[str, Union[roi.ROI, None]]
    """Map of channel name -> :py:class:`roi.ROI` instances (or `None`).
    Contains "donor" and "acceptor" keys.
    """
    sources: Dict[str, Dict[int, Union[str, Sequence[str]]]]
    """Map of dataset name (see :py:meth:`add_dataset`) -> file id -> source
    image file name(s).
    """
    special_sources: Dict[str, Dict[int, Union[str, Sequence[str]]]]
    """Map of dataset name (see :py:meth:`add_dataset`) -> file id -> source
    image file name(s). This is for special purpose datasets. Allowed keys:
    - ``"registration"`` (fiducial markers for image registration)
    - ``"donor-profile"``, ``"acceptor-profile"`` (densly labeled samples for
        determination of excitation intensity profiles)
    - ``"donor-only"``, ``"acceptor-only"`` (samples for determination of
        leakage and direct excitation correction factors, respectively)
    - ``"multi-state"`` (sample featuring multiple (>= 2) FRET states for
        calculation of detection and excitation efficiency correction factors)
    """
    loc_data: Dict[str, pd.DataFrame]
    """Map of dataset name -> single-molecule localization data"""
    special_loc_data: Dict[str, pd.DataFrame]
    """Map of dataset name -> single-molecule localization data for
    special-purpose data (donor-only, acceptor-only)
    """
    track_data: Dict[str, pd.DataFrame]
    """Map of dataset name -> single-molecule tracking data"""
    special_track_data: Dict[str, pd.DataFrame]
    """Map of dataset name -> single-molecule tracking data for
    special-purpose data (donor-only, acceptor-only)
    """
    segment_images: Dict[str, Dict[str, np.ndarray]]
    """Map of dataset name -> file id -> image data for segmentation"""
    flatfield: Dict[str, _flatfield.Corrector]
    """Map of excitation channel name -> flatfield corrector instance"""
    locators: Dict[str, nbui.Locator]
    """Widgets for locating fiducials, donor and acceptor emission data"""
    def __init__(self, excitation_seq: str = "da",
                 data_dir: Union[str, Path] = ""):
        """Parameters
        ----------
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
        self.rois = {"donor": None, "acceptor": None}
        self.tracker = fret.SmFRETTracker(excitation_seq)
        self.sources = {}
        self.special_sources = {}
        self.loc_data = {}
        self.special_loc_data = {}
        self.track_data = {}
        self.special_track_data = {}
        self.segment_images = {}
        self.flatfield = {}
        self.data_dir = Path(data_dir)

        self.locators = {
            "donor": nbui.Locator(), "acceptor": nbui.Locator(),
            "reg_donor": nbui.Locator(), "reg_acceptor": nbui.Locator()}
        self.channel_splitter = nbui.ChannelSplitter()
        self.channel_splitter.channel_names = ("donor", "acceptor")
        self._channel_splitter_active = False
        self.channel_splitter.observe(self._channel_rois_set, "rois")

    def _get_files(self, files_re: str, acc_files_re: Optional[str] = None):
        files = io.get_files(files_re, self.data_dir)[0]
        if acc_files_re is not None:
            acc_files = io.get_files(acc_files_re, self.data_dir)[0]
            files = list(zip(files, acc_files))
        return {i: f for i, f in enumerate(files)}

    def add_dataset(self, key: str, files_re: str,
                    acc_files_re: Optional[str] = None):
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
        """
        files = self._get_files(files_re, acc_files_re)
        if not files:
            warnings.warn(f"Empty dataset added: {key}")
        self.sources[key] = files

    def split_channels(self, files_re: str) -> ipywidgets.Widget:
        """Split image data into donor and acceptor emission channels

        Define rectangular regions for the channels. This is applicable if
        both channels were recorded side-by-side using the same camera.

        Parameters
        ----------
        files_re
            Regular expression describing some examplary image file names.
            Names should be relative to :py:attr:`data_dir`. Use forward
            slashes as path separators.
        """
        self._splitter_active = False
        files = [self.data_dir / f
                 for f in self._get_files(files_re, None).values()]
        splt = _ChannelSplitter(self)
        splt.image_selector.images = files
        self.channel_splitter.image_display.auto_scale()

        self.channel_splitter.rois = (self.rois["donor"],
                                      self.rois["acceptor"])
        self._splitter_active = True
        return splt

    def _channel_rois_set(self, change=None):
        """Callback if channel ROIs were set using the ChannelSplitter UI"""
        if not self._splitter_active:
            return
        self.rois["donor"], self.rois["acceptor"] = \
            self.channel_splitter.rois

    def add_special_dataset(self,
                            kind: Literal["registration", "acceptor-profile",
                                          "donor-profile", "donor-only",
                                          "acceptor-only", "multi-state"],
                            files_re: str,
                            acc_files_re: Optional[str] = None):
        files = self._get_files(files_re, acc_files_re)
        if not files:
            warnings.warn(f"Empty dataset added: {kind}")
        self.special_sources[kind] = files

    def set_registration_loc_opts(self) -> nbui.Locator:
        """Display widget to set localization options for channel registration

        for image registration. After finishing, call
        :py:meth:`calc_registration` to create a
        :py:class:`multicolor.Registrator` object.

        Parameters
        ----------
        files_re
            Regular expression describing image file names relative to
            :py:attr:`data_dir`. Use forward slashes as path separators.
        acc_files_re
            If donor and acceptor emission are recorded in separate files,
            `files_re` refers to the donor files and this describes
            acceptor files.

        Returns
        -------
            Widget
        """
        don_loc = RegistrationLocator(self, "donor")
        acc_loc = RegistrationLocator(self, "acceptor")
        ret = ipywidgets.Tab([don_loc, acc_loc])
        ret.set_title(0, "donor")
        ret.set_title(1, "acceptor")
        return ret

    def calc_registration(self, plot: bool = True,
                          max_frame: Optional[int] = None,
                          params: Optional[Dict[str, Any]] = None
                          ) -> Optional[mpl.figure.FigureCanvasBase]:
        """Calculate transformation between color channels

        Localize beads using the options set with :py:meth:`set_bead_loc_opts`,
        find pairs and fit transformation. Store result in
        :py:attr:`self.tracker.registrator`.

        Parameters
        ----------
        plot
            If True, plot the fit results and return the figure canvas.
        max_frame
            Maximum frame number to consider. Useful if beads defocused in
            later frames. If `None` use all frames.
        params
            Passed to :py:meth:`multicolor.Registrator.determine_parameters`.

        Returns
        -------
            If ``plot=True``, return the figure canvas which can be displayed
            in Jupyter notebooks.
        """
        label = ipywidgets.Label(value="Starting…")
        display(label)

        n_files = len(self.special_sources["registration"])

        locs = {"donor": [], "acceptor": []}
        for n, img_file in enumerate(
                self.special_sources["registration"].values()):
            label.value = f"Locating beads (file {n+1}/{n_files})"

            im_seq, to_close = self._open_image_sequence(img_file)
            for chan in "donor", "acceptor":
                locator = self.locators[f"reg_{chan}"]
                lo = locator.batch_func(im_seq[chan][:max_frame],
                                        **locator.options)
                locs[chan].append(lo)
        label.layout = ipywidgets.Layout(display="none")
        cc = multicolor.Registrator(locs["donor"], locs["acceptor"])
        cc.determine_parameters(**params or {})
        self.tracker.registrator = cc

        if plot:
            fig, ax = plt.subplots(1, 2)
            cc.test(ax=ax)
            return fig.canvas

    def _pims_open_no_warn(self, f):
        pth = self.data_dir / f
        if pth.suffix.lower() == ".spe":
            # Disable warnings about file size being wrong which is caused
            # by SDT-control not dumping the whole file
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return pims.open(str(pth))
        return pims.open(str(pth))

    def _open_image_sequence(self, f):
        if isinstance(f, tuple):
            don = self._pims_open_no_warn(f[0])
            acc = self._pims_open_no_warn(f[1])
            to_close = [don, acc]
        else:
            don = acc = self._pims_open_no_warn(f)
            to_close = [don]

        ret = {}
        for ims, chan in [(don, "donor"), (acc, "acceptor")]:
            if self.rois[chan] is not None:
                ims = self.rois[chan](ims)
            ret[chan] = ims
        return ret, to_close

    def donor_sum(self, don_fr: Union[helper.Slicerator, np.ndarray], acc_fr) \
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
        don_fr_corr = self.tracker.registrator(don_fr, channel=1, cval=np.mean)
        s = _img_sum(don_fr_corr, acc_fr)
        return self.tracker.frame_selector.select(s, "d")

    def set_loc_opts(self):
        don_loc = DonorLocator(self)
        acc_loc = AcceptorLocator(self)
        ret = ipywidgets.Tab([don_loc, acc_loc])
        ret.set_title(0, "donor")
        ret.set_title(1, "acceptor")
        return ret

    def locate(self):
        """Locate single-molecule signals

        For this, use the settings selected with help of the widgets
        returned by :py:meth:`set_loc_opts` for donor and acceptor excitation
        or loaded with :py:meth:`load`.

        Localization data for each dataset is collected in the
        :py:attr:`loc_data` dictionary.
        """
        special_keys = (set(self.special_sources) &
                        {"donor-only", "acceptor-only", "multi-state"})
        num_files = (sum(len(s) for s in self.sources.values()) +
                     sum(len(self.special_sources[k]) for k in special_keys))
        cnt = 1
        label = ipywidgets.Label(value="Starting…")
        display(label)

        iter_data = itertools.chain(
            itertools.product(
                [self.loc_data], self.sources.items()),
            itertools.product(
                [self.special_loc_data],
                ((k, self.special_sources[k]) for k in special_keys)))
        for tgt, (key, files) in iter_data:
            ret = []
            for f in files.values():
                label.value = f"Locating {f} ({cnt}/{num_files})"
                cnt += 1

                im_seq, opened = self._open_image_sequence(f)

                don_fr = self.donor_sum(im_seq["donor"], im_seq["acceptor"])
                lo = self.locators["donor"].batch_func(
                    don_fr, **self.locators["donor"].options)

                acc_fr = self.tracker.frame_selector.select(im_seq["acceptor"],
                                                            "a")
                if len(acc_fr):
                    lo_a = self.locators["acceptor"].batch_func(
                        acc_fr, **self.locators["acceptor"].options)
                    lo = pd.concat([lo, lo_a]).sort_values("frame")
                    lo = lo.reset_index(drop=True)
                ret.append(lo)

                for o in opened:
                    o.close()
            tgt[key] = pd.concat(ret, keys=files.keys())

    def track(self, feat_radius: int = 4, bg_frame: int = 3,
              link_radius: float = 1.0, link_mem: int = 1, min_length: int = 4,
              bg_estimator: Union[Literal["mean", "median"],
                                  Callable[[np.array], float]] = "mean",
              image_filter: Optional[helper.Pipeline] =
                  lambda i: image.gaussian_filter(i, 1),  # noqa: E127
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
        special_keys = list(self.special_loc_data)
        num_files = (sum(len(s) for s in self.sources.values()) +
                     sum(len(self.special_sources[k]) for k in special_keys))
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

        iter_data = itertools.chain(
            itertools.product(
                [(self.loc_data, self.track_data)], self.sources.items()),
            itertools.product(
                [(self.special_loc_data, self.special_track_data)],
                ((k, self.special_sources[k]) for k in special_keys)))
        for (src, tgt), (key, files) in iter_data:
            ret = []
            new_p = 0  # Particle ID unique across files
            for fk, f in files.items():
                label.value = f"Tracking {f} ({cnt}/{num_files})"
                cnt += 1

                loc = src[key]
                try:
                    loc = loc.loc[fk].copy()
                except KeyError:
                    # No localizations in this file
                    continue

                # All localizations are in the acceptor channel as they were
                # found in transformed donor + acceptor images upon donor
                # excitation and acceptor images upon acceptor excitation
                acc_loc = loc
                don_loc = loc.iloc[:0]

                im_seq, opened = self._open_image_sequence(f)

                if image_filter is not None:
                    for chan in "donor", "acceptor":
                        im_seq[chan] = image_filter(im_seq[chan])

                try:
                    d = self.tracker.track(im_seq["donor"], im_seq["acceptor"],
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

                for o in opened:
                    o.close()
            tgt[key] = pd.concat(ret, keys=files.keys())

    def extract_segment_images(self, key: str = "s", channel: str = "donor"):
        """Get images for segmentation

        This extracts the images used for segmentation (e.g., cell outlines)
        for use with :py:class:`Analyzer`. Images are copied to
        :py:attr:`segment_images`.

        Parameters
        ----------
        key
            Character describing cell images in the excitation sequence
            ``excitation_seq`` parameter to :py:meth:`__init__`.
        """
        self.segment_images.clear()

        for k, files in self.sources.items():
            seg = {}
            for fk, f in files.items():
                im_seq, opened = self._open_image_sequence(f)
                seg_fr = self.tracker.frame_selector.select(im_seq[channel],
                                                            key)
                seg[fk] = np.array(seg_fr)
                for o in opened:
                    o.close()
            self.segment_images[k] = seg

    def make_flatfield(self, dest: Literal["donor", "acceptor"],
                       src: Optional[Literal["donor", "acceptor"]] = None,
                       frame: Union[int, slice, Literal["all"],
                                    Sequence[int]] = 0,
                       bg: Union[float, np.ndarray] = 200,
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
            use the same as `dest`.
        frame
            Which frame in the image sequences described by `files_re` to use.
        bg
            Camera background. Either a value or an image data array.
        smooth_sigma
            Sigma for Gaussian blur to smoothe images.
        gaussian_fit
            If `True`, perform a Gaussian fit to the excitation profiles.
        """
        if src is None:
            src = dest
        if frame == "all":
            frame = slice(None)
        elif isinstance(frame, int):
            frame = [frame]

        imgs = []
        all_opened = []
        try:
            for f in self.special_sources[f"{src}-profile"].values():
                im_seq, opened = self._open_image_sequence(f)
                all_opened.extend(opened)
                im = im_seq[src][frame]
                imgs.append(im)

            if src.startswith("a") and dest.startswith("d"):
                imgs = [self.tracker.registrator(i, channel=2)
                        for i in imgs]
            elif src != dest:
                raise ValueError("src != dest and (src != \"acceptor\" and "
                                 "dest != \"donor\")")

            self.flatfield[dest] = _flatfield.Corrector(
                imgs, smooth_sigma=smooth_sigma, bg=bg,
                gaussian_fit=gaussian_fit)
        finally:
            for o in all_opened:
                o.close()

    def make_flatfield_sm(self, dest: Literal["donor", "acceptor"],
                          keys: Union[Sequence[str], Literal["all"]] = "all",
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
            datasets.
        weighted
            Weigh data inversely to density when performing a Gaussian fit.
            Not very robust, thus the default is `False`.
        frame
            Select only data from this frame. If `None`, use the first frame
            corresponding to `dest`.
        """
        if not len(keys):
            return
        if keys == "all":
            keys = self.track_data.keys()

        if frame is None:
            e_seq = self.tracker.frame_selector.eval_seq(-1)
            frame = np.nonzero(e_seq == dest[0])[0][0]

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
        if r is not None:
            img_shape = (r.bottom_right[1] - r.top_left[1],
                         r.bottom_right[0] - r.top_left[0])
        else:
            raise NotImplementedError("Determination of image size without "
                                      "channel ROIs has not been implemented "
                                      "yet.")
        self.flatfield[dest] = _flatfield.Corrector(data, shape=img_shape,
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

        DataStore(localizations=self.loc_data,
                  special_localizations=self.special_loc_data,
                  tracks=self.track_data,
                  special_tracks=self.special_track_data,
                  flatfield=self.flatfield,
                  segment_images=self.segment_images, tracker=self.tracker,
                  rois=self.rois, loc_options=loc_options,
                  data_dir=self.data_dir, sources=self.sources,
                  special_sources=self.special_sources).save(file_prefix)

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
        ds = DataStore.load(file_prefix)
        ret = cls()

        for key in ("rois", "data_dir", "tracker", "segment_images",
                    "flatfield", "sources", "special_sources"):
            with contextlib.suppress(AttributeError):
                setattr(ret, key, getattr(ds, key))
        with contextlib.suppress(AttributeError):
            ret.loc_data = ds.localizations
        with contextlib.suppress(AttributeError):
            ret.special_loc_data = ds.special_localizations
        with contextlib.suppress(AttributeError):
            ret.track_data = ds.tracks
        with contextlib.suppress(AttributeError):
            ret.special_track_data = ds.special_tracks

        with contextlib.suppress(AttributeError):
            for n, lo in ds.loc_options.items():
                ret.locators[n].set_settings(lo)

        return ret
