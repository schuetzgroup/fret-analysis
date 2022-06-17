# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import concurrent
import contextlib
import itertools
import multiprocessing
from pathlib import Path
from typing import (Any, Callable, Dict, List, Literal, Optional, Sequence,
                    Tuple, Union, overload)
import warnings

import numpy as np
import pandas as pd
from scipy import ndimage
from sdt import (brightness, flatfield as _flatfield, helper, io, loc,
                 multicolor, roi, spatial)
import trackpy
import traitlets

from ..data_store import DataStore


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


_loc_funcs = {"3D-DAOSTORM": loc.daostorm_3d.locate,
              "Crocker-Grier": loc.cg.locate}
_loc_batch_funcs = {"3D-DAOSTORM": loc.daostorm_3d.batch,
                    "Crocker-Grier": loc.cg.batch}


class BaseTracker(traitlets.HasTraits):
    """Class for tracking of smFRET data """

    frame_selector: multicolor.FrameSelector
    """A :py:class:`FrameSelector` instance with the matching
    :py:attr:`excitation_seq`.
    """
    registrator: multicolor.Registrator
    """multicolor.Registrator used to overlay channels"""
    data_dir: Path
    """All paths to source image files are relative to this"""
    rois: Dict[str, Union[roi.ROI, None]] = traitlets.Dict()
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
    registration_options: Dict[str, Any] = traitlets.Dict()
    """Options for image registration. See
    :py:meth:`multicolor.Registrator.determine_parameters` for details.
    """
    locate_options: Dict[str, Dict] = traitlets.Dict()
    """Localization algorithm and options for donor excitation (``"donor"``),
    acceptor excitation (``"acceptor"``), fiducial marker emission in the
    donor channel (``"reg_donor"``) and acceptor channel (``"reg_acceptor"``).
    """
    link_options: Dict[str, Any] = traitlets.Dict()
    """Options passed to :py:func:`trackpy.link`"""
    brightness_options: Dict[str, Any] = traitlets.Dict()
    """Options for fluorescence brightness measurement. See
    :py:func:`brightness.from_raw_image` for details.

    Additionally, ``smooth_sigma`` may be specified to apply a Gaussian filter
    with given sigma for noise reduction.
    """
    neighbor_distance: float = traitlets.Float(default_value=None,
                                               allow_none=True)
    """How far two features may be apart while still being considered close
    enough so that one influences the brightness measurement of the other.
    This is related to the `radius` option of
    :py:func:`brightness.from_raw_image`.

    If `None`, ``2 * brightness_options["radius"] + 1`` is used.
    """
    flatfield_options: Dict[str, Any] = traitlets.Dict()
    """Options for flatfield correction. See
    :py:meth:`flatfield.Corrector.__init__` for details.
    """
    sm_data: Dict[str, pd.DataFrame]
    """Map of dataset name -> single-molecule tracking data"""
    special_sm_data: Dict[str, pd.DataFrame]
    """Map of dataset name -> single-molecule tracking data for
    special-purpose data (donor-only, acceptor-only)
    """
    segment_images: Dict[str, Dict[str, np.ndarray]]
    """Map of dataset name -> file id -> image data for segmentation"""
    flatfield: Dict[str, _flatfield.Corrector]
    """Map of excitation channel name -> flatfield corrector instance"""

    def __init__(self, excitation_seq: str = "da",
                 data_dir: Union[str, Path] = "", link_quiet: bool = True):
        """Parameters
        ----------
        excitation_seq
            Excitation sequence. Use "d" for excitation, "a" for acceptor
            excitation, and "s" for a image for segmentation.
            The given sequence is repeated as necessary. E.g., "da" is
            equivalent to "dadadadada"... Defaults to "da".
        data_dir
            Data directory path. All data paths (e.g. in
            :py:meth:`add_dataset`) are take relative to this. Defaults to "",
            which is the current working directory.
        link_quiet
            If `True`, call :py:func:`trackpy.quiet`.
        """
        super().__init__()
        self.rois = {"donor": None, "acceptor": None}
        self.frame_selector = multicolor.FrameSelector(excitation_seq)
        self.sources = {}
        self.special_sources = {}
        self.sm_data = {}
        self.special_sm_data = {}
        self.segment_images = {}
        self.flatfield = {}
        self.flatfield_options = {"bg": 200, "smooth_sigma": 3.0,
                                  "gaussian_fit": False}
        self.data_dir = Path(data_dir)
        self.registrator = multicolor.Registrator()
        if link_quiet:
            trackpy.quiet()

    @property
    def excitation_seq(self) -> str:
        """Excitation sequence. "d" stands for donor, "a" for acceptor,
        anything else describes other kinds of frames which are irrelevant for
        tracking.

        One needs only specify the shortest sequence that is repeated,
        i. e. "ddddaddddadddda" is the same as "dddda".
        """
        return self.frame_selector.excitation_seq

    @excitation_seq.setter
    def excitation_seq(self, seq: str):
        self.frame_selector.excitation_seq = seq

    @overload
    def _get_files(self, files_re: str, acc_files_re: None) -> Dict[int, str]:
        ...

    @overload
    def _get_files(self, files_re: str, acc_files_re: str
                   ) -> Dict[int, List[str]]:
        ...

    def _get_files(self, files_re, acc_files_re=None):
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

    def _img_open_no_warn(self, f: Union[Path, str]) -> io.ImageSequence:
        pth = self.data_dir / f
        if pth.suffix.lower() == ".spe":
            # Disable warnings about file size being wrong which is caused
            # by SDT-control not dumping the whole file
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                return io.ImageSequence(pth).open()
        return io.ImageSequence(pth).open()

    def _open_image_sequence(self,
                             f: Union[str, Tuple[str, str]]
                             ) -> Tuple[Dict, List[io.ImageSequence]]:
        if isinstance(f, tuple):
            don = self._img_open_no_warn(f[0])
            acc = self._img_open_no_warn(f[1])
            to_close = [don, acc]
        else:
            don = acc = self._img_open_no_warn(f)
            to_close = [don]

        ret = {}
        for ims, chan in [(don, "donor"), (acc, "acceptor")]:
            if self.rois[chan] is not None:
                ims = self.rois[chan](ims)
            ret[chan] = ims
        return ret, to_close

    def calc_registration_impl(
            self,
            progress_callback: Callable[[int, int], None] = lambda x, y: None):
        """Calculate transformation between color channels

        Localize beads using the options set with :py:meth:`set_bead_loc_opts`,
        find pairs and fit transformation. Store result in
        :py:attr:`self.tracker.registrator`.

        Parameters
        ----------
        plot
            If True, plot the fit results and return the figure canvas.
        progress_callback
            Takes number currently processed file (starting at 0) and file
            count as arguments. This can be used to update GUI elements to
            reflect the progress.
        """
        n_files = len(self.special_sources["registration"])

        locs = {"donor": [], "acceptor": []}
        for n, img_file in enumerate(
                self.special_sources["registration"].values()):
            im_seq, to_close = self._open_image_sequence(img_file)
            progress_callback(n, n_files)
            for chan in "donor", "acceptor":
                opts = self.locate_options[f"reg_{chan}"]
                lo = _loc_batch_funcs[opts["algorithm"]](
                    im_seq[chan], **opts["options"])
                locs[chan].append(lo)
        self.registrator = multicolor.Registrator(locs["donor"],
                                                  locs["acceptor"])
        self.registrator.determine_parameters(**self.registration_options)

    @overload
    def donor_sum(self, don_fr: helper.Slicerator, acc_fr: helper.Slicerator,
                  select_frames: bool) -> helper.Slicerator:
        ...

    @overload
    def donor_sum(self, don_fr: np.ndarray, acc_fr: np.ndarray,
                  select_frames: bool) -> np.ndarray:
        ...

    def donor_sum(self, don_fr, acc_fr, select_frames=True):
        """Get sum image (sequence) of both channels upon donor excitation

        Transform donor channel image(s) and add acceptor channel image(s).

        Parameters
        ----------
        don_fr, acc_fr
            Image (sequence) of donor and acceptor emission
        select_frames
            If `True`, use :py:attr:`frame_selector` to return only frames
            upon donor excitation. Only effective if `don_fr` and `acc_fr`
            are Slicerators.

        Returns
        -------
            Sum of transformed donor emission image(s) and acceptor emission
            image(s)
        """
        don_fr_corr = self.registrator(don_fr, channel=1, cval=np.mean)
        s = _img_sum(don_fr_corr, acc_fr)
        if select_frames and not isinstance(s, np.ndarray):
            return self.frame_selector.select(s, "d")
        return s

    def _get_neighbor_distance(self):
        """Get maximum distance up to which two signals are considered close

        Returns
        -------
        :py:attr:`neighbor_distance` if not `None`, else
        ``2 * self.brightness_options["radius"] + 1``.
        """
        if self.neighbor_distance is None:
            return 2 * self.brightness_options["radius"] + 1
        return self.neighbor_distance

    def _get_brightness_options_filter(self) -> Tuple[Dict, Callable]:
        """Get options for `brightness.from_raw_image` and filter function

        Returns
        -------
        Dict of options to be passed to
        :py:func:`sdt.brightness.from_raw_image` as well as a function that
        applies a Gaussian blur if `smooth_sigma` is in
        :py:attr:`brightness_options` or does nothing if not.
        """
        opts = self.brightness_options.copy()
        sig = opts.pop("smooth_sigma", 0)
        if sig > 0:
            @helper.pipeline
            def flt(x):
                return ndimage.gaussian_filter(
                    np.asanyarray(x, dtype=float), sig)
        else:
            def flt(x):
                return x
        return opts, flt

    def locate_frame(self, donor_frame: np.ndarray, acceptor_frame: np.ndarray,
                     exc_type: str) -> pd.DataFrame:
        """Locate single molecules in a single frame

        Which localization algorithm to use and corresponding options are
        taken from :py:attr:`locate_options`.

        Additionally, the feature brightness is determined for
        both donor and acceptor for raw image data using
        :py:func:`brightness.from_raw_image`. Parameters are taken from
        :py:attr:`brightness_options`.

        Also, near neighbors are detected as these distort brightness
        measurements. See :py:func:`sdt.spacial.has_near_neighbor` and
        :py:attr:`neighbor_radius`.

        Parameters
        ----------
        donor_frame
            Donor emission image
        acceptor_frame
            Acceptor emission image
        exc_type
            Use ``"donor"`` in case of donor excitation. Molecules are detected
            in the sum of donor and acceptor emission images. Use
            ``"acceptor"`` in case of acceptor excitation. Molecules are
            detected in the acceptor emission image.

        Returns
        -------
        Single-molecule data. Columns are described by a multi-index.
        ``"donor"`` and ``"acceptor"`` columns contain data like coordinates
        and brightness in the donor and acceptor emission channels,
        respectively. ``("fret", "has_neighbor")`` column specifies whether
        a molecule has a near neighbor.
        """
        if exc_type == "donor":
            loc_frame = self.donor_sum(donor_frame, acceptor_frame)
        else:
            loc_frame = acceptor_frame
        opts = self.locate_options[exc_type]
        loc_func = _loc_funcs[opts["algorithm"]]

        a_loc = loc_func(loc_frame, **opts["options"])
        a_loc["frame"] = 0  # needed for `brightness.from_raw_image`
        b_opts, b_filt = self._get_brightness_options_filter()
        brightness.from_raw_image(a_loc, [b_filt(acceptor_frame)], **b_opts)
        d_loc = self.registrator(a_loc, channel=2)
        brightness.from_raw_image(d_loc, [b_filt(donor_frame)], **b_opts)
        res = pd.concat({"donor": d_loc.drop(columns="frame"),
                         "acceptor": a_loc.drop(columns="frame")},
                        axis=1)

        # Flag localizations that are too close together
        nd = self._get_neighbor_distance()
        if nd > 0:
            spatial.has_near_neighbor(
                res, nd, columns={"coords": [("acceptor", "x"),
                                             ("acceptor", "y")]})
            # DataFrame.rename() does not work with MultiIndex
            cn = res.columns.tolist()
            cn[res.columns.get_loc(("has_neighbor", ""))] = ("fret",
                                                             "has_neighbor")
            res.columns = pd.MultiIndex.from_tuples(cn)
        else:
            res["fret", "has_neighbor"] = -1
        return res

    def locate_video(self, source: Union[str, Tuple[str]],
                     n_threads: int = multiprocessing.cpu_count()
                     ) -> Dict[str, pd.DataFrame]:
        opened = []
        ret = {}
        try:
            im_seq, opened = self._open_image_sequence(source)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=n_threads) as e:
                for et in "donor", "acceptor":
                    r = e.map(
                        lambda x, y: self.locate_frame(x, y, exc_type=et),
                        self.frame_selector.select(im_seq["donor"], et[0]),
                        self.frame_selector.select(im_seq["acceptor"], et[0]))
                    ret[et] = list(r)
        finally:
            for o in opened:
                o.close()

        # set frame number
        for et, rs in ret.items():
            fnos = self.frame_selector.renumber_frames(np.arange(len(rs)),
                                                       et[0], restore=True)
            for r, n in zip(rs, fnos):
                r["fret", "frame"] = n
            ret[et] = pd.concat(rs, ignore_index=True)
        return ret

    def locate_all(self,
                   progress_callback: Callable[[str, int, int],
                                               None] = lambda x, y, z: None):
        """Locate single-molecule signals

        For this, use the settings selected with help of the widgets
        returned by :py:meth:`set_loc_opts` for donor and acceptor excitation
        or loaded with :py:meth:`load`.

        Localization data for each dataset is collected in the
        :py:attr:`sm_data` dictionary.

        Parameters
        ----------
        progress_callback
            Before starting tracking for each video, this function is called
            with file name, current file number and total file number as
            arguments. Use this to display progress in an UI.
        """
        special_keys = (set(self.special_sources) &
                        {"donor-only", "acceptor-only", "multi-state"})
        num_files = (sum(len(s) for s in self.sources.values()) +
                     sum(len(self.special_sources[k]) for k in special_keys))
        cnt = 0

        iter_data = itertools.chain(
            itertools.product(
                [self.sm_data], self.sources.items()),
            itertools.product(
                [self.special_sm_data],
                ((k, self.special_sources[k]) for k in special_keys)))

        for tgt, (key, files) in iter_data:
            tgt[key] = {}
            for fid, f in files.items():
                progress_callback(f, cnt, num_files)
                cnt += 1
                tgt[key][fid] = self.locate_video(f)

    def track_video(self, loc_data: Dict[str, pd.DataFrame]):
        raise NotImplementedError("`track_video` needs to be "
                                  "implemented in child class")

    def track_all(self,
                  progress_callback: Callable[[str, int, int],
                                              None] = lambda x, y: None):
        """Link single-molecule localizations into trajectories

        This is to be called after :py:meth:`locate_all`. Results are
        appended to data in the :py:attr:`sm_data` dictionary.

        Parameters
        ----------
        progress_callback
            Before starting tracking for each video, this function is called
            with file name, current file number and total file number as
            arguments. Use this to display progress in an UI.
        """
        special_keys = list(self.special_sm_data)
        num_files = (sum(len(s) for s in self.sm_data.values()) +
                     sum(len(self.special_sm_data[k]) for k in special_keys))
        cnt = 0

        for key, sm_data, src in itertools.chain(
                ((k, v, self.sources) for k, v in self.sm_data.items()),
                ((k, self.special_sm_data[k], self.special_sources)
                 for k in special_keys)):
            for f_id, sm in sm_data.items():
                try:
                    cur_file = src[key][f_id]
                except KeyError:
                    cur_file = "unknown"
                progress_callback(cur_file, cnt, num_files)
                cnt += 1

                try:
                    self.track_video(sm)
                except Exception as e:
                    warnings.warn(f"Tracking failed for {cur_file}. "
                                  f"Reason: {e}")
                    sm["fret", "particle"] = -1

    def interpolate_missing_video(self, source: Union[str, Tuple[str, str]],
                                  loc_data: Dict[str, pd.DataFrame]):
        raise NotImplementedError("`interpolate_missing_video` needs to be "
                                  "implemented in child class")

    def interpolate_missing_all(
            self,
            progress_callback: Callable[[str, int, int],
                                        None] = lambda x, y: None):
        special_keys = list(self.special_sm_data)
        num_files = (sum(len(s) for s in self.sm_data.values()) +
                     sum(len(self.special_sm_data[k]) for k in special_keys))
        cnt = 0

        for key, sm_data, src in itertools.chain(
                ((k, v, self.sources) for k, v in self.sm_data.items()),
                ((k, self.special_sm_data[k], self.special_sources)
                 for k in special_keys)):
            for f_id, sm in sm_data.items():
                cur_file = src[key][f_id]
                progress_callback(cur_file, cnt, num_files)
                cnt += 1

                self.interpolate_missing_video(cur_file, sm)

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
                opened = []
                try:
                    im_seq, opened = self._open_image_sequence(f)
                    seg_fr = self.frame_selector.select(im_seq[channel], key)
                    seg[fk] = np.array(seg_fr)
                finally:
                    for o in opened:
                        o.close()
            self.segment_images[k] = seg

    def make_flatfield(self, excitation: Literal["donor", "acceptor"],
                       frame: Union[int, slice, Literal["all"],
                                    Sequence[int]] = 0,
                       emission: Optional[Literal["donor", "acceptor"]] = None
                       ):
        """Calculate flatfield correction from separate bulk images

        If images the excitation laser profile were recorded using
        homogeneously labeled surfaces, this method can be used to calculate
        the flatfield corrections for donor and acceptor excitation.
        Results are saved to :py:attr:`flatfield`.

        Parameters
        ----------
        excitation
            Whether to calculate the flatfield correction for the donor
            (``"donor"``) or acceptor (``"acceptor"``) excitation.
        frame
            Which frame in the image sequences described by `files_re` to use.
        emission
            Whether to use the donor or acceptor emission channel. If `None`,
            use the same as `excitation`.
        """
        if emission is None:
            emission = excitation
        if frame == "all":
            frame = slice(None)
        elif isinstance(frame, int):
            frame = [frame]

        imgs = []
        all_opened = []
        try:
            for f in self.special_sources[f"{excitation}-profile"].values():
                im_seq, opened = self._open_image_sequence(f)
                all_opened.extend(opened)
                im = im_seq[emission][frame]
                imgs.append(im)

            if emission.startswith("a") and excitation.startswith("d"):
                imgs = [self.registrator(i, channel=2, cval=np.mean)
                        for i in imgs]
            elif emission != excitation:
                raise ValueError(
                    "emission != excitation and (emission != \"acceptor\" and "
                    "excitation != \"donor\")")

            self.flatfield[excitation] = _flatfield.Corrector(
                imgs, **self.flatfield_options)
        finally:
            for o in all_opened:
                o.close()

    def make_flatfield_sm(self, excitation: Literal["donor", "acceptor"],
                          frame: Optional[int] = None,
                          keys: Union[Sequence[str], Literal["all"]] = "all"):
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
        frame
            Select only data from this frame. If `None`, use the first frame
            corresponding to `dest`.
        """
        if len(keys) < 1:
            return
        if keys == "all":
            keys = self.sm_data.keys()

        if frame is None:
            e_seq = self.frame_selector.eval_seq(-1)
            frame = np.nonzero(e_seq == excitation[0])[0][0]

        data = []
        for k in keys:
            for d in self.sm_data[k].values():
                d = d[(d["fret", "frame"] == frame) &
                      (d["fret", "has_neighbor"] == 0)]
                if excitation.startswith("d"):
                    mass = d["donor", "mass"] + d["acceptor", "mass"]
                else:
                    mass = d[excitation, "mass"]
                data.append(pd.DataFrame({"x": d[excitation, "x"],
                                          "y": d[excitation, "y"],
                                          "mass": mass}))
        data = pd.concat(data, ignore_index=True)

        r = self.rois[excitation]
        if r is not None:
            img_shape = r.size[::-1]
        else:
            raise NotImplementedError("Determination of image size without "
                                      "channel ROIs has not been implemented "
                                      "yet.")
        self.flatfield[excitation] = _flatfield.Corrector(
            data, shape=img_shape, **self.flatfield_options)

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
        DataStore(sm_data=self.sm_data,
                  special_sm_data=self.special_sm_data,
                  flatfield=self.flatfield,
                  segment_images=self.segment_images,
                  data_dir=self.data_dir,
                  excitation_seq=self.excitation_seq,
                  rois=self.rois,
                  registrator=self.registrator,
                  registration_options=self.registration_options,
                  locate_options=self.locate_options,
                  neighbor_distance=self.neighbor_distance,
                  brightness_options=self.brightness_options,
                  link_options=self.link_options,
                  flatfield_options=self.flatfield_options,
                  sources=self.sources,
                  special_sources=self.special_sources).save(file_prefix)

    @classmethod
    def load(cls, file_prefix: str = "tracking", sm_data: bool = True,
             segment_images: bool = True, flatfield: bool = True
             ) -> "BaseTracker":
        """Construct class instance from saved data

        Raw data needs to be accessible for this.

        Parameters
        ----------
        file_prefix
            Prefix used for saving via :py:meth:`save`.
        sm_data
            Whether to load single-molecule data.
        segment_images
            Whether to load cell images. Defaults to `True`.
        flatfield
            Whether to load flatfield corrections. Defaults to `True`.

        Returns
        -------
        New TrackerBase instance with attributes reflect saved settings and
        data.
        """
        ds = DataStore.load(file_prefix)
        ret = cls()

        for key in ("rois", "data_dir", "segment_images", "flatfield",
                    "excitation_seq", "registrator", "registration_options",
                    "locate_options", "neighbor_distance",
                    "brightness_options", "link_options", "flatfield_options",
                    "sources", "special_sources", "sm_data",
                    "special_sm_data"):
            with contextlib.suppress(AttributeError):
                setattr(ret, key, getattr(ds, key))

        return ret
