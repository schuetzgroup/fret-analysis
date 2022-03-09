# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provide :py:class:`Tracker` as a Jupyter notebook UI for smFRET tracking"""
from typing import Any, Callable, Dict, Optional, Union
try:
    from typing import Literal
except ImportError:
    # Python < 3.8
    from .typing_extensions import Literal

from IPython.display import display
import ipywidgets
import matplotlib.pyplot as plt
import numpy as np

from sdt import helper, image, nbui

from .tracker_base import TrackerBase


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


class TrackerNbUIBase:
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
    locators: Dict[str, nbui.Locator]
    """Widgets for locating fiducials, donor and acceptor emission data"""

    def __init__(self):
        super().__init__()
        self.locators = {
            "donor": nbui.Locator(), "acceptor": nbui.Locator(),
            "reg_donor": nbui.Locator(), "reg_acceptor": nbui.Locator()}
        self._splitter_active = False
        self.channel_splitter = nbui.ChannelSplitter()
        self.channel_splitter.channel_names = ("donor", "acceptor")
        self.channel_splitter.observe(self._ui_channel_rois_changed, "rois")
        self.observe(self._cls_channel_rois_changed, "rois")

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

        self.channel_splitter.rois = self.rois["donor"], self.rois["acceptor"]
        self._splitter_active = True
        return splt

    def _ui_channel_rois_changed(self, change=None):
        """Callback if channel ROIs were set using the ChannelSplitter UI"""
        if not self._splitter_active:
            return
        r = self.channel_splitter.rois
        self.rois = {"donor": r[0], "acceptor": r[1]}

    def _cls_channel_rois_changed(self, change=None):
        """Callback if channel ROIs were set using :py:attr:`rois`"""
        if not self._splitter_active:
            return
        self.channel_splitter.rois = self.rois["donor"], self.rois["acceptor"]

    def set_registration_loc_opts(self) -> ipywidgets.Widget:
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
                          ) -> Optional[ipywidgets.Widget]:
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

        def update_label(n, n_files):
            label.value = f"Locating beads (file {n+1}/{n_files})"

        self.calc_registration_impl(max_frame, params, update_label)
        label.layout = ipywidgets.Layout(display="none")

        if plot:
            fig, ax = plt.subplots(1, 2)
            self.registrator.test(ax=ax)
            return fig.canvas

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
        label = ipywidgets.Label(value="Starting…")
        display(label)

        def update_label(file, n, num_files):
            label.value = f"Locating {file} ({n + 1}/{num_files})"

        self.locate_impl(update_label)

        label.value = "Finished."

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
        label = ipywidgets.Label(value="Starting…")
        display(label)

        def update_label(file, n, num_files):
            label.value = f"Tracking {file} ({n + 1}/{num_files})"

        self.track_impl(update_label)

        label.value = "Finished"


class Tracker(TrackerBase, TrackerNbUIBase):
    pass
