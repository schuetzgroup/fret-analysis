# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Provide :py:class:`Tracker` as a Jupyter notebook UI for smFRET tracking"""
from typing import Dict, Mapping

from IPython.display import display
import ipywidgets
import matplotlib.pyplot as plt
from sdt import nbui
import traitlets

from .. import base


class Locator(ipywidgets.VBox):
    settings: Dict = traitlets.Dict()

    def __init__(self, tracker: base.Tracker):
        self._tracker = tracker
        self._locator = nbui.Locator()
        self._files = []

        self._file_selector = ipywidgets.Dropdown(description="file")
        self._frame_selector = ipywidgets.BoundedIntText(
            description="frame", min=0, max=0)
        self._selector_box = ipywidgets.HBox([self._file_selector,
                                              self._frame_selector])

        super().__init__([self._selector_box, self._locator])

        self._opened = []

        # Also update when options are set
        # These are mapping file id -> file name, thus changeing options will
        # not necessarily change the value (file id)
        self._file_selector.observe(self._file_changed, ["value", "options"])
        self._frame_selector.observe(self._frame_changed, "value")
        # Assuming that changing the algorithm changes the options, this will
        # suffice
        self._locator.observe(self._loc_options_changed, "options")
        self._loc_options_changed()

    def _make_image_sequence(self, don, acc):
        raise NotImplementedError("_make_image_sequence() needs to be "
                                  "implemented by sub-class.")

    def _set_files(self, files: Mapping):
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

    def _loc_options_changed(self, change=None):
        o = {"algorithm": self._locator.algorithm,
             "options": self._locator.options}
        self.settings = o

    @traitlets.observe("settings")
    def _settings_changed(self, change=None):
        o = self.settings
        self._locator.algorithm = o["algorithm"]
        self._locator.options = o["options"]


class RegistrationLocator(Locator):
    def __init__(self, tracker: base.Tracker, channel: str):
        super().__init__(tracker)
        self._channel = channel

    def _make_image_sequence(self, don, acc):
        if self._channel == "donor":
            return don
        return acc

    def update_sources(self):
        self._set_files(self._tracker.special_sources["registration"])
        self._locator.image_display.auto_scale()


class FRETLocator(Locator):
    def __init__(self, tracker: base.Tracker):
        super().__init__(tracker)

        self._dataset_selector = ipywidgets.Dropdown(description="dataset")
        self._selector_box.children = [self._dataset_selector,
                                       *self._selector_box.children]

        self._dataset_selector.observe(self._dataset_changed, "value")

    def _dataset_changed(self, change=None):
        src = self._tracker.sources[self._dataset_selector.value]
        self._set_files(src)

    def update_sources(self):
        self._dataset_selector.options = list(self._tracker.sources)
        self._locator.image_display.auto_scale()


class DonorLocator(FRETLocator):
    def _make_image_sequence(self, don, acc):
        return self._tracker.donor_sum(don, acc)


class AcceptorLocator(FRETLocator):
    def _make_image_sequence(self, don, acc):
        return self._tracker.frame_selector.select(acc, "a")


class _ChannelSplitter(ipywidgets.VBox):
    def __init__(self, tracker):
        self.image_selector = nbui.ImageSelector()

        # The following fails due to output and input being arrays and `link`
        # trying to compare using a simple !=
        # traitlets.link((self.image_selector, "output"),
        #                (tracker.channel_splitter, "input"))
        # Therefore, use `observe`
        self._tracker = tracker
        self.image_selector.observe(self._image_selected, "output")

        super().__init__([self.image_selector, tracker.channel_splitter])

    def _image_selected(self, change=None):
        self._tracker.channel_splitter.input = self.image_selector.output


class BaseTrackerNbUI:
    """Jupyter Notebook/Lab UI for :py:class:`base.Tracker`

    This provides the UI part. Create a class derived from
    :py:class:`base.Tracker` and this for a UI to perform single-molecule FRET
    tracking in Jupyter notebooks.

    Examples
    --------
    >>> class Tracker(base.Tracker, BaseTrackerNbUI):
    ...     pass
    """

    def __init__(self):
        super().__init__()
        self._splitter_active = False

        # create UI elements for setting localization options
        self.locators = {"donor": DonorLocator(self),
                         "acceptor": AcceptorLocator(self),
                         "reg_donor": RegistrationLocator(self, "donor"),
                         "reg_acceptor": RegistrationLocator(self, "acceptor")}
        for k, v in self.locators.items():
            # monitor changes via UI
            v.observe(lambda c, k=k: self._ui_loc_options_changed(k, c),
                      "settings")
            # set base.Tracker.locate_options
            self._ui_loc_options_changed(k)

        # create UI elements for choosing emission channel ROIs
        self.channel_splitter = nbui.ChannelSplitter()
        self.channel_splitter.channel_names = ("donor", "acceptor")
        self.channel_splitter.observe(self._ui_channel_rois_changed, "rois")
        self.observe(self._cls_channel_rois_changed, "rois")
        self.observe(self._cls_loc_options_changed, "locate_options")

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

    def _ui_loc_options_changed(self, channel, change=None):
        lo = self.locate_options.copy()
        lo[channel] = self.locators[channel].settings
        self.locate_options = lo

    def _cls_loc_options_changed(self, change=None):
        for k, v in self.locate_options.items():
            self.locators[k].settings = v

    def set_registration_locate_options(self) -> ipywidgets.Widget:
        """Display widget to set localization options for channel registration

        After finishing, you should call :py:meth:`calc_registration`.

        Returns
        -------
        Widget which displays localization results for single frames.
        """
        don_loc = self.locators["reg_donor"]
        acc_loc = self.locators["reg_acceptor"]
        don_loc.update_sources()
        acc_loc.update_sources()
        ret = ipywidgets.Tab([don_loc, acc_loc])
        ret.set_title(0, "donor")
        ret.set_title(1, "acceptor")
        return ret

    def calc_registration(self) -> ipywidgets.Widget:
        """Calculate transformation between color channels

        Localize beads using the options set with
        :py:meth:`set_bead_locate_options`, find pairs and fit transformation.
        Store result in :py:attr:`registrator`.

        Returns
        -------
        Figure canvas which can be displayed in Jupyter notebooks.
        """
        label = ipywidgets.Label(value="Starting…")
        display(label)

        def update_label(n, n_files):
            label.value = f"Locating beads (file {n+1}/{n_files})"

        self.calc_registration_impl(update_label)
        label.layout = ipywidgets.Layout(display="none")

        fig, ax = plt.subplots(1, 2)
        self.registrator.test(ax=ax)
        return fig.canvas

    def set_locate_options(self):
        don_loc = self.locators["donor"]
        acc_loc = self.locators["acceptor"]
        don_loc.update_sources()
        acc_loc.update_sources()
        ret = ipywidgets.Tab([don_loc, acc_loc])
        ret.set_title(0, "donor")
        ret.set_title(1, "acceptor")
        return ret

    def locate(self):
        """Locate single-molecule signals

        For this, use the settings selected with help of the widgets
        returned by :py:meth:`set_locate_options` for donor and acceptor
        excitation or loaded with :py:meth:`load`.

        Localization data for each dataset is collected in the
        :py:attr:`sm_data` dictionary.
        """
        label = ipywidgets.Label(value="Starting…")
        display(label)

        def update_label(file, n, num_files):
            label.value = (f"Locating molecules in {file} "
                           f"({n + 1}/{num_files})…")

        self.locate_all(update_label)

        label.value = "Finished locating."

    def track(self):
        """Link single-molecule localizations into trajectories

        This updates data in :py:attr:`sm_data` with tracking information
        """
        label = ipywidgets.Label(value="Starting…")
        display(label)

        def update_label(file, n, num_files):
            label.value = ("Tracking molecules in "
                           f"{file} ({n + 1}/{num_files})")

        self.track_all(update_label)

        label.value = "Finished tracking."

    def interpolate_missing(self):
        label = ipywidgets.Label(value="Starting…")
        display(label)

        def update_label(file, n, num_files):
            label.value = ("Interpolating tracks in "
                           f"{file} ({n + 1}/{num_files})")

        self.interpolate_missing_all(update_label)

        label.value = "Finished interpolating."


class Tracker(base.Tracker, BaseTrackerNbUI):
    """Jupyter notebook UI for tracking intramolecular single-molecule FRET

    This allows for image registration, single molecule localization,
    tracking, and brightness determination. These are the most time-consuming
    tasks and are thus isolated from the rest of the analysis. Additionally,
    these tasks are the only ones that require access to raw image data, which
    can take up large amounts of disk space and are thus often located on
    external storage. After running the methods of this class, this storage
    can be disconnected and further filtering and analysis can be efficiently
    performed using the :py:class:`Analyzer` class.
    """


class IntermolecularTracker(base.IntermolecularTracker, BaseTrackerNbUI):
    """Jupyter notebook UI for tracking intermolecular single-molecule FRET

    This allows for image registration, single molecule localization,
    tracking, and brightness determination. These are the most time-consuming
    tasks and are thus isolated from the rest of the analysis. Additionally,
    these tasks are the only ones that require access to raw image data, which
    can take up large amounts of disk space and are thus often located on
    external storage. After running the methods of this class, this storage
    can be disconnected and further filtering and analysis can be efficiently
    performed using the :py:class:`Analyzer` class.
    """
