# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import contextlib
import enum
from pathlib import Path
import re
import sys
import warnings

from PyQt5 import QtCore, QtQml, QtQuick, QtWidgets
import numpy as np
import pandas as pd
from sdt import fret, gui, io, multicolor

from .data_store import DataStore


class Dataset(gui.Dataset):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._channels = {}
        self._frameSel = multicolor.FrameSelector("")
        self._registrator = multicolor.Registrator()

        self.channelsChanged.connect(self._imageDataChanged)
        self.registratorChanged.connect(self._imageDataChanged)
        self.excitationSeqChanged.connect(self._imageDataChanged)

    channels = gui.SimpleQtProperty("QVariantMap")
    registrator = gui.SimpleQtProperty("QVariant")

    excitationSeqChanged = QtCore.pyqtSignal()
    """:py:attr:`excitationSeq` changed"""

    @QtCore.pyqtProperty(str, notify=excitationSeqChanged)
    def excitationSeq(self) -> str:
        """Excitation sequence. See :py:class:`multicolor.FrameSelector` for
        details. No error checking es performend here.
        """
        return self._frameSel.excitation_seq

    @excitationSeq.setter
    def excitationSeq(self, seq: str):
        if seq == self.excitationSeq:
            return
        self._frameSel.excitation_seq = seq
        for d in self._data:
            d["particles"].frameSelector = self._frameSel
        self.excitationSeqChanged.emit()

    @QtCore.pyqtSlot(int, str, result=QtCore.QVariant)
    def get(self, index, role):
        if not (0 <= index <= self.rowCount() and role in self.roles):
            return None
        if role == "display":
            return "; ".join(str(self.get(index, r)) for r in self.fileRoles)
        if role in ("ddImg", "daImg", "aaImg"):
            r = "donor" if role == "ddImg" else "acceptor"
            chan = self.channels[r]
            fname = self.get(index, self.fileRoles[chan["source_id"]])
            fname = Path(self.dataDir, fname)
            seq = io.ImageSequence(fname).open()
            if chan["roi"] is not None:
                seq = chan["roi"](seq)
            if self._frameSel.excitation_seq:
                if role[0] == "d":
                    seq = self._frameSel.select(seq, role[0])
                else:
                    seq = self._frameSel.find_other_frames(seq, "d", "a")
            return seq
        return super().get(index, role)

    def _imageDataChanged(self):
        self.dataChanged.emit(self.index(0), self.index(self.count - 1),
                              [self.Roles.ddImg, self.Roles.daImg])


class DatasetCollection(gui.DatasetCollection):
    DatasetClass = Dataset

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dataRoles = ["display", "key", "ddImg", "daImg", "aaImg",
                          "particles"]
        self._channels = {"acceptor": {"source_id": 0, "roi": None},
                          "donor": {"source_id": 0, "roi": None}}
        self._excitationSeq = ""
        self._registrator = multicolor.Registrator()

        self.propagateProperty("channels")
        self.propagateProperty("excitationSeq")
        self.propagateProperty("registrator")

    channels = gui.SimpleQtProperty("QVariantMap")
    excitationSeq = gui.SimpleQtProperty(str)
    registrator = gui.SimpleQtProperty(QtCore.QVariant)


class ParticleList(gui.ListModel):
    class Roles(enum.IntEnum):
        number = QtCore.Qt.UserRole
        display = enum.auto()
        smData = enum.auto()
        dTrackData = enum.auto()
        aTrackData = enum.auto()
        manualFilter = enum.auto()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._smData = None
        self._filterTable = pd.DataFrame(columns=["track_len", "manual"])
        self._frameSel = multicolor.FrameSelector("")
        self._trackLengthRange = [0, np.inf]
        self._showManuallyFiltered = True
        self.showManuallyFilteredChanged.connect(self._updateFiltered)
        self.filterTableChanged.connect(self._updateFiltered)
        self._hideInterpolated = False
        self.hideInterpolatedChanged.connect(
            lambda: self.itemsChanged.emit(0, self.rowCount(),
                                           ["dTrackData", "aTrackData",
                                            "smData"]))

    hideInterpolated = gui.SimpleQtProperty(bool)

    smDataChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(QtCore.QVariant, notify=smDataChanged)
    def smData(self):
        return self._smData

    @smData.setter
    def smData(self, t):
        if len(t):
            t = t[t["fret", "exc_type"] == "d"].copy()
            rn = self._frameSel.renumber_frames(t["donor", "frame"], "d")
            t["donor", "frame"] = t["acceptor", "frame"] = rn

        self._smData = t
        ap = np.sort(t["fret", "particle"].unique())
        self._filterTable = pd.DataFrame(
            {"track_len": np.zeros(len(ap), dtype=int),
             "manual": np.full(len(ap), -1, dtype=int)}, index=ap)
        self.reset(self._filterTable.index.tolist())

        self.smDataChanged.emit()

    excitationSeqChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(QtCore.QVariant, notify=excitationSeqChanged)
    def excitationSeq(self):
        return self._frameSel.excitation_seq

    @excitationSeq.setter
    def excitationSeq(self, s):
        if s == self._frameSel.excitation_seq:
            return
        self._frameSel.excitation_seq = s
        self.itemsChanged.emit(0, self.rowCount(),
                               ["dTrackData", "aTrackData"])

    trackLengthRangeChanged = QtCore.pyqtSignal()

    @QtCore.pyqtProperty(list, notify=trackLengthRangeChanged)
    def trackLengthRange(self):
        return self._trackLengthRange

    @trackLengthRange.setter
    def trackLengthRange(self, r):
        if r == self._trackLengthRange:
            return

        m = self._smData.groupby(("fret", "particle")).apply(
            lambda x: int(not r[0] <= np.ptp(x["donor", "frame"]) <= r[1]))
        self._filterTable["track_len"] = m
        self._updateFiltered()
        self._trackLengthRange = r
        self.trackLengthRangeChanged.emit()

    showManuallyFiltered = gui.SimpleQtProperty(bool)
    filterTable = gui.SimpleQtProperty(QtCore.QVariant, comp=None)

    def get(self, index, role):
        if not 0 <= index < self.rowCount():
            return None
        n = self._data[index]
        if role == "number":
            return n
        if role == "display":
            return str(n)
        if role == "manualFilter":
            return int(self._filterTable.loc[n, "manual"])
        d = self._smData[self._smData["fret", "particle"] == n]
        if self._hideInterpolated:
            d = d[d["fret", "interp"] == 0]
        if role == "smData":
            return d
        if role in ("dTrackData", "aTrackData"):
            d = d["donor" if role[0] == "d" else "acceptor"].copy()
            d["particle"] = 0  # Fake particle number for TrackDisplay
            d["size"] = 3.0  # Size is often undefined. Just draw large circle
            return d

    @QtCore.pyqtSlot(int, bool)
    def manuallyFilterTrack(self, index, reject):
        pNo = self.get(index, "number")
        self._filterTable.loc[pNo, "manual"] = int(reject)
        self._updateFiltered()

    def _updateFiltered(self):
        mask = self._filterTable["track_len"] == 0
        if not self._showManuallyFiltered:
            mask &= self._filterTable["manual"] < 1
        newList = self._filterTable.index.to_numpy()[mask]

        dSet = set(self._data)
        nSet = set(newList)
        rm = iter(sorted(dSet - nSet, reverse=True))
        ins = iter(sorted(nSet - dSet, reverse=True))
        curRm = next(rm, -1)
        curIns = next(ins, -1)
        for i in range(len(self._data) - 1, -1, -1):
            p = self._data[i]
            if curRm == p:
                self.remove(i)
                curRm = next(rm, -1)
            else:
                while curIns > p:
                    self.insert(i+1, curIns)
                    curIns = next(ins, -1)
        while curIns >= 0:
            self.insert(0, curIns)
            curIns = next(ins, -1)


class Backend(QtCore.QObject):
    prefixRe = re.compile(r"(.+)-v(\d{3}).yaml")

    def __init__(self, parent=None):
        super().__init__(parent)

        self._datasets = DatasetCollection()
        self._figureCanvas = None
        self._filePath = None
        self._minTrackLength = 0
        self._maxTrackLength = 10000

        self.figureCanvasChanged.connect(self._setupFigure)

    figureCanvas = gui.SimpleQtProperty(QtQuick.QQuickItem)

    @QtCore.pyqtProperty(QtCore.QVariant, constant=True)
    def datasets(self):
        return self._datasets

    minTrackLength = gui.SimpleQtProperty(int)
    maxTrackLength = gui.SimpleQtProperty(int)

    @QtCore.pyqtSlot(QtCore.QUrl)
    def load(self, url):
        if isinstance(url, QtCore.QUrl):
            self._filePath = Path(url.toLocalFile())
        else:
            self._filePath = Path(url)
        prefix, fileVer = self.prefixRe.match(self._filePath.name).groups()
        fileVer = int(fileVer)

        with io.chdir(self._filePath.parent):
            ld = DataStore.load(prefix, loc=False, segment_images=False,
                                flatfield=False, version=fileVer)

        with contextlib.suppress(AttributeError, KeyError):
            self.minTrackLength = ld.filter["track_len"]["min"]
        with contextlib.suppress(AttributeError, KeyError):
            self.maxTrackLength = ld.filter["track_len"]["max"]

        self.datasets.clear()
        self.datasets.excitationSeq = ld.tracker.excitation_seq
        self.datasets.registrator = ld.tracker.registrator
        if ld.data_dir.is_absolute():
            self.datasets.dataDir = str(ld.data_dir)
        else:
            self.datasets.dataDir = str(
                (self._filePath.parent / ld.data_dir).resolve())

        for key, fileList in ld.sources.items():
            if not fileList:
                continue
            self.datasets.append(key)
            ds = self.datasets.get(self.datasets.rowCount() - 1, "dataset")

            if isinstance(next(iter(fileList.values())), str):
                ids = {"donor": 0, "acceptor": 0}
                ds.fileRoles = ["source_0"]
                sourceCount = 1
            else:
                ids = {"donor": 0, "acceptor": 1}
                ds.fileRoles = ["source_0", "source_1"]
                sourceCount = 2
            ds.channels = {chan: {"source_id": ids[chan],
                                  "roi": ld.rois[chan]}
                           for chan in ("donor", "acceptor")}
            modelFileList = []
            trc = ld.tracks[key]
            # Compute apparent eff and stoi
            ana = fret.SmFRETAnalyzer(trc, reset_filters=False)
            ana.calc_fret_values(a_mass_interp="next", skip_neighbors=False)
            trc = ana.tracks
            for trcKey, files in fileList.items():
                if sourceCount < 2:
                    entry = {"source_0": files}
                else:
                    entry = {f"source_{i}": f for i, f in enumerate(files)}
                entry["key"] = trcKey
                pList = ParticleList(ds)
                pList.excitationSeq = ds.excitationSeq
                try:
                    pTrc = trc.loc[trcKey].copy()
                except KeyError:
                    mi = pd.MultiIndex.from_tuples(
                        [(c, col)
                         for c in ("donor", "acceptor")
                         for col in ("x", "y", "frame", "mass")] +
                        [("fret", col) for col in ("particle", "a_mass",
                                                   "eff_app", "stoi_app",
                                                   "exc_type")])
                    pTrc = pd.DataFrame(columns=mi)
                pList.smData = pTrc
                ft = pList.filterTable
                for c in "track_len", "manual":
                    if ("filter", c) not in pTrc.columns:
                        continue
                    ft[c] = pTrc.groupby(("fret", "particle")).apply(
                        lambda x: x["filter", c].iloc[0])
                pList.filterTable = ft
                entry["particles"] = pList
                modelFileList.append(entry)
            ds.reset(modelFileList)

    @QtCore.pyqtSlot()
    @QtCore.pyqtSlot(bool)
    def save(self, exportSpreadsheet=False):
        prefix, fileVer = self.prefixRe.match(self._filePath.name).groups()
        fileVer = int(fileVer)

        with io.chdir(self._filePath.parent):
            ld = DataStore.load(prefix, loc=False, segment_images=False,
                                flatfield=False, version=fileVer)
        ld.filter = {"track_len": {"min": self.minTrackLength,
                                   "max": self.maxTrackLength}}

        with warnings.catch_warnings():
            import tables
            warnings.simplefilter("ignore", tables.NaturalNameWarning)

            ew = None
            try:
                if exportSpreadsheet:
                    ew = pd.ExcelWriter(self._filePath.with_suffix(".xlsx"))
                for i in range(self.datasets.rowCount()):
                    key = self.datasets.get(i, "key")
                    ds = self.datasets.get(i, "dataset")

                    loaded = ld.tracks[key]
                    filt_arr = np.full((len(loaded), 2), -1)
                    idx_lvl_0 = loaded.index.get_level_values(0)
                    for j in range(ds.rowCount()):
                        pList = ds.get(j, "particles")
                        f = ds.get(j, "key")
                        try:
                            p = loaded.loc[f, ("fret", "particle")]
                        except KeyError:
                            continue
                        filt = pList.filterTable.loc[p]
                        # Cannot use
                        # loaded.loc[f, ("filter", "track_len")] = \
                        #     filt["track_len"].to_numpy()
                        # as this does not work if f is a tuple
                        idx_mask = idx_lvl_0 == f
                        filt_arr[idx_mask, 0] = filt["track_len"].to_numpy()
                        filt_arr[idx_mask, 1] = filt["manual"].to_numpy()
                    loaded["filter", "track_len"] = filt_arr[:, 0]
                    loaded["filter", "manual"] = filt_arr[:, 1]
                    # Categorical exc_type does not allow for storing in fixed
                    # format while multiindex for both rows and columns does
                    # not work with table format…
                    if exportSpreadsheet:
                        loaded.to_excel(ew, sheet_name=key)
                if not exportSpreadsheet:
                    with io.chdir(self._filePath.parent):
                        ld.save(prefix, mode="update")
            finally:
                if ew:
                    ew.close()

    @QtCore.pyqtSlot(QtCore.QVariant, result=int)
    def frameCount(self, fileData):
        try:
            return len(fileData.toVariant()["ddImg"])
        except (AttributeError, TypeError):
            # e.g. fileData is None, ddImg is None
            return 0

    @QtCore.pyqtSlot(QtCore.QVariant, result=int)
    def firstFrame(self, t):
        if t is None:
            return 0
        return t["frame"].min()

    @QtCore.pyqtSlot(QtCore.QVariant, result=int)
    def lastFrame(self, t):
        if t is None:
            return 0
        return t["frame"].max()

    @QtCore.pyqtSlot(QtCore.QVariant, int, result=QtCore.QVariant)
    def image(self, imageSeq, frameNo):
        if imageSeq is None:
            return None
        return imageSeq[frameNo]

    def _setupFigure(self):
        fig = self.figureCanvas.figure
        fig.clf()
        fig.set_constrained_layout(True)
        self._ax = fig.subplots(1, 2)

    @QtCore.pyqtSlot(QtCore.QVariant)
    @QtCore.pyqtSlot(QtCore.QVariant, bool)
    def plot(self, t, scatter=True):
        if t is None:
            return
        self._ax[0].cla()
        self._ax[0].plot(t["donor", "frame"], t["donor", "mass"], "g",
                         label="D $\\to$ D")
        self._ax[0].plot(t["acceptor", "frame"], t["acceptor", "mass"], "r",
                         label="D $\\to$ A")
        self._ax[0].plot(t["acceptor", "frame"], t["fret", "a_mass"], "b",
                         label="A $\\to$ A")
        self._ax[0].set_title("fluorescence")
        self._ax[0].set_xlabel("frame no.")
        self._ax[0].set_ylabel("intensity")

        self._ax[1].cla()
        if scatter:
            has_neigh = t["fret", "has_neighbor"].astype(bool)
            wo_neigh = t[~has_neigh]
            w_neigh = t[has_neigh]
            self._ax[1].scatter(wo_neigh["fret", "eff_app"],
                                wo_neigh["fret", "stoi_app"],
                                label="no neighbor")
            self._ax[1].scatter(w_neigh["fret", "eff_app"],
                                w_neigh["fret", "stoi_app"],
                                label="has neighbor")
            self._ax[1].set_xlim(-0.25, 1.25)
            self._ax[1].set_ylim(-0.25, 1.25)
            self._ax[1].set_xlabel("apparent efficiency")
            self._ax[1].set_ylabel("apparent stoichiometry")
        else:
            self._ax[1].plot(t["donor", "frame"], t["fret", "eff_app"], "C0",
                             label="app. eff.")
            self._ax[1].plot(t["donor", "frame"], t["fret", "stoi_app"], "C1",
                             label="app. stoi.")
            self._ax[1].set_title("FRET")
            self._ax[1].set_xlabel("frame no.")
            self._ax[1].set_ylabel("value")

        color = "gray"
        alpha = 0.4
        hn = t["fret", "has_neighbor"].to_numpy()
        fno = t["donor", "frame"].to_numpy()
        diff_idx = np.diff(hn)
        up_idx = np.nonzero(diff_idx == 1)[0].tolist()
        down_idx = np.nonzero(diff_idx == -1)[0].tolist()
        if hn[0]:
            up_idx.insert(0, 0)
        if hn[-1]:
            down_idx.append(len(t) - 1)
        up_fno = fno[up_idx]
        down_fno = fno[down_idx]
        for a in self._ax[:1] if scatter else self._ax:
            hn_drawn = None
            for u, d in zip(up_fno, down_fno):
                hn_drawn = a.axvspan(u - 0.5, d + 0.5, facecolor=color,
                                     alpha=alpha)

            han, lab = a.get_legend_handles_labels()
            if hn_drawn is not None:
                han.append(hn_drawn)
                lab.append("has neighbor")
            a.legend(han, lab, loc=0, labelspacing=0.05)
        if scatter:
            self._ax[1].legend(loc=0, labelspacing=0.05)

        self.figureCanvas.draw_idle()


QtQml.qmlRegisterType(Backend, "FRETInspector", 1, 0, "Backend")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setOrganizationName("schuetzgroup")
    app.setOrganizationDomain("biophysics.iap.tuwien.ac.at")
    app.setApplicationName("FRETInspector")
    app.setApplicationVersion("0.1")

    argp = argparse.ArgumentParser(
        description="Inspect and manually filter smFRET traces")
    argp.add_argument("tracks", help="Tracking yaml file", nargs="?")
    args = argp.parse_args()

    # gui.mpl_use_qt_font()

    comp = gui.Component(Path(__file__).absolute().with_suffix(".qml"))
    if comp.status_ == gui.Component.Status.Error:
        sys.exit(1)
    if args.tracks is not None:
        comp.backend.load(args.tracks)

    sys.exit(app.exec_())
