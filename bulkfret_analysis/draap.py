# SPDX-FileCopyrightText: 2020 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import OrderedDict
from pathlib import Path
import warnings

import cv2
from ipywidgets import BoundedIntText, Button, Dropdown, HBox, VBox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pims

from sdt import flatfield, fret, image, io, nbui, roi


class Draap:
    def __init__(self, excitation_seq, roi, data_dir, background=0):
        self.channel_roi = roi
        self.data_dir = Path(data_dir)
        self.background = background

        self.cell_thresholder = None
        self.cell_selector = None

        self.sources = OrderedDict()
        self.rois = OrderedDict()
        self.exc_img_filter = fret.FrameSelector(excitation_seq)

        self.beam_shape = np.ones(roi.size[::-1])
        self.beam_shape_roi = None
        self._beam_shape_fig = None
        self._beam_shape_artist = None

    def _open_image_sequences(self, files):
        r = self.channel_roi or (lambda x: x)
        return OrderedDict(
            [(f, r(pims.open(str(self.data_dir / f)))) for f in files])

    def add_dataset(self, key, files_re, cells=False):
        files = io.get_files(files_re, self.data_dir)[0]
        if not files:
            warnings.warn(f"Empty dataset added: {key}")
        s = OrderedDict(
            [("files", self._open_image_sequences(files)), ("cells", cells)])
        self.sources[key] = s
        if cells:
            self.rois[key] = None

    def background_from_dark_img(self, files_re):
        files = io.get_files(files_re, self.data_dir)[0]
        bg = []
        for f in files:
            with pims.open(str(self.data_dir / f)) as img:
                img_don = self.channel_roi(self.exc_img_filter(img, "d"))
                bg.append(img_don[0])
        self.background = np.mean(bg, axis=0)

    def beam_shape_from_files(self, files_re, frame=0, **kwargs):
        files = io.get_files(files_re, self.data_dir)[0]
        imgs = []
        for f in files:
            with pims.open(str(self.data_dir / f)) as fr:
                imgs.append(self.channel_roi(fr[frame]))

        bs = np.median(imgs, axis=0)
        self.beam_shape = flatfield.Corrector([bs], **kwargs).corr_img

    def beam_shape_from_post_bleach(self, **kwargs):
        imgs = []
        for s in self.sources.values():
            if s["cells"]:
                continue
            for i in s["files"].values():
                imgs.append(self.exc_img_filter(i, "d")[1])
        bs = np.median(imgs, axis=0)
        kwargs.setdefault("bg", self.background)
        self.beam_shape = flatfield.Corrector([bs], **kwargs).corr_img

    def find_beam_shape_thresh(self):
        if self._beam_shape_fig is None:
            fig, ax = plt.subplots()
            self._beam_shape_fig = fig

        ax = self._beam_shape_fig.axes[0]

        thresh_sel = BoundedIntText(value=50, min=0, max=100,
                                    description="threshold")

        def update(change=None):
            if self._beam_shape_artist is not None:
                self._beam_shape_artist.remove()
            self._beam_shape_artist = ax.imshow(
                self.beam_shape * 100 > thresh_sel.value)
            self._beam_shape_fig.canvas.draw_idle()

        thresh_sel.observe(update, "value")
        update()

        return VBox([thresh_sel, self._beam_shape_fig.canvas])

    def threshold_beam_shape(self, thresh):
        self.beam_shape_roi = roi.MaskROI(self.beam_shape > thresh / 100)

    def draw_cell_rois(self):
        if self.cell_selector is None:
            self.cell_selector = nbui.ROISelector()
            self.cell_selector.categories = ["cell", "background"]
            self.cell_selector.auto_category = True

        dataset_sel = Dropdown(
            options=[k for k, v in self.sources.items() if v["cells"]],
            description="data set")
        save_button = Button(description="save ROIs", button_style="success",
                             tooltip="do this before changing the dataset")

        def dataset_changed(change=None):
            imgs = OrderedDict()
            src = self.sources[dataset_sel.value]
            imgs = OrderedDict()
            for k, i in src["files"].items():
                imgs[k] = self.exc_img_filter(i, "c")[0]
            self.cell_selector.images = imgs
            if self.rois[dataset_sel.value] is not None:
                self.cell_selector.rois = self.rois[dataset_sel.value]

        def save_clicked(b=None):
            self.rois[dataset_sel.value] = self.cell_selector.rois

        dataset_sel.observe(dataset_changed, "value")
        save_button.on_click(save_clicked)

        dataset_changed()

        return VBox([HBox([dataset_sel, save_button]), self.cell_selector])

    def _get_cell_imgs(self):
        cell_imgs = OrderedDict()
        for src in self.sources.values():
            if not src["cells"]:
                continue
            for fname, imgs in src["files"].items():
                cell_imgs[fname] = self.exc_img_filter(
                    imgs, "c")[0]
        return cell_imgs

    def find_cell_threshold(self):
        if self.cell_thresholder is None:
            self.cell_thresholder = nbui.Thresholder()
        self.cell_thresholder.images = self._get_cell_imgs()
        return self.cell_thresholder

    def threshold_cells(self, thresh_algorithm="adaptive",
                        background_erosion=20, **kwargs):
        cell_imgs = self._get_cell_imgs()

        if isinstance(thresh_algorithm, str):
            thresh_algorithm = getattr(image, thresh_algorithm + "_thresh")

        self.rois = OrderedDict()

        for typ, src in self.sources.items():
            if not src["cells"]:
                continue

            cell_rois = OrderedDict()
            bg_rois = OrderedDict()

            for fname in src["files"].keys():
                ci = cell_imgs[fname]
                cell_mask = thresh_algorithm(ci, **kwargs)

                e_struct = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * background_erosion + 1, 2 * background_erosion + 1),
                    (background_erosion, background_erosion))
                bg_mask = cv2.erode((~cell_mask).astype(np.uint8), e_struct)

                cell_rois[fname] = roi.MaskROI(cell_mask)
                bg_rois[fname] = roi.MaskROI(bg_mask.astype(bool))
            cur_rois = OrderedDict([("cell", cell_rois),
                                    ("background", bg_rois)])
            self.rois[typ] = cur_rois

    def analyze_cell_vs_bg(self):
        eff_diff = OrderedDict()

        for typ, src in self.sources.items():
            if not src["cells"]:
                continue

            eff_cells = []
            eff_bg = []

            for n, imgs in src["files"].items():
                imgs = [self.beam_shape_roi(i.astype(float) - self.background,
                                            fill_value=np.NaN)
                        for i in self.exc_img_filter(imgs, "d")]
                cell_roi = self.rois[typ]["cell"][n]
                bg_roi = self.rois[typ]["background"][n]
                
                if cell_roi is None or bg_roi is None:
                    warnings.warn(f"No ROI set for {typ}/{n}.")
                    eff_cells.append(np.NaN)
                    eff_bg.append(np.NaN)
                    continue

                imgs_cells = [cell_roi(i, fill_value=np.NaN) for i in imgs]
                imgs_bg = [bg_roi(i, fill_value=np.NaN) for i in imgs]

                for s, d in [(imgs_cells, eff_cells), (imgs_bg, eff_bg)]:
                    pre = np.nanmean(s[0])
                    post = np.nanmean(s[1])
                    d.append((post - pre) / post)

            eff_diff[typ] = pd.DataFrame(
                {"cell": eff_cells, "background": eff_bg,
                 "diff": np.subtract(eff_cells, eff_bg)},
                index=src["files"].keys())

        return eff_diff

    def test_roi_cell_vs_bg(self, data_pairs):
        eff_diff = OrderedDict()

        for cell_id, nocell_id in data_pairs:
            cell_rois = self.rois[cell_id]["cell"]
            bg_rois = self.rois[cell_id]["background"]

            eff_cells = []
            eff_bg = []

            for i, imgs in self.sources[nocell_id]["files"].items():
                imgs = [self.beam_shape_roi(i.astype(float) - self.background,
                                            fill_value=np.NaN)
                        for i in self.exc_img_filter(imgs, "d")]
                for r in cell_rois.keys():
                    cr = cell_rois[r]
                    br = bg_rois[r]
                    if cr is None or br is None:
                        continue
                        
                    imgs_cells = [cell_rois[r](i, fill_value=np.NaN)
                                  for i in imgs]
                    imgs_bg = [bg_rois[r](i, fill_value=np.NaN) for i in imgs]

                    for s, d in [(imgs_cells, eff_cells), (imgs_bg, eff_bg)]:
                        pre = np.nanmean(s[0])
                        post = np.nanmean(s[1])
                        d.append((post - pre) / post)

            eff_diff[cell_id] = pd.DataFrame(
                {"cell": eff_cells, "background": eff_bg,
                 "diff": np.subtract(eff_cells, eff_bg)},
                index=pd.MultiIndex.from_product(
                    [self.sources[nocell_id]["files"].keys(),
                     cell_rois.keys()],
                    names=["image", "roi"]))

        return eff_diff

    def analyze_cell_vs_nocell(self, data_pairs):
        eff_diff = OrderedDict()
        for cell_id, nocell_id in data_pairs:
            eff_cells = []
            eff_nocells = []

            for n, img_seq in self.sources[cell_id]["files"].items():
                cell_roi = self.rois[cell_id]["cell"][n]
                if cell_roi is None:
                    warnings.warn(f"No ROI set for {cell_id}/{n}.")
                    eff_cells.append(np.NaN)
                    eff_nocells.append(np.NaN)
                    continue

                cur_eff_cell = []
                cur_eff_nocell = []
                for src, dest in [([img_seq], cur_eff_cell),
                                  (self.sources[nocell_id]["files"].values(),
                                   cur_eff_nocell)]:
                    for s in src:
                        im = [self.beam_shape_roi(
                            i.astype(float) - self.background,
                            fill_value=np.NaN)
                              for i in self.exc_img_filter(s, "d")]
                        pre = np.nanmean(cell_roi(im[0], fill_value=np.NaN))
                        post = np.nanmean(cell_roi(im[1], fill_value=np.NaN))

                        dest.append((post - pre) / post)
                eff_cells.append(cur_eff_cell[0])
                eff_nocells.append(np.mean(cur_eff_nocell))

            eff_diff[cell_id] = pd.DataFrame(
                {"cells": eff_cells, "no cells": eff_nocells,
                 "diff": np.subtract(eff_cells, eff_nocells)},
                index=self.sources[cell_id]["files"].keys())

        return eff_diff

    def get_images(self, dataset, img, exc_type="d"):
        imgs = [i.astype(float) - self.background
                for i in self.exc_img_filter(self.sources[dataset]["files"][img], exc_type)]
        return imgs
