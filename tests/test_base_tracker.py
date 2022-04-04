# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import itertools

import imageio
import numpy as np
import pandas as pd
import pytest
from sdt import flatfield, funcs, helper, io, roi, sim

from smfret_analysis import base


class TestBaseTracker:
    def test_excitation_seq(self):
        seq = "sdddax"
        tr = base.Tracker(seq)
        assert tr.excitation_seq == seq
        assert tr.frame_selector.excitation_seq == seq

        seq2 = "das"
        tr.excitation_seq = seq2
        assert tr.excitation_seq == seq2
        assert tr.frame_selector.excitation_seq == seq2

    def test_sources(self, tmp_path):
        # create files
        good_files_d = ["a_d.tif", "b_d.tif", "c_d.tif"]
        good_files_a = ["a_a.tif", "b_a.tif", "c_a.tif"]
        bad_files = ["a_d.jpg", "aa_d.tif", "d_d.tif"]
        for f in itertools.chain(good_files_d, good_files_a, bad_files):
            (tmp_path / f).touch()

        # regexes
        re_d = r"^[a-c]_d\.tif"
        re_a = r"^[a-c]_a\.tif"
        re_no_match = r".*\.bmp"
        # expected results
        exp_one_src = dict(enumerate(good_files_d))
        exp_two_src = dict(enumerate(zip(good_files_d, good_files_a)))

        # test _get_files()
        tr = base.Tracker(data_dir=tmp_path)
        assert tr._get_files(re_d) == exp_one_src
        assert tr._get_files(re_d, re_a) == exp_two_src
        assert tr._get_files(re_no_match) == {}
        assert tr._get_files(re_no_match, re_no_match) == {}

        # test add_dataset()
        tr.add_dataset("ds1", re_d)
        assert tr.sources.get("ds1") == exp_one_src
        tr.add_dataset("ds2", re_d, re_a)
        assert tr.sources.get("ds2") == exp_two_src
        with pytest.warns(UserWarning):
            tr.add_dataset("bad1", re_no_match)
        assert tr.sources.get("bad1") == {}
        with pytest.warns(UserWarning):
            tr.add_dataset("bad2", re_no_match, re_no_match)
        assert tr.sources.get("bad2") == {}

        # test add_special_dataset()
        tr.add_special_dataset("acceptor-profile", re_d)
        assert tr.special_sources.get("acceptor-profile") == exp_one_src
        tr.add_special_dataset("registration", re_d, re_a)
        assert tr.special_sources.get("registration") == exp_two_src
        with pytest.warns(UserWarning):
            tr.add_special_dataset("donor-profile", re_no_match)
        assert tr.special_sources.get("donor-profile") == {}
        with pytest.warns(UserWarning):
            tr.add_special_dataset("multi-state", re_no_match, re_no_match)
        assert tr.special_sources.get("multi-state") == {}

    def test_open_image_sequence(self, tmp_path):
        # Create image files
        ims1 = [np.array([[0, 0, 0, 1, 1, 1, 1],
                          [0, 0, 0, 1, 1, 1, 1]])]
        ims1.append(ims1[0] + 2)
        ims2 = [i + 4 for i in ims1]

        imageio.mimwrite(tmp_path / "img1.tif", ims1)
        imageio.mimwrite(tmp_path / "img2.tif", ims2)

        tr = base.Tracker(data_dir=tmp_path)
        # Without ROIs, single source
        ims, to_close = tr._open_image_sequence("img1.tif")
        assert "donor" in ims
        assert "acceptor" in ims
        np.testing.assert_array_equal(ims["donor"], ims1)
        np.testing.assert_array_equal(ims["acceptor"], ims1)
        for c in to_close:
            c.close()
        # Without ROIs, two sources
        ims, to_close = tr._open_image_sequence(("img1.tif", "img2.tif"))
        assert "donor" in ims
        assert "acceptor" in ims
        np.testing.assert_array_equal(ims["donor"], ims1)
        np.testing.assert_array_equal(ims["acceptor"], ims2)
        for c in to_close:
            c.close()
        # With ROIs, single source
        tr.rois = {"donor": roi.ROI((0, 0), (3, 2)),
                   "acceptor": roi.ROI((3, 0), (7, 2))}
        ims, to_close = tr._open_image_sequence("img1.tif")
        assert "donor" in ims
        assert "acceptor" in ims
        np.testing.assert_array_equal(ims["donor"], [i[:2, :3] for i in ims1])
        np.testing.assert_array_equal(ims["acceptor"],
                                      [i[:2, 3:] for i in ims1])
        for c in to_close:
            c.close()
        # With ROIs, two sources
        ims, to_close = tr._open_image_sequence(("img1.tif", "img2.tif"))
        assert "donor" in ims
        assert "acceptor" in ims
        np.testing.assert_array_equal(ims["donor"], [i[:2, :3] for i in ims1])
        np.testing.assert_array_equal(ims["acceptor"],
                                      [i[:2, 3:] for i in ims2])
        for c in to_close:
            c.close()

    @pytest.fixture
    def channel_transform(self):
        return np.array([[1.1, 0.1, 2.0],
                         [-0.2, 1.2, 1.0],
                         [0.0, 0.0, 1.0]])

    @pytest.fixture
    def coords(self, channel_transform):
        coords1 = [np.array([[20, 15], [110, 30], [50, 70], [30, 100]]),
                   np.array([[20, 110], [70, 50], [100, 105], [40, 20]]),
                   np.array([[50, 55], [30, 80], [100, 20], [65, 110]]),
                   np.array([[105, 25], [40, 80], [70, 30], [30, 90]])]
        coords2 = [c @ channel_transform[:2, :2].T + channel_transform[:2, -1]
                   for c in coords1]
        return coords1, coords2

    def test_calc_registration_impl(self, tmp_path, channel_transform, coords):
        d_roi = roi.ROI((0, 0), size=(140, 150))
        a_roi = roi.ROI((170, 0), size=d_roi.size)

        img_shape = (150, 320)
        imgs = []
        for c1, c2 in zip(*coords):
            im = np.full(img_shape, 200.0)
            im += sim.simulate_gauss(img_shape[::-1], c1, 500, 1,
                                     engine="python")
            im += sim.simulate_gauss(img_shape[::-1], c2 + a_roi.top_left,
                                     300, 1, engine="python")
            imgs.append(im)
        imageio.mimwrite(tmp_path / "beads1.tif", [imgs[0], imgs[1]])
        imageio.mimwrite(tmp_path / "beads2.tif", [imgs[2]])

        tr = base.Tracker(data_dir=tmp_path)
        tr.rois = {"donor": d_roi, "acceptor": a_roi}
        tr.add_special_dataset("registration", r"beads\d\.tif")
        lo = {"algorithm": "Crocker-Grier",
              "options": {"radius": 3, "signal_thresh": 400,
                          "mass_thresh": 1000}}
        tr.locate_options["reg_donor"] = lo.copy()
        lo["options"]["signal_thresh"] = 200
        tr.locate_options["reg_acceptor"] = lo.copy()

        cur_n = []
        total_n = []

        def progress_cb(cur, total):
            cur_n.append(cur)
            total_n.append(total)

        tr.calc_registration_impl(progress_callback=progress_cb)

        assert cur_n == [0, 1]
        assert total_n == [2, 2]
        np.testing.assert_allclose(tr.registrator.parameters1,
                                   channel_transform)

    def test_donor_sum(self, channel_transform, coords):
        img_shape = (150, 140)
        imgs_d = helper.Slicerator(
            [sim.simulate_gauss(img_shape[::-1], c1, 300, 1, engine="python")
             for c1 in coords[0]])
        imgs_a = helper.Slicerator(
            [sim.simulate_gauss(img_shape[::-1], c2, 200, 1, engine="python")
             for c2 in coords[1]])

        tr = base.Tracker("da")
        tr.registrator.parameters1 = channel_transform
        tr.registrator.parameters2 = np.linalg.inv(channel_transform)

        imgs_exp = [a + tr.registrator(d, channel=1, cval=np.mean)
                    for d, a in zip(imgs_d, imgs_a)]

        # Slicerator
        r = tr.donor_sum(imgs_d, imgs_a, select_frames=False)
        np.testing.assert_allclose(r, imgs_exp, atol=1e-6)
        r = tr.donor_sum(imgs_d, imgs_a, select_frames=True)
        np.testing.assert_allclose(r, [imgs_exp[0], imgs_exp[2]], atol=1e-6)

        # ndarray
        r = tr.donor_sum(imgs_d[0], imgs_a[0], select_frames=False)
        np.testing.assert_allclose(r, imgs_exp[0], atol=1e-6)
        r = tr.donor_sum(imgs_d[0], imgs_a[0], select_frames=True)
        np.testing.assert_allclose(r, imgs_exp[0], atol=1e-6)

    def test_locate_frame(self, channel_transform, coords):
        img_shape = (150, 140)
        img_dd = sim.simulate_gauss(img_shape[::-1], coords[0][0], 1300, 1,
                                    mass=True, engine="python") + 200
        img_da = sim.simulate_gauss(img_shape[::-1], coords[1][0], 2000, 1,
                                    mass=True, engine="python") + 200
        img_ad = np.full(img_shape, 200.0)
        img_aa = sim.simulate_gauss(img_shape[::-1], coords[1][1], 1300, 1,
                                    mass=True, engine="python") + 200

        tr = base.Tracker("da")
        tr.registrator.parameters1 = channel_transform
        tr.registrator.parameters2 = np.linalg.inv(channel_transform)
        lo = {"algorithm": "Crocker-Grier",
              "options": {"radius": 3, "signal_thresh": 400,
                          "mass_thresh": 1000}}
        tr.locate_options["donor"] = lo.copy()
        lo["options"]["signal_thresh"] = 120
        tr.locate_options["acceptor"] = lo.copy()
        tr.brightness_options = {"radius": 4, "bg_frame": 3, "mask": "circle"}

        d_loc = tr.locate_frame(img_dd, img_da, "donor")
        a_loc = tr.locate_frame(img_ad, img_aa, "acceptor")

        for c in itertools.product(["donor", "acceptor"],
                                   ["x", "y", "mass", "size", "ecc", "signal",
                                    "bg", "bg_dev"]):
            assert c in d_loc
            assert c in a_loc
        assert ("fret", "has_neighbor") in d_loc
        assert ("fret", "has_neighbor") in a_loc

        for (lo, chan), (j, i) in zip(
                itertools.product([d_loc, a_loc], ["donor", "acceptor"]),
                itertools.product([0, 1], [0, 1])):
            r = lo[[(chan, "x"), (chan, "y")]].to_numpy()
            r = r[np.argsort(r[:, 0])]
            e = coords[i][j]
            e = e[np.argsort(e[:, 0])]
            np.testing.assert_allclose(r, e)

        np.testing.assert_allclose(d_loc["donor", "mass"], 1300, atol=1.0)
        np.testing.assert_allclose(d_loc["acceptor", "mass"], 2000, atol=1.0)
        np.testing.assert_allclose(a_loc["donor", "mass"], 0, atol=1.0)
        np.testing.assert_allclose(a_loc["acceptor", "mass"], 1300, atol=1.0)
        np.testing.assert_allclose(d_loc["donor", "bg"], 200, atol=0.5)
        np.testing.assert_allclose(d_loc["acceptor", "bg"], 200, atol=0.5)
        np.testing.assert_allclose(a_loc["donor", "bg"], 200, atol=0.5)
        np.testing.assert_allclose(a_loc["acceptor", "bg"], 200, atol=0.5)

        # TODO: Check near neighbor detection
        img_dd = sim.simulate_gauss(img_shape[::-1], [[50, 60]], 1300, 1)

    def _check_locate_result(self, res, coords, dd_mass, da_mass, ad_mass,
                             aa_mass, bg):
        assert "donor" in res
        assert "acceptor" in res
        assert len(res) == 2

        d_frames = np.arange(0, len(coords[0]), 2)
        d_frames = np.repeat(d_frames, [len(c) for c in coords[0][::2]])
        np.testing.assert_equal(res["donor"]["fret", "frame"], d_frames)
        a_frames = np.arange(1, len(coords[0]), 2)
        a_frames = np.repeat(a_frames, [len(c) for c in coords[0][1::2]])
        np.testing.assert_equal(res["acceptor"]["fret", "frame"], a_frames)

        for r in res.values():
            for c in itertools.product(["donor", "acceptor"],
                                       ["x", "y", "mass", "size", "ecc",
                                        "signal", "bg", "bg_dev"]):
                assert c in r
            assert ("fret", "frame") in r
            assert ("fret", "has_neighbor") in r

        for lo in res["donor"], res["acceptor"]:
            for f, lo_f in lo.groupby(("fret", "frame")):
                for chan in "donor", "acceptor":
                    r = lo_f[[(chan, "x"), (chan, "y")]].to_numpy()
                    r = r[np.argsort(r[:, 0])]
                    e = coords[int(chan == "acceptor")][f]
                    e = e[np.argsort(e[:, 0])]
                    np.testing.assert_allclose(r, e)

        np.testing.assert_allclose(res["donor"]["donor", "mass"], dd_mass,
                                   atol=1.0)
        np.testing.assert_allclose(res["donor"]["acceptor", "mass"], da_mass,
                                   atol=1.0)
        np.testing.assert_allclose(res["acceptor"]["donor", "mass"], ad_mass,
                                   atol=1.0)
        np.testing.assert_allclose(res["acceptor"]["acceptor", "mass"],
                                   aa_mass, atol=1.0)
        np.testing.assert_allclose(res["donor"]["donor", "bg"], bg, atol=0.5)
        np.testing.assert_allclose(res["donor"]["acceptor", "bg"], bg,
                                   atol=0.5)
        np.testing.assert_allclose(res["acceptor"]["donor", "bg"], bg,
                                   atol=0.5)
        np.testing.assert_allclose(res["acceptor"]["acceptor", "bg"], bg,
                                   atol=0.5)

    def test_locate_video(self, tmp_path, channel_transform, coords):
        d_roi = roi.ROI((0, 0), size=(140, 150))
        a_roi = roi.ROI((170, 0), size=d_roi.size)

        img_shape = (150, 320)
        imgs = [sim.simulate_gauss(img_shape[::-1], c + a_roi.top_left,
                                   1300 if i % 2 else 2000,
                                   1, mass=True, engine="python") + 200
                for i, c in enumerate(coords[1])]
        for im, c in zip(imgs[::2], coords[0][::2]):
            im += sim.simulate_gauss(img_shape[::-1], c + d_roi.top_left, 1300,
                                     1, mass=True, engine="python")
        imageio.mimwrite(tmp_path / "single.tif", imgs)
        imageio.mimwrite(tmp_path / "split_don.tif", [d_roi(i) for i in imgs])
        imageio.mimwrite(tmp_path / "split_acc.tif", [a_roi(i) for i in imgs])

        tr = base.Tracker("da", data_dir=tmp_path)
        tr.registrator.parameters1 = channel_transform
        tr.registrator.parameters2 = np.linalg.inv(channel_transform)
        lo = {"algorithm": "Crocker-Grier",
              "options": {"radius": 3, "signal_thresh": 400,
                          "mass_thresh": 1000}}
        tr.locate_options["donor"] = lo.copy()
        lo["options"]["signal_thresh"] = 120
        tr.locate_options["acceptor"] = lo.copy()
        tr.brightness_options = {"radius": 4, "bg_frame": 3, "mask": "circle"}

        for src, rois in [("single.tif", {"donor": d_roi, "acceptor": a_roi}),
                          (("split_don.tif", "split_acc.tif"),
                           {"donor": None, "acceptor": None})]:
            tr.rois = rois
            res = tr.locate_video(src, n_threads=2)
            self._check_locate_result(res, coords, 1300, 2000, 0, 1300, 200)

    def test_locate_all(self, tmp_path, channel_transform, coords):
        d_roi = roi.ROI((0, 0), size=(140, 150))
        a_roi = roi.ROI((170, 0), size=d_roi.size)

        img_shape = (150, 320)
        imgs = [sim.simulate_gauss(img_shape[::-1], c + a_roi.top_left,
                                   1300 if i % 2 else 2000,
                                   1, mass=True, engine="python") + 200
                for i, c in enumerate(coords[1])]
        for im, c in zip(imgs[::2], coords[0][::2]):
            im += sim.simulate_gauss(img_shape[::-1], c + d_roi.top_left, 1300,
                                     1, mass=True, engine="python")
        # change background so datasets are not identical
        imgs2 = [i + 100 for i in imgs]
        for n, ims in enumerate([imgs, imgs2]):
            imageio.mimwrite(tmp_path / f"single{n+1}.tif", ims)
            imageio.mimwrite(tmp_path / f"split{n+1}_don.tif",
                             [d_roi(i) for i in ims])
            imageio.mimwrite(tmp_path / f"split{n+1}_acc.tif",
                             [a_roi(i) for i in ims])

        tr = base.Tracker("da", data_dir=tmp_path)
        tr.registrator.parameters1 = channel_transform
        tr.registrator.parameters2 = np.linalg.inv(channel_transform)
        lo = {"algorithm": "Crocker-Grier",
              "options": {"radius": 3, "signal_thresh": 400,
                          "mass_thresh": 1000}}
        tr.locate_options["donor"] = lo.copy()
        lo["options"]["signal_thresh"] = 120
        tr.locate_options["acceptor"] = lo.copy()
        tr.brightness_options = {"radius": 4, "bg_frame": 3, "mask": "circle"}

        for src, rois in [({0: "single1.tif", 1: "single2.tif"},
                           {"donor": d_roi, "acceptor": a_roi}),
                          ({0: ("split1_don.tif", "split1_acc.tif"),
                            1: ("split2_don.tif", "split2_acc.tif")},
                           {"donor": None, "acceptor": None})]:
            tr.rois = rois
            tr.sm_data = {}
            tr.sources = {"data1": src, "data2": {2: src[1], 3: src[0]}}
            tr.special_sources = {"donor-only": {0: src[0]},
                                  "donor-profile": src}

            cur_f = []
            cur_n = []
            total_n = []

            def progress_cb(f, cur, total):
                cur_f.append(f)
                cur_n.append(cur)
                total_n.append(total)

            tr.locate_all(progress_callback=progress_cb)

            assert cur_f == [src[0], src[1], src[1], src[0], src[0]]
            assert cur_n == list(range(len(cur_f)))
            assert total_n == [len(cur_f)] * len(cur_f)

            assert list(tr.sm_data.keys()) == ["data1", "data2"]
            assert sorted(tr.sm_data["data1"].keys()) == [0, 1]
            assert sorted(tr.sm_data["data2"].keys()) == [2, 3]
            assert list(tr.special_sm_data.keys()) == ["donor-only"]
            assert list(tr.special_sm_data["donor-only"].keys()) == [0]

            sm_desc = [("data1", 0, 200), ("data1", 1, 300), ("data2", 2, 300),
                       ("data2", 3, 200)]  # dataset key, file key, background
            special_sm_desc = [("donor-only", 0, 200)]
            for res, bg in itertools.chain(
                    ((tr.sm_data[k1][k2], bg) for k1, k2, bg in sm_desc),
                    ((tr.special_sm_data[k1][k2], bg)
                     for k1, k2, bg in special_sm_desc)):
                self._check_locate_result(res, coords, 1300, 2000, 0, 1300, bg)

    @pytest.fixture
    def loc_data(self):
        d_loc = pd.DataFrame({("acceptor", "x"): [0.0, 2, 0, 2, 0, 2],
                              ("acceptor", "y"): [0.0, 0, 2, 2, 4, 5],
                              ("fret", "frame"): [0, 0, 2, 2, 4, 6],
                              ("fret", "bla"): 111})
        d_loc["donor", "x"] = d_loc["acceptor", "x"]
        d_loc["donor", "y"] = d_loc["acceptor", "y"]
        a_loc = pd.DataFrame({("acceptor", "x"): [0.0, 0, 0, 2, 2, 2],
                              ("acceptor", "y"): [1.0, 3, 5, 1, 3, 4],
                              ("fret", "frame"): [1, 3, 5, 1, 3, 5]})
        a_loc["donor", "x"] = a_loc["acceptor", "x"]
        a_loc["donor", "y"] = a_loc["acceptor", "y"]

        return {"donor": d_loc.copy(), "acceptor": a_loc.copy()}

    def test_track_video(self, loc_data):
        tr = base.Tracker("da")

        tr.link_options = {"search_range": 1.5, "memory": 0}
        sm_data = {k: v.copy() for k, v in loc_data.items()}
        tr.track_video(sm_data)
        d_exp = loc_data["donor"].copy()
        d_exp["fret", "particle"] = [0, 1, 0, 1, 0, 2]
        a_exp = loc_data["acceptor"].copy()
        a_exp["fret", "particle"] = [0, 0, 0, 1, 1, 2]
        pd.testing.assert_frame_equal(sm_data["donor"], d_exp)
        pd.testing.assert_frame_equal(sm_data["acceptor"], a_exp)

        # Also run with different link options to make sure they get picked
        # up correctly
        tr.link_options = {"search_range": 1.5, "memory": 1}
        sm_data = {k: v.copy() for k, v in loc_data.items()}
        tr.track_video(sm_data)
        d_exp = loc_data["donor"].copy()
        d_exp["fret", "particle"] = [0, 1, 0, 1, 0, 1]
        a_exp = loc_data["acceptor"].copy()
        a_exp["fret", "particle"] = [0, 0, 0, 1, 1, 1]
        pd.testing.assert_frame_equal(sm_data["donor"], d_exp)
        pd.testing.assert_frame_equal(sm_data["acceptor"], a_exp)

    def test_track_all(self, loc_data):
        tr = base.Tracker("da")
        tr.link_options = {"search_range": 1.5, "memory": 0}
        tr.sources = {"data1": {0: "f1.tif", 1: "f2.tif"},
                      "data2": {3: "f3.tif"}}
        tr.special_sources = {"donor-only": {0: "f4.tif"}}
        tr.sm_data = {"data1": {i: copy.deepcopy(loc_data) for i in (0, 1)},
                      "data2": {3: copy.deepcopy(loc_data)}}
        # Modify data to make sure each dataset is analyzed separately
        tr.sm_data["data1"][1]["donor"].loc[0, ("donor", "x")] = -5
        tr.sm_data["data1"][1]["donor"].loc[0, ("acceptor", "x")] = -5
        tr.sm_data["data2"][3]["acceptor"].loc[0, ("donor", "x")] = -3
        tr.sm_data["data2"][3]["acceptor"].loc[0, ("acceptor", "x")] = -3
        tr.special_sm_data = {"donor-only": {0: copy.deepcopy(loc_data)}}
        tr.special_sm_data["donor-only"][0]["donor"].loc[
            5, ("fret", "frame")] = 4
        tr.special_sm_data["donor-only"][0]["donor"].loc[
            5, ("donor", "y")] = 4
        tr.special_sm_data["donor-only"][0]["donor"].loc[
            5, ("acceptor", "y")] = 4

        e_data = copy.deepcopy(tr.sm_data)
        se_data = copy.deepcopy(tr.special_sm_data)

        cur_f = []
        cur_n = []
        total_n = []

        def progress_cb(f, cur, total):
            cur_f.append(f)
            cur_n.append(cur)
            total_n.append(total)

        tr.track_all(progress_cb)

        assert cur_f == [f"f{i}.tif" for i in range(1, 5)]
        assert cur_n == list(range(len(cur_f)))
        assert total_n == [len(cur_f)] * len(cur_f)

        d_exp = e_data["data1"][0]["donor"]
        d_exp["fret", "particle"] = [0, 1, 0, 1, 0, 2]
        a_exp = e_data["data1"][0]["acceptor"]
        a_exp["fret", "particle"] = [0, 0, 0, 1, 1, 2]
        pd.testing.assert_frame_equal(tr.sm_data["data1"][0]["donor"], d_exp)
        pd.testing.assert_frame_equal(tr.sm_data["data1"][0]["acceptor"],
                                      a_exp)

        d_exp = e_data["data1"][1]["donor"]
        d_exp["fret", "particle"] = [0, 1, 2, 1, 2, 3]
        a_exp = e_data["data1"][1]["acceptor"]
        a_exp["fret", "particle"] = [2, 2, 2, 1, 1, 3]
        pd.testing.assert_frame_equal(tr.sm_data["data1"][1]["donor"], d_exp)
        pd.testing.assert_frame_equal(tr.sm_data["data1"][1]["acceptor"],
                                      a_exp)

        d_exp = e_data["data2"][3]["donor"]
        d_exp["fret", "particle"] = [0, 1, 3, 1, 3, 4]
        a_exp = e_data["data2"][3]["acceptor"]
        a_exp["fret", "particle"] = [2, 3, 3, 1, 1, 4]
        pd.testing.assert_frame_equal(tr.sm_data["data2"][3]["donor"], d_exp)
        pd.testing.assert_frame_equal(tr.sm_data["data2"][3]["acceptor"],
                                      a_exp)

        d_exp = se_data["donor-only"][0]["donor"]
        d_exp["fret", "particle"] = [0, 1, 0, 1, 0, 1]
        a_exp = se_data["donor-only"][0]["acceptor"]
        a_exp["fret", "particle"] = [0, 0, 0, 1, 1, 1]
        pd.testing.assert_frame_equal(
            tr.special_sm_data["donor-only"][0]["donor"], d_exp)
        pd.testing.assert_frame_equal(
            tr.special_sm_data["donor-only"][0]["acceptor"], a_exp)

    def test_interpolate_missing_video(self, loc_data, tmp_path):
        for ld in loc_data.values():
            ld["donor", "x"] += 30
            ld["acceptor", "x"] += 30
            ld["donor", "y"] += 25
            ld["acceptor", "y"] += 25
            ld["fret", "has_neighbor"] = 0
        loc_data["donor"]["fret", "particle"] = [0, 1, 0, 1, 0, 2]
        loc_data["acceptor"]["fret", "particle"] = [0, 0, 0, 1, 1, 2]

        d_ims = np.array([np.full((50, 75), i) for i in range(1, 7)])
        a_ims = np.array([np.full((50, 75), i) for i in range(2, 8)])
        d_ims[2, 27, 30] = 5
        a_ims[2, 27, 30] = 7
        d_ims[3, 28, 30] = 8
        a_ims[3, 28, 30] = 10
        da_ims = np.concatenate([d_ims, a_ims], axis=2)

        imageio.mimwrite(tmp_path / "d_ims.tif", d_ims)
        imageio.mimwrite(tmp_path / "a_ims.tif", a_ims)
        imageio.mimwrite(tmp_path / "da_ims.tif", da_ims)

        d_exp = loc_data["donor"].copy()
        d_exp["fret", "interp"] = 0
        d_int = d_exp.loc[[2]]
        d_int["donor", "bg"] = 3.0
        d_int["donor", "bg_dev"] = 0.0
        d_int["donor", "mass"] = 2.0
        d_int["donor", "signal"] = 2.0
        d_int["acceptor", "bg"] = 4.0
        d_int["acceptor", "bg_dev"] = 0.0
        d_int["acceptor", "mass"] = 3.0
        d_int["acceptor", "signal"] = 3.0
        d_int["fret", "has_neighbor"] = 1
        d_int["fret", "interp"] = 1
        d_int.drop(columns=("fret", "bla"), inplace=True)
        d_exp = pd.concat([d_exp.drop(index=2), d_int])
        d_exp.sort_values([("fret", "particle"), ("fret", "frame")],
                          ignore_index=True, inplace=True)
        d_exp.sort_index(axis=1, inplace=True)
        a_exp = loc_data["acceptor"].copy()
        a_exp["fret", "interp"] = 0
        a_int = a_exp.loc[[1]]
        a_int["donor", "bg"] = 4.0
        a_int["donor", "bg_dev"] = 0.0
        a_int["donor", "mass"] = 4.0
        a_int["donor", "signal"] = 4.0
        a_int["acceptor", "bg"] = 5.0
        a_int["acceptor", "bg_dev"] = 0.0
        a_int["acceptor", "mass"] = 5.0
        a_int["acceptor", "signal"] = 5.0
        a_int["fret", "has_neighbor"] = 1
        a_int["fret", "interp"] = 1
        a_exp = pd.concat([a_exp.drop(index=1), a_int])
        a_exp.sort_values([("fret", "particle"), ("fret", "frame")],
                          ignore_index=True, inplace=True)
        a_exp.sort_index(axis=1, inplace=True)

        loc_data["donor"].drop(index=2, inplace=True)
        loc_data["acceptor"].drop(index=1, inplace=True)

        tr = base.Tracker("da", tmp_path)
        tr.brightness_options = {"radius": 1, "bg_frame": 1}
        tr.neighbor_distance = 2

        res = copy.deepcopy(loc_data)
        tr.interpolate_missing_video(("d_ims.tif", "a_ims.tif"), res)
        pd.testing.assert_frame_equal(res["donor"].sort_index(axis=1), d_exp)
        pd.testing.assert_frame_equal(res["acceptor"].sort_index(axis=1),
                                      a_exp)

        tr.rois = {"donor": roi.ROI((0, 0), size=(75, 50)),
                   "acceptor": roi.ROI((75, 0), size=(75, 50))}
        res = copy.deepcopy(loc_data)
        tr.interpolate_missing_video("da_ims.tif", res)
        pd.testing.assert_frame_equal(res["donor"].sort_index(axis=1), d_exp)
        pd.testing.assert_frame_equal(res["acceptor"].sort_index(axis=1),
                                      a_exp)

    def test_interpolate_missing_all(self, loc_data, tmp_path):
        tr = base.Tracker("da", tmp_path)
        tr.brightness_options = {"radius": 1, "bg_frame": 1}
        tr.neighbor_distance = 2
        tr.sources = {"data1": {0: "f1.tif", 1: "f2.tif"},
                      "data2": {3: "f3.tif"}}
        tr.special_sources = {"donor-only": {0: "f4.tif"}}
        tr.rois = {"donor": roi.ROI((0, 0), size=(75, 50)),
                   "acceptor": roi.ROI((75, 0), size=(75, 50))}

        for ld in loc_data.values():
            ld["donor", "x"] += 30
            ld["acceptor", "x"] += 30
            ld["donor", "y"] += 25
            ld["acceptor", "y"] += 25
            ld["fret", "has_neighbor"] = 0
        loc_data["donor"]["fret", "particle"] = [0, 1, 0, 1, 0, 2]
        loc_data["acceptor"]["fret", "particle"] = [0, 0, 0, 1, 1, 2]

        d_exp = loc_data["donor"].copy()
        d_exp["fret", "interp"] = 0
        d_int = d_exp.loc[[2]]
        d_int["donor", "bg"] = 3.0
        d_int["acceptor", "bg"] = 4.0
        d_int.drop(columns=("fret", "bla"), inplace=True)
        a_exp = loc_data["acceptor"].copy()
        a_exp["fret", "interp"] = 0
        a_int = a_exp.loc[[1]]
        a_int["donor", "bg"] = 4.0
        a_int["acceptor", "bg"] = 5.0

        for df in d_int, a_int:
            for ch in "donor", "acceptor":
                for col in "bg_dev", "mass", "signal":
                    df[ch, col] = 0.0
            for col in "has_neighbor", "interp":
                df["fret", col] = 1

        d_exp = pd.concat([d_exp.drop(index=2), d_int])
        d_exp.sort_values([("fret", "particle"), ("fret", "frame")],
                          ignore_index=True, inplace=True)
        d_exp.sort_index(axis=1, inplace=True)
        a_exp = pd.concat([a_exp.drop(index=1), a_int])
        a_exp.sort_values([("fret", "particle"), ("fret", "frame")],
                          ignore_index=True, inplace=True)
        a_exp.sort_index(axis=1, inplace=True)

        loc_data["donor"].drop(index=2, inplace=True)
        loc_data["acceptor"].drop(index=1, inplace=True)

        for i in range(1, 5):
            d_ims = np.array([np.full((50, 75), i) for i in range(1, 7)])
            a_ims = np.array([np.full((50, 75), i) for i in range(2, 8)])
            d_ims[2, 27, 30] = 5 + i
            a_ims[2, 27, 30] = 7 + i
            d_ims[3, 28, 30] = 8 + i
            a_ims[3, 28, 30] = 10 + i
            da_ims = np.concatenate([d_ims, a_ims], axis=2)

            imageio.mimwrite(tmp_path / f"f{i}.tif", da_ims)

        tr.sm_data = {"data1": {i: copy.deepcopy(loc_data) for i in (0, 1)},
                      "data2": {3: copy.deepcopy(loc_data)}}
        tr.special_sm_data = {"donor-only": {0: copy.deepcopy(loc_data)}}

        cur_f = []
        cur_n = []
        total_n = []

        def progress_cb(f, cur, total):
            cur_f.append(f)
            cur_n.append(cur)
            total_n.append(total)

        tr.interpolate_missing_all(progress_cb)

        assert cur_f == [f"f{i}.tif" for i in range(1, 5)]
        assert cur_n == list(range(len(cur_f)))
        assert total_n == [len(cur_f)] * len(cur_f)

        for i, v in enumerate(itertools.chain(
                tr.sm_data["data1"].values(), tr.sm_data["data2"].values(),
                tr.special_sm_data["donor-only"].values())):

            d_exp.loc[1, [("donor", "mass"), ("donor", "signal")]] = 3 + i
            d_exp.loc[1, [("acceptor", "mass"),
                          ("acceptor", "signal")]] = 4 + i
            a_exp.loc[1, [("donor", "mass"), ("donor", "signal")]] = 5 + i
            a_exp.loc[1, [("acceptor", "mass"),
                          ("acceptor", "signal")]] = 6 + i

            pd.testing.assert_frame_equal(v["donor"].sort_index(axis=1),
                                          d_exp)
            pd.testing.assert_frame_equal(v["acceptor"].sort_index(axis=1),
                                          a_exp)

    def test_extract_segment_images(self, tmp_path):
        tr = base.Tracker("sxs", data_dir=tmp_path)
        tr.sources = {"data1": {0: "f1.tif", 1: "f2.tif"},
                      "data2": {3: "f3.tif"}}
        tr.rois = {"donor": roi.ROI((0, 0), (2, 2)),
                   "acceptor": roi.ROI((2, 0), (3, 2))}

        for i in range(1, 4):
            ims = np.array([np.full((2, 3), i * j) for j in range(10, 31, 10)])
            ims[:, :, 2] += 1
            imageio.mimwrite(tmp_path / f"f{i}.tif", ims)

        def check_results(actual, desired):
            assert actual.keys() == desired.keys()
            for k in actual.keys():
                a = actual[k]
                d = desired[k]

                assert a.keys() == d.keys()
                for k2 in a.keys():
                    np.testing.assert_equal(a[k2], d[k2])

        tr.extract_segment_images("s", "donor")
        exp = {"data1": {0: np.array([np.full((2, 2), 10),
                                      np.full((2, 2), 30)]),
                         1: np.array([np.full((2, 2), 20),
                                      np.full((2, 2), 60)])},
               "data2": {3: np.array([np.full((2, 2), 30),
                                      np.full((2, 2), 90)])}}
        check_results(tr.segment_images, exp)
        tr.extract_segment_images("s", "acceptor")
        exp = {"data1": {0: np.array([np.full((2, 1), 11),
                                      np.full((2, 1), 31)]),
                         1: np.array([np.full((2, 1), 21),
                                      np.full((2, 1), 61)])},
               "data2": {3: np.array([np.full((2, 1), 31),
                                      np.full((2, 1), 91)])}}
        check_results(tr.segment_images, exp)
        tr.extract_segment_images("x", "donor")
        exp = {"data1": {0: np.full((1, 2, 2), 20),
                         1: np.full((1, 2, 2), 40)},
               "data2": {3: np.full((1, 2, 2), 60)}}
        check_results(tr.segment_images, exp)
        tr.extract_segment_images("x", "acceptor")
        exp = {"data1": {0: np.full((1, 2, 1), 21),
                         1: np.full((1, 2, 1), 41)},
               "data2": {3: np.full((1, 2, 1), 61)}}
        check_results(tr.segment_images, exp)

    def test_make_flatfield(self, tmp_path):
        tr = base.Tracker("da", data_dir=tmp_path)
        tr.special_sources["donor-profile"] = {0: "d1.tif", 1: "d2.tif"}
        tr.special_sources["acceptor-profile"] = {0: "a1.tif", 1: "a2.tif"}
        tr.flatfield_options["bg"] = 15
        tr.flatfield_options["smooth_sigma"] = 0
        tr.rois = {"donor": roi.ROI((0, 0), size=(4, 4)),
                   "acceptor": roi.ROI((4, 0), size=(4, 4))}
        tr.registrator.parameters1 = np.array([[1, 0, 1],
                                               [0, 1, 0],
                                               [0, 0, 1]])
        tr.registrator.parameters2 = np.array([[1, 0, -1],
                                               [0, 1, 0],
                                               [0, 0, 1]])

        ims_d1 = np.empty((2, 4, 8), dtype=float)
        ims_d1[0, :, :2] = 1
        ims_d1[0, :, 2:4] = 2
        ims_d1[1, :, :2] = 2
        ims_d1[1, :, 2:4] = 1
        ims_d1[0, :2, 4:] = 1
        ims_d1[0, 2:, 4:] = 2
        ims_d1[1, :2, 4:] = 2
        ims_d1[1, 2:, 4:] = 1
        ims_d1 += 15
        imageio.mimwrite(tmp_path / "d1.tif", ims_d1)
        ims_d2 = ims_d1 + 1
        imageio.mimwrite(tmp_path / "d2.tif", ims_d2)
        imageio.mimwrite(tmp_path / "a1.tif", ims_d1[::-1, ...])
        imageio.mimwrite(tmp_path / "a2.tif", ims_d2[::-1, ...])

        exp_f0 = np.array([[(0.5 + 2/3) / 2] * 4] * 2 + [[1.0] * 4] * 2).T
        tr.make_flatfield("donor", frame=0)
        np.testing.assert_allclose(tr.flatfield["donor"].corr_img, exp_f0)
        tr.make_flatfield("donor", frame="all")
        np.testing.assert_allclose(tr.flatfield["donor"].corr_img,
                                   np.ones((4, 4)))
        tr.make_flatfield("donor", frame=[0])
        np.testing.assert_allclose(tr.flatfield["donor"].corr_img, exp_f0)
        tr.make_flatfield("donor", frame=[0, 1])
        np.testing.assert_allclose(tr.flatfield["donor"].corr_img,
                                   np.ones((4, 4)))
        # Acceptor emission is (here) the transposed of donor emission
        # Due to the image registration transform, the rightmost column is
        # filled with the mean of the mean values
        exp_f0_tr = exp_f0.T.copy()
        exp_f0_tr[:, -1] = (1.5 / 2 + 2.5 / 3) / 2
        tr.make_flatfield("donor", frame=0, emission="acceptor")
        np.testing.assert_allclose(tr.flatfield["donor"].corr_img, exp_f0_tr)
        exp_f0_a = exp_f0.T[::-1, :]
        tr.make_flatfield("acceptor", frame=0)
        np.testing.assert_allclose(tr.flatfield["acceptor"].corr_img, exp_f0_a)
        tr.make_flatfield("acceptor", frame="all")
        np.testing.assert_allclose(tr.flatfield["acceptor"].corr_img,
                                   np.ones((4, 4)))
        tr.make_flatfield("acceptor", frame=[0])
        np.testing.assert_allclose(tr.flatfield["acceptor"].corr_img, exp_f0_a)
        tr.make_flatfield("acceptor", frame=[0, 1])
        np.testing.assert_allclose(tr.flatfield["acceptor"].corr_img,
                                   np.ones((4, 4)))

    def test_make_flatfield_sm(self):
        tr = base.Tracker("da")
        tr.rois = {"donor": roi.ROI((5, 2), size=(25, 21)),
                   "acceptor": roi.ROI((7, 3), size=(25, 21))}

        x, y = np.mgrid[0.0:25.0, 0.0:21.0].reshape((2, -1))

        # Generate donor excitation data
        center_d = (12, 10)
        sigma_d = (3, 2)
        mass_d = funcs.gaussian_2d(x, y, amplitude=1, center=center_d,
                                   sigma=sigma_d)
        d0 = pd.DataFrame({"x": x, "y": y, "mass": mass_d})
        a0 = pd.DataFrame({"x": x, "y": y, "mass": 2 * mass_d})
        s0 = pd.concat({"donor": d0, "acceptor": a0}, axis=1)
        s0["fret", "has_neighbor"] = 0
        s1 = s0.copy()
        s0["fret", "frame"] = 0
        s1["fret", "frame"] = 2
        s1[[("donor", "mass"), ("acceptor", "mass")]] *= 2
        d1d = pd.concat([s0, s1], ignore_index=True)
        s0[[("donor", "mass"), ("acceptor", "mass")]] *= 2
        s1[[("donor", "mass"), ("acceptor", "mass")]] *= 3 / 2
        d2d = pd.concat([s0, s1], ignore_index=True)

        # Generate acceptor excitation data
        center_a = (10, 9)
        sigma_a = (4, 3)
        mass_a = funcs.gaussian_2d(x, y, amplitude=2, center=center_a,
                                   sigma=sigma_a)
        d0 = pd.DataFrame({"x": x, "y": y, "mass": mass_a})
        a0 = pd.DataFrame({"x": x, "y": y, "mass": 2 * mass_a})
        s0 = pd.concat({"donor": d0, "acceptor": a0}, axis=1)
        s0["fret", "has_neighbor"] = 0
        s1 = s0.copy()
        s0["fret", "frame"] = 1
        s1["fret", "frame"] = 3
        s1[[("donor", "mass"), ("acceptor", "mass")]] *= 2
        d1a = pd.concat([s0, s1], ignore_index=True)
        s0[[("donor", "mass"), ("acceptor", "mass")]] *= 2
        s1[[("donor", "mass"), ("acceptor", "mass")]] *= 3 / 2
        d2a = pd.concat([s0, s1], ignore_index=True)

        tr.sm_data["d1"] = {"donor": d1d, "acceptor": d1a}
        tr.sm_data["d2"] = {"donor": d2d, "acceptor": d2a}

        def check_fit_result(actual, desired):
            assert actual.keys() == desired.keys()
            for k in desired.keys():
                np.testing.assert_allclose(actual[k], desired[k], atol=1e-3)

        # d1, frame 0: d amp = 1, a amp = 2 ==> amp = 3
        # d2, frame 0: d amp = 2, a amp = 4 ==> amp = 6
        exp_d = {"amplitude": 4.5, "center": center_d, "sigma": sigma_d,
                 "rotation": 0, "offset": 0}
        tr.make_flatfield_sm("donor")
        check_fit_result(tr.flatfield["donor"].fit_result, exp_d)
        # d1, frame 2: d amp = 2, a amp = 4 ==> amp = 6
        # d2, frame 2: d amp = 3, a amp = 6 ==> amp = 9
        exp_d["amplitude"] = 7.5
        tr.make_flatfield_sm("donor", frame=2)
        check_fit_result(tr.flatfield["donor"].fit_result, exp_d)
        exp_d["amplitude"] = 3
        tr.make_flatfield_sm("donor", keys=["d1"])
        check_fit_result(tr.flatfield["donor"].fit_result, exp_d)
        # d1, frame 1: a amp = 4
        # d2, frame 1: a amp = 8
        exp_a = {"amplitude": 6.0, "center": center_a, "sigma": sigma_a,
                 "rotation": 0, "offset": 0}
        tr.make_flatfield_sm("acceptor")
        check_fit_result(tr.flatfield["acceptor"].fit_result, exp_a)
        # d1, frame 3: a amp = 8
        # d2, frame 3: a amp = 12
        exp_a["amplitude"] = 10
        tr.make_flatfield_sm("acceptor", frame=3)
        check_fit_result(tr.flatfield["acceptor"].fit_result, exp_a)
        exp_a["amplitude"] = 8
        tr.make_flatfield_sm("acceptor", keys=["d2"])
        check_fit_result(tr.flatfield["acceptor"].fit_result, exp_a)

    def test_save_load(self, tmp_path, channel_transform, loc_data):
        tr = base.Tracker("dda", data_dir=tmp_path)
        tr.rois = {"donor": roi.ROI([5, 2], bottom_right=[30, 23]),
                   "acceptor": roi.ROI([7, 3], bottom_right=[32, 24])}
        tr.registrator.parameters1 = channel_transform
        tr.registrator.parameters2 = np.linalg.inv(channel_transform)
        lo = {"algorithm": "Crocker-Grier",
              "options": {"radius": 3, "signal_thresh": 400,
                          "mass_thresh": 1000}}
        tr.locate_options["donor"] = lo.copy()
        lo["options"]["signal_thresh"] = 120
        tr.locate_options["acceptor"] = lo.copy()
        tr.brightness_options = {"radius": 4, "bg_frame": 3, "mask": "circle"}
        tr.link_options = {"search_range": 1.5, "memory": 0}
        tr.sources = {"data1": {0: "f1.tif", 1: "f2.tif"},
                      "data2": {3: "f3.tif"}}
        tr.special_sources = {"donor-only": {0: "f4.tif"}}
        tr.sm_data = {"data1": {i: copy.deepcopy(loc_data) for i in (0, 1)},
                      "data2": {3: copy.deepcopy(loc_data)}}
        tr.sm_data["data1"][1]["donor"].loc[0, ("donor", "x")] = -5
        tr.sm_data["data2"][3]["acceptor"].loc[0, ("donor", "x")] = -3
        tr.special_sm_data = {"donor-only": {0: copy.deepcopy(loc_data)}}
        tr.special_sm_data["donor-only"][0]["donor"].loc[
            5, ("fret", "frame")] = 4
        tr.flatfield_options["bg"] = 15
        tr.flatfield_options["smooth_sigma"] = 0
        tr.neighbor_distance = 2
        tr.flatfield["donor"] = flatfield.Corrector(
            np.arange(1, 31).reshape((5, -1)), gaussian_fit=False)
        tr.flatfield["acceptor"] = flatfield.Corrector(
            np.arange(1, 31)[::-1].reshape((5, -1)), gaussian_fit=False)
        tr.segment_images = {"data1": {0: np.array([np.full((2, 2), 10),
                                                    np.full((2, 2), 30)]),
                                       1: np.array([np.full((2, 2), 20),
                                                    np.full((2, 2), 60)])},
                             "data2": {3: np.array([np.full((2, 2), 30),
                                                    np.full((2, 2), 90)])}}

        check_attrs = (list(filter(lambda x: not x.startswith("_"),
                                   tr.__dict__.keys())) +
                       tr.trait_names())
        check_attrs.append("excitation_seq")

        with io.chdir(tmp_path):
            tr.save()
            tr_l = base.Tracker.load()

        check_attrs.remove("frame_selector")
        assert (tr_l.frame_selector.excitation_seq ==
                tr.frame_selector.excitation_seq)
        for a in ("sm_data", "special_sm_data"):
            check_attrs.remove(a)
            sm_l = getattr(tr_l, a)
            sm = getattr(tr, a)
            assert sm_l.keys() == sm.keys()
            for k in sm_l.keys():
                v_l = sm_l[k]
                v = sm[k]
                assert v_l.keys() == v.keys()
                for fid in v_l.keys():
                    pd.testing.assert_frame_equal(
                        v_l[fid]["donor"], v[fid]["donor"])
                    pd.testing.assert_frame_equal(
                        v_l[fid]["acceptor"], v[fid]["acceptor"])
        check_attrs.remove("flatfield")
        np.testing.assert_allclose(tr_l.flatfield["donor"].corr_img,
                                   tr.flatfield["donor"].corr_img)
        np.testing.assert_allclose(tr_l.flatfield["acceptor"].corr_img,
                                   tr.flatfield["acceptor"].corr_img)
        check_attrs.remove("segment_images")
        assert tr_l.segment_images.keys() == tr.segment_images.keys()
        for k in tr_l.segment_images.keys():
            v_l = tr_l.segment_images[k]
            v = tr.segment_images[k]
            assert v_l.keys() == v.keys()
            for fid in v_l.keys():
                np.testing.assert_allclose(v_l[fid], v[fid])
        for a in check_attrs:
            assert getattr(tr_l, a) == getattr(tr, a)


class TestBaseIntermolecularTracker(TestBaseTracker):
    @pytest.fixture
    def loc_data(self):
        da1 = pd.DataFrame({"x": [0.0] * 15, "y": [0.0] * 15})
        aa1 = pd.DataFrame({"x": [0.0] * 8,
                            "y": [2.0, 1.5, 1.0, 0.5, 2.0, 0.5, 2.0, 2.0]})
        aa2 = pd.DataFrame({"x": [0.0] * 5, "y": [1.5, 0.5, 0.5, 0.5, 1.5]})
        aa = pd.concat([aa1, aa2], ignore_index=True)
        don_loc = pd.concat({"donor": da1, "acceptor": da1}, axis=1)
        don_loc["fret", "frame"] = np.arange(0, 30, 2)
        acc_loc = pd.concat({"donor": aa, "acceptor": aa}, axis=1)
        acc_loc["fret", "frame"] = np.concatenate([np.arange(1, 17, 2),
                                                   np.arange(21, 31, 2)])

        return {"donor": don_loc, "acceptor": acc_loc}

    def test_track_video(self, loc_data):
        don_exp = loc_data["donor"].copy()
        don_exp["fret", "d_particle"] = 0
        don_exp["fret", "a_particle"] = ([-1] * 2 + [0] * 4 + [-1] * 5 +
                                         [1] * 3 + [-1])
        don_exp["fret", "particle"] = ([-1] * 2 + [0] * 4 + [-1] * 5 +
                                       [1] * 3 + [-1])
        acc_exp = loc_data["acceptor"].copy()
        acc_exp["fret", "d_particle"] = ([-1] * 2 + [0] * 4 + [-1] * 3 +
                                         [0] * 3 + [-1])
        acc_exp["fret", "a_particle"] = [0] * 8 + [1] * 5
        acc_exp["fret", "particle"] = ([-1] * 2 + [0] * 4 + [-1] * 3 +
                                       [1] * 3 + [-1])

        tr = base.IntermolecularTracker("da")
        tr.link_options = {"search_range": 2.0, "memory": 0}
        tr.codiffusion_options["max_dist"] = 1.0
        tr.track_video(loc_data)

        pd.testing.assert_frame_equal(loc_data["donor"], don_exp)
        pd.testing.assert_frame_equal(loc_data["acceptor"], acc_exp)

    def test_track_all(self, loc_data):
        tr = base.IntermolecularTracker("da")
        tr.link_options = {"search_range": 2.0, "memory": 0}
        tr.codiffusion_options["max_dist"] = 1.0
        tr.sources = {"data1": {0: "f1.tif", 1: "f2.tif"},
                      "data2": {3: "f3.tif"}}
        tr.special_sources = {"donor-only": {0: "f4.tif"}}
        tr.sm_data = {"data1": {i: copy.deepcopy(loc_data) for i in (0, 1)},
                      "data2": {3: copy.deepcopy(loc_data)}}
        # Modify data to make sure each dataset is analyzed separately
        tr.sm_data["data1"][1]["donor"].loc[2, ("donor", "x")] = -1
        tr.sm_data["data1"][1]["donor"].loc[2, ("acceptor", "x")] = -1
        tr.sm_data["data2"][3]["acceptor"].loc[5, ("donor", "x")] = -3
        tr.sm_data["data2"][3]["acceptor"].loc[5, ("acceptor", "x")] = -3
        tr.special_sm_data = {"donor-only": {0: copy.deepcopy(loc_data)}}
        tr.special_sm_data["donor-only"][0]["donor"].drop(index=9,
                                                          inplace=True)

        e_data = copy.deepcopy(tr.sm_data)
        se_data = copy.deepcopy(tr.special_sm_data)

        cur_f = []
        cur_n = []
        total_n = []

        def progress_cb(f, cur, total):
            cur_f.append(f)
            cur_n.append(cur)
            total_n.append(total)

        tr.track_all(progress_cb)

        assert cur_f == [f"f{i}.tif" for i in range(1, 5)]
        assert cur_n == list(range(len(cur_f)))
        assert total_n == [len(cur_f)] * len(cur_f)

        d_exp = e_data["data1"][0]["donor"]
        d_exp["fret", "d_particle"] = 0
        d_exp["fret", "a_particle"] = ([-1] * 2 + [0] * 4 + [-1] * 5 +
                                       [1] * 3 + [-1])
        d_exp["fret", "particle"] = ([-1] * 2 + [0] * 4 + [-1] * 5 +
                                     [1] * 3 + [-1])
        a_exp = e_data["data1"][0]["acceptor"]
        a_exp["fret", "d_particle"] = ([-1] * 2 + [0] * 4 + [-1] * 3 +
                                       [0] * 3 + [-1])
        a_exp["fret", "a_particle"] = [0] * 8 + [1] * 5
        a_exp["fret", "particle"] = ([-1] * 2 + [0] * 4 + [-1] * 3 +
                                     [1] * 3 + [-1])
        pd.testing.assert_frame_equal(tr.sm_data["data1"][0]["donor"], d_exp)
        pd.testing.assert_frame_equal(tr.sm_data["data1"][0]["acceptor"],
                                      a_exp)

        d_exp = e_data["data1"][1]["donor"]
        d_exp["fret", "d_particle"] = 0
        d_exp["fret", "a_particle"] = ([-1] * 3 + [0] * 3 + [-1] * 5 +
                                       [1] * 3 + [-1])
        d_exp["fret", "particle"] = ([-1] * 3 + [0] * 3 + [-1] * 5 +
                                     [1] * 3 + [-1])
        a_exp = e_data["data1"][1]["acceptor"]
        a_exp["fret", "d_particle"] = ([-1] * 3 + [0] * 3 + [-1] * 3 +
                                       [0] * 3 + [-1])
        a_exp["fret", "a_particle"] = [0] * 8 + [1] * 5
        a_exp["fret", "particle"] = ([-1] * 3 + [0] * 3 + [-1] * 3 +
                                     [1] * 3 + [-1])
        pd.testing.assert_frame_equal(tr.sm_data["data1"][1]["donor"], d_exp)
        pd.testing.assert_frame_equal(tr.sm_data["data1"][1]["acceptor"],
                                      a_exp)

        d_exp = e_data["data2"][3]["donor"]
        d_exp["fret", "d_particle"] = 0
        d_exp["fret", "a_particle"] = ([-1] * 2 + [0] * 2 + [-1] * 7 +
                                       [3] * 3 + [-1])
        d_exp["fret", "particle"] = ([-1] * 2 + [0] * 2 + [-1] * 7 +
                                     [1] * 3 + [-1])
        a_exp = e_data["data2"][3]["acceptor"]
        a_exp["fret", "d_particle"] = ([-1] * 2 + [0] * 2 + [-1] * 5 +
                                       [0] * 3 + [-1])
        a_exp["fret", "a_particle"] = [0] * 5 + [1] + [2] * 2 + [3] * 5
        a_exp["fret", "particle"] = ([-1] * 2 + [0] * 2 + [-1] * 5 +
                                     [1] * 3 + [-1])
        pd.testing.assert_frame_equal(tr.sm_data["data2"][3]["donor"], d_exp)
        pd.testing.assert_frame_equal(tr.sm_data["data2"][3]["acceptor"],
                                      a_exp)

        d_exp = se_data["donor-only"][0]["donor"]
        d_exp["fret", "d_particle"] = [0] * 9 + [1] * 5
        d_exp["fret", "a_particle"] = ([-1] * 2 + [0] * 4 + [-1] * 4 +
                                       [1] * 3 + [-1])
        d_exp["fret", "particle"] = ([-1] * 2 + [0] * 4 + [-1] * 4 +
                                     [1] * 3 + [-1])
        a_exp = se_data["donor-only"][0]["acceptor"]
        a_exp["fret", "d_particle"] = ([-1] * 2 + [0] * 4 + [-1] * 3 +
                                       [1] * 3 + [-1])
        a_exp["fret", "a_particle"] = [0] * 8 + [1] * 5
        a_exp["fret", "particle"] = ([-1] * 2 + [0] * 4 + [-1] * 3 +
                                     [1] * 3 + [-1])
        pd.testing.assert_frame_equal(
            tr.special_sm_data["donor-only"][0]["donor"], d_exp)
        pd.testing.assert_frame_equal(
            tr.special_sm_data["donor-only"][0]["acceptor"], a_exp)

    def test_interpolate_missing_video(self, loc_data, tmp_path):
        for ld in loc_data.values():
            ld["donor", "x"] += 30
            ld["acceptor", "x"] += 30
            ld["donor", "y"] += 25
            ld["acceptor", "y"] += 25
            ld["fret", "has_neighbor"] = 0
        loc_data["donor"]["fret", "d_particle"] = 0
        loc_data["donor"]["fret", "a_particle"] = (
            [-1] * 2 + [0] * 4 + [-1] * 5 + [1] * 3 + [-1])
        loc_data["donor"]["fret", "particle"] = (
            [-1] * 2 + [0] * 4 + [-1] * 5 + [1] * 3 + [-1])
        loc_data["acceptor"]["fret", "d_particle"] = (
            [-1] * 2 + [0] * 4 + [-1] * 3 + [0] * 3 + [-1])
        loc_data["acceptor"]["fret", "a_particle"] = [0] * 8 + [1] * 5
        loc_data["acceptor"]["fret", "particle"] = (
            [-1] * 2 + [0] * 4 + [-1] * 3 + [1] * 3 + [-1])

        d_ims = np.array([np.full((50, 75), i) for i in range(1, 31)])
        a_ims = np.array([np.full((50, 75), i) for i in range(2, 32)])
        d_ims[2, 27, 30] = 5
        a_ims[2, 27, 30] = 7
        d_ims[1, 25, 30] = 8
        a_ims[1, 25, 30] = 10
        da_ims = np.concatenate([d_ims, a_ims], axis=2)

        imageio.mimwrite(tmp_path / "d_ims.tif", d_ims)
        imageio.mimwrite(tmp_path / "a_ims.tif", a_ims)
        imageio.mimwrite(tmp_path / "da_ims.tif", da_ims)

        d_exp = loc_data["donor"].copy()
        d_exp["fret", "interp"] = 0
        d_int_f = pd.DataFrame({
            # Last entry is for the localization dropped below, others are for
            # non co-diffusing
            "frame": [2, 12, 14, 22, 28, 22],
            "a_particle": [0, 0, 0, 1, 1, -1],
            "d_particle": [-1] * 5 + [0],
            "particle": -1,
            "interp": 1,
            "has_neighbor": 1})
        d_int_d = pd.DataFrame({
            "x": 30.0,
            "y": [26.75, 26.25, 27.0, 26.0, 26.0, 25.0],
            "bg": [3, 13, 15, 23, 29, 23],
            "bg_dev": 0.0,
            "mass": [2.0] + [0.0] * 5,
            "signal": [2.0] + [0.0] * 5})
        d_int_a = d_int_d.copy()
        d_int_a["bg"] += 1
        d_int_a.loc[0, ["mass", "signal"]] += 1
        d_int = pd.concat({"fret": d_int_f, "donor": d_int_d,
                           "acceptor": d_int_a}, axis=1)
        d_exp = pd.concat([d_exp.drop(index=11), d_int])
        d_exp.sort_values([("fret", "particle"), ("fret", "frame"),
                           ("fret", "d_particle")],
                          ignore_index=True, inplace=True)
        d_exp.sort_index(axis=1, inplace=True)
        a_exp = loc_data["acceptor"].copy()
        a_exp["fret", "interp"] = 0
        a_int_f = pd.DataFrame({
            # Last entry is for the localization dropped below, others are for
            # non co-diffusing
            "frame": [1, 3, 13, 15, 17, 19, 21, 7],
            "a_particle": [-1] * 7 + [0],
            "d_particle": [0] * 8,
            "particle": [-1] * 7 + [0],
            "interp": 1,
            "has_neighbor": [1, 1, 1, 1, 0, 0, 1, 0]})
        a_int_d = pd.DataFrame({
            "x": 30.0,
            "y": [25.0] * 7 + [26.5],
            "bg": [2, 4, 14, 16, 18, 20, 22, 8],
            "bg_dev": 0.0,
            "mass": [6.0] + [0.0] * 7,
            "signal": [6.0] + [0.0] * 7})
        a_int_a = a_int_d.copy()
        a_int_a["bg"] += 1
        a_int_a.loc[0, ["mass", "signal"]] += 1
        a_int = pd.concat({"fret": a_int_f, "donor": a_int_d,
                           "acceptor": a_int_a}, axis=1)
        a_exp = pd.concat([a_exp.drop(index=3), a_int])
        a_exp.sort_values([("fret", "particle"), ("fret", "frame"),
                           ("fret", "a_particle")],
                          ignore_index=True, inplace=True)
        a_exp.sort_index(axis=1, inplace=True)

        # at the beginning of "fret", "particle" no. 1
        loc_data["donor"].drop(index=11, inplace=True)
        # somwhere in the middle of "fret", "particle" no. 0
        loc_data["acceptor"].drop(index=3, inplace=True)

        tr = base.IntermolecularTracker("da", tmp_path)
        tr.brightness_options = {"radius": 1, "bg_frame": 1}
        tr.codiffusion_options["max_dist"] = 1.0
        tr.neighbor_distance = 2

        res = copy.deepcopy(loc_data)
        tr.interpolate_missing_video(("d_ims.tif", "a_ims.tif"), res)
        pd.testing.assert_frame_equal(
            res["donor"].sort_values(
                [("fret", "particle"), ("fret", "frame"),
                 ("fret", "d_particle")], ignore_index=True
                ).sort_index(axis=1),
            d_exp)
        pd.testing.assert_frame_equal(
            res["acceptor"].sort_values(
                [("fret", "particle"), ("fret", "frame"),
                 ("fret", "a_particle")], ignore_index=True
                ).sort_index(axis=1),
            a_exp)

        tr.rois = {"donor": roi.ROI((0, 0), size=(75, 50)),
                   "acceptor": roi.ROI((75, 0), size=(75, 50))}
        res = copy.deepcopy(loc_data)
        tr.interpolate_missing_video("da_ims.tif", res)
        pd.testing.assert_frame_equal(
            res["donor"].sort_values(
                [("fret", "particle"), ("fret", "frame"),
                 ("fret", "d_particle")], ignore_index=True
                ).sort_index(axis=1),
            d_exp)
        pd.testing.assert_frame_equal(
            res["acceptor"].sort_values(
                [("fret", "particle"), ("fret", "frame"),
                 ("fret", "a_particle")], ignore_index=True
                ).sort_index(axis=1),
            a_exp)

    def test_interpolate_missing_all(self, loc_data):
        # This method works for the base class and has not been reimplemented
        pass
