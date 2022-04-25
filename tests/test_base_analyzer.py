# SPDX-FileCopyrightText: 2022 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import itertools

import pandas as pd
import numpy as np
import pytest
from sdt import changepoint, flatfield, multicolor

from smfret_analysis import base


class TestBaseAnalyzer:
    ana1_seq = np.array(["d", "a"])

    @pytest.fixture
    def ana1(self):
        """Analyzer used in some tests"""
        sz = 20

        # Two bleach steps in acceptor, none in donor
        loc1 = pd.DataFrame({"x": np.full(sz, 50.0), "y": np.full(sz, 70.0)})
        fret1 = pd.DataFrame({
            "frame": np.arange(sz),
            "d_mass": [4000.0, 0.0] * (sz // 2),
            "a_mass": [3000.0] * 6 + [1500.0] * 6 + [100.0] * 8,
            "d_seg": [0] * sz,
            "a_seg": [0] * 6 + [1] * 6 + [2] * 8,
            "particle": [0] * sz,
            "d_seg_mean": 4000.0})
        fret1["a_seg_mean"] = fret1["a_mass"]
        data1 = pd.concat([loc1, loc1, fret1], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, none in donor
        loc2 = loc1.copy()
        loc2[["x", "y"]] = [20.0, 10.0]
        fret2 = fret1.copy()
        fret2["a_mass"] = [1600.0] * 10 + [150.0] * 10
        fret2["a_seg"] = [0] * 10 + [1] * 10
        fret2["a_seg_mean"] = fret2["a_mass"]
        fret2["particle"] = 1
        data2 = pd.concat([loc2, loc2, fret2], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step to non-zero in acceptor, none in donor
        loc3 = loc1.copy()
        loc3[["x", "y"]] = [120.0, 30.0]
        fret3 = fret2.copy()
        fret3["a_mass"] = [3500.0] * 10 + [1500.0] * 10
        fret3["a_seg"] = [0] * 10 + [1] * 10
        fret3["a_seg_mean"] = fret3["a_mass"]
        fret3["particle"] = 2
        data3 = pd.concat([loc3, loc3, fret3], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, one in donor before acceptor
        loc4 = loc2.copy()
        loc4[["x", "y"]] = [50.0, 60.0]
        fret4 = fret2.copy()
        fret4["d_mass"] = [3000.0, 0.0] * 3 + [600.0, 0.0] * 7
        fret4["d_seg"] = [0] * 5 + [1] * 15
        fret4["d_seg_mean"] = [3000.0] * 5 + [600.0] * 15
        fret4["particle"] = 3
        data4 = pd.concat([loc4, loc4, fret4], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, one in donor after acceptor
        loc5 = loc4.copy()
        loc5[["x", "y"]] = [60.0, 50.0]
        fret5 = fret4.copy()
        fret5["d_mass"] = [3000.0, 0.0] * 7 + [600.0, 0.0] * 3
        fret5["d_seg"] = [0] * 13 + [1] * 7
        fret5["d_seg_mean"] = [3000.0] * 13 + [600.0] * 7
        fret5["particle"] = 4
        data5 = pd.concat([loc5, loc5, fret5], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, one in donor to non-zero
        loc6 = loc4.copy()
        loc6[["x", "y"]] = [90.0, 70.0]
        fret6 = fret4.copy()
        fret6["d_mass"] = [5000.0, 0.0] * 7 + [2000.0, 0.0] * 3
        fret6["d_seg"] = [0] * 13 + [1] * 7
        fret6["d_seg_mean"] = [5000.0] * 13 + [2000.0] * 7
        fret6["particle"] = 5
        data6 = pd.concat([loc6, loc6, fret6], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # One bleach step in acceptor, two in donor
        loc7 = loc4.copy()
        loc7[["x", "y"]] = [100.0, 70.0]
        fret7 = fret4.copy()
        fret7["d_mass"] = ([5000.0, 0.0] * 3 + [2000.0, 0.0] * 3 +
                           [400.0, 0.0] * 4)
        fret7["d_seg"] = [0] * 5 + [1] * 6 + [2] * 9
        fret7["d_seg_mean"] = [5000.0] * 5 + [2000.0] * 6 + [400.0] * 9
        fret7["particle"] = 6
        data7 = pd.concat([loc7, loc7, fret7], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # No bleach steps in either channel
        loc8 = loc1.copy()
        loc8[["x", "y"]] = [190.0, 70.0]
        fret8 = fret1.copy()
        fret8["a_mass"] = 2000.0
        fret8["a_seg"] = 0
        fret8["a_seg_mean"] = fret8["a_mass"]
        fret8["particle"] = 7
        data8 = pd.concat([loc8, loc8, fret8], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # No bleach steps in acceptor, one in donor
        loc9 = loc1.copy()
        loc9[["x", "y"]] = [190.0, 20.0]
        fret9 = fret8.copy()
        fret9["d_mass"] = [3000.0, 0.0] * 7 + [600.0, 0.0] * 3
        fret9["d_seg"] = [0] * 13 + [1] * 7
        fret9["d_seg_mean"] = [3000.0] * 13 + [600.0] * 7
        fret9["particle"] = 8
        data9 = pd.concat([loc9, loc9, fret9], axis=1,
                          keys=["donor", "acceptor", "fret"])

        # Changepoint detection failed
        loc10 = loc1.copy()
        loc10[["x", "y"]] = [190.0, 150.0]
        fret10 = fret9.copy()
        fret10["d_mass"] = [3000.0, 0.0] * 7 + [600.0, 0.0] * 3
        fret10["d_seg"] = -1
        fret10["d_seg_mean"] = np.NaN
        fret10["particle"] = 9
        data10 = pd.concat([loc10, loc10, fret10], axis=1,
                           keys=["donor", "acceptor", "fret"])

        fsel = multicolor.FrameSelector("".join(self.ana1_seq))

        def split_df(df):
            return {
                "donor": fsel.select(
                    df, "d", columns={"time": ("fret", "frame")}),
                "acceptor": fsel.select(
                    df, "a", columns={"time": ("fret", "frame")})}

        ret = base.Analyzer()
        ret.sm_data = {"k1": {0: split_df(pd.concat([data1, data2, data3],
                                                    ignore_index=True)),
                              1: split_df(pd.concat([data4, data5],
                                                    ignore_index=True))},
                       "k2": {2: split_df(pd.concat([data6, data7, data8],
                                                    ignore_index=True))}}
        ret.special_sm_data = {"multi-state": {
            0: split_df(pd.concat([data9, data10], ignore_index=True))}}
        ret.bleach_thresh = (800, 500)
        return ret

    @pytest.fixture
    def ana_query_part(self, ana1):
        """Analyzer for query_particles tests"""
        trc = ana1.sm_data["k1"][0]["donor"]
        d0 = trc[trc["fret", "particle"] == 0].copy().reset_index(drop=True)
        d0.loc[2, ("fret", "a_mass")] = -1
        d1 = trc[trc["fret", "particle"] == 1].copy().reset_index(drop=True)
        d1.loc[[3, 5], ("fret", "a_mass")] = -1
        d2 = trc[trc["fret", "particle"] == 2].copy().reset_index(drop=True)
        don = pd.concat([d0, d1, d2], ignore_index=True)
        acc = ana1.sm_data["k1"][0]["acceptor"]
        acc = acc[acc["fret", "particle"].isin([0, 1, 2])
                  ].reset_index(drop=True)
        don["fret", "bla"] = 0
        data1 = {"donor": don.copy(), "acceptor": acc}
        don["fret", "bla"] = 1
        data2 = {"donor": don.copy(), "acceptor": acc}
        don["fret", "bla"] = 2
        data3 = {"donor": don.copy(), "acceptor": acc}
        don["fret", "bla"] = 3
        data4 = {"donor": don.copy(), "acceptor": acc}

        ana1.sm_data = {"k1": {0: data1, 1: data2}, "k2": {2: data3}}
        ana1.special_sm_data = {"multi-state": {3: data4}}
        return ana1

    ana2_seq = np.array(["d", "d", "d", "a"])

    @pytest.fixture
    def ana2(self):
        """Analyzer used in some tests"""
        num_frames = 20
        seq_len = len(self.ana2_seq)
        fsel = multicolor.FrameSelector("".join(self.ana2_seq))
        frames = np.arange(seq_len, seq_len + num_frames)

        df = pd.DataFrame({("fret", "frame"): frames})
        df["donor", "mass"] = 1000.0
        df["acceptor", "mass"] = 1500.0
        df["fret", "particle"] = 0

        df2 = df.copy()
        df2["donor", "mass"] = 2000.0

        df3 = df.copy()
        df3["donor", "mass"] = 3000.0

        df4 = df.copy()
        df4["donor", "mass"] = 4000.0

        def split_df(df):
            return {
                "donor": fsel.select(
                    df, "d", columns={"time": ("fret", "frame")}),
                "acceptor": fsel.select(
                    df, "a", columns={"time": ("fret", "frame")})}

        ret = base.Analyzer()
        ret.sm_data = {"k1": {0: split_df(df), 1: split_df(df2)},
                       "k2": {2: split_df(df3)}}
        ret.special_sm_data = {"multi-state": {0: split_df(df4)}}
        return ret

    def _assert_smdata_equal(self, actual, desired):
        assert actual.keys() == desired.keys()
        for dk in actual.keys():
            a_set = actual[dk]
            d_set = desired[dk]
            assert a_set.keys() == d_set.keys()
            for fk in a_set.keys():
                a_ch = a_set[fk]
                d_ch = d_set[fk]
                assert a_ch.keys() == d_ch.keys()
                for ch in a_ch.keys():
                    pd.testing.assert_frame_equal(a_ch[ch], d_ch[ch])

    def _apply_smdata(self, func, data):
        for dset in data.values():
            for d in dset.values():
                for sub in d.values():
                    func(sub)

    def test_update_filter(self, ana2):
        trc = pd.concat([d["donor"] for d in
                         (ana2.sm_data["k1"][0], ana2.sm_data["k1"][1],
                          ana2.sm_data["k2"][2],
                          ana2.special_sm_data["multi-state"][0])],
                        ignore_index=True)
        n = len(trc)

        # First filtering
        flt = np.full(n, -1, dtype=np.intp)
        flt[[2, 4, 6]] = 0
        flt[[5, 9, 13]] = 1
        cnt = ana2._increment_reason_counter("filt")
        ana2._update_filter(trc, flt, "filt", cnt)
        assert ("filter", "filt") in trc
        np.testing.assert_array_equal(trc["filter", "filt"], flt)

        # Second filtering, same reason
        flt2 = np.full(n, -1, dtype=np.intp)
        flt2[[1, 2, 5]] = 0
        flt2[[4, 5, 7]] = 1
        cnt = ana2._increment_reason_counter("filt")
        ana2._update_filter(trc, flt2, "filt", cnt)
        flt[1] = 0
        flt[[4, 7]] = 2
        np.testing.assert_array_equal(trc["filter", "filt"], flt)

        # Third filtering, different reason
        flt3 = np.full(n, -1, dtype=np.intp)
        flt3[[11, 16]] = 1
        cnt = ana2._increment_reason_counter("other_filt")
        ana2._update_filter(trc, flt3, "other_filt", cnt)
        assert ("filter", "other_filt") in trc
        np.testing.assert_array_equal(trc["filter", "filt"], flt)
        np.testing.assert_array_equal(trc["filter", "other_filt"], flt3)

    def test_apply_filters(self, ana1):
        t = pd.concat([d["donor"] for d in (
                ana1.sm_data["k1"][0], ana1.sm_data["k1"][1],
                ana1.sm_data["k2"][2],
                ana1.special_sm_data["multi-state"][0])],
            ignore_index=True)
        f1 = np.zeros(len(t), dtype=bool)
        f1[::3] = True
        f1_neg = np.zeros_like(f1)
        f1_neg[[2, 10, 14]] = True
        f2 = np.zeros_like(f1)
        f2[1::3] = True

        t["filter", "f1"] = 0
        t.loc[f1, ("filter", "f1")] = 2
        t.loc[f1_neg, ("filter", "f1")] = -1
        t["filter", "f2"] = 0
        t.loc[f2, ("filter", "f2")] = 1

        r = base.Analyzer._apply_filters(t)
        pd.testing.assert_frame_equal(r, t[~(f1 | f1_neg | f2)])
        r = base.Analyzer._apply_filters(t, type="mask")
        np.testing.assert_array_equal(r, ~(f1 | f1_neg | f2))
        r = base.Analyzer._apply_filters(t, include_negative=True)
        pd.testing.assert_frame_equal(r, t[~(f1 | f2)])
        r = base.Analyzer._apply_filters(t, include_negative=True, type="mask")
        np.testing.assert_array_equal(r, ~(f1 | f2))
        r = base.Analyzer._apply_filters(t, ignore="f1")
        pd.testing.assert_frame_equal(r, t[~f2])
        r = base.Analyzer._apply_filters(t, ignore="f1", type="mask")
        np.testing.assert_array_equal(r, ~f2)
        r = base.Analyzer._apply_filters(t, ignore="f2")
        pd.testing.assert_frame_equal(r, t[~(f1 | f1_neg)])
        r = base.Analyzer._apply_filters(t, ignore="f2", type="mask")
        np.testing.assert_array_equal(r, ~(f1 | f1_neg))
        r = base.Analyzer._apply_filters(t, include_negative=True, ignore="f2")
        pd.testing.assert_frame_equal(r, t[~f1])
        r = base.Analyzer._apply_filters(t, include_negative=True, ignore="f2",
                                         type="mask")
        np.testing.assert_array_equal(r, ~f1)

        hn = np.zeros(len(t), dtype=bool)
        hn[2::6] = 1
        t["fret", "has_neighbor"] = hn
        r = base.Analyzer._apply_filters(t, skip_neighbors=False)
        pd.testing.assert_frame_equal(r, t[~(f1 | f1_neg | f2)])
        r = base.Analyzer._apply_filters(t, skip_neighbors=True)
        pd.testing.assert_frame_equal(r, t[~(f1 | f1_neg | f2 | hn)])

    def test_reset_filters(self, ana1):
        t = copy.deepcopy(ana1.sm_data)
        st = copy.deepcopy(ana1.special_sm_data)

        # Reset without "filter" columns
        ana1.sm_data = copy.deepcopy(t)
        ana1.special_sm_data = copy.deepcopy(st)
        ana1.reset_filters()
        self._assert_smdata_equal(ana1.sm_data, t)
        self._assert_smdata_equal(ana1.special_sm_data, st)

        def add_filters(df):
            df["filter", "f1"] = 0
            df["filter", "f2"] = -1
            df["filter", "f3"] = 1

        ft = copy.deepcopy(t)
        sft = copy.deepcopy(st)
        self._apply_smdata(add_filters, ft)
        self._apply_smdata(add_filters, sft)

        # Reset all
        ana1.sm_data = copy.deepcopy(ft)
        ana1.special_sm_data = copy.deepcopy(sft)
        ana1.reset_filters()
        self._assert_smdata_equal(ana1.sm_data, t)
        self._assert_smdata_equal(ana1.special_sm_data, st)

        # Keep single
        def add_f2(df):
            df["filter", "f2"] = -1

        des = copy.deepcopy(t)
        sdes = copy.deepcopy(st)
        self._apply_smdata(add_f2, des)
        self._apply_smdata(add_f2, sdes)

        for k in "f2", ["f2"]:
            ana1.sm_data = copy.deepcopy(ft)
            ana1.special_sm_data = copy.deepcopy(sft)
            ana1.reset_filters(keep=k)
            self._assert_smdata_equal(ana1.sm_data, des)
            self._assert_smdata_equal(ana1.special_sm_data, sdes)

        # Keep multiple
        def add_f1_f3(df):
            df["filter", "f1"] = 0
            df["filter", "f3"] = 1

        des = copy.deepcopy(t)
        sdes = copy.deepcopy(st)
        self._apply_smdata(add_f1_f3, des)
        self._apply_smdata(add_f1_f3, sdes)

        ana1.sm_data = copy.deepcopy(ft)
        ana1.special_sm_data = copy.deepcopy(sft)
        ana1.reset_filters(keep=["f1", "f3"])
        self._assert_smdata_equal(ana1.sm_data, des)
        self._assert_smdata_equal(ana1.special_sm_data, sdes)

    def test_calc_apparent_values_eff(self, ana2):
        acc_mass = (np.arange(1, len(ana2.sm_data["k1"][0]["donor"]) + 1,
                              dtype=float)) * 1000

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                d = d["donor"]
                d["acceptor", "mass"] = acc_mass
                # Filter should not matter for efficiency calculation
                d["filter", "test"] = 0
                d.loc[1, ("filter", "test")] = 1

        ana2.calc_apparent_values()

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                d = d["donor"]
                d_mass = d["donor", "mass"] + acc_mass
                eff = acc_mass / d_mass

                assert ("fret", "eff_app") in d
                assert ("fret", "d_mass") in d
                np.testing.assert_allclose(d["fret", "eff_app"], eff)
                np.testing.assert_allclose(d["fret", "d_mass"], d_mass)

    def test_calc_apparent_values_stoi_linear(self, ana2):
        ex_data = ana2.sm_data["k1"][0]
        s = min(e["fret", "frame"].min() for e in ex_data.values())
        e = max(e["fret", "frame"].max() for e in ex_data.values())
        linear_mass = pd.Series(np.arange(s, e + 1, dtype=float) * 100.0)
        # Extrapolate constant value
        ld = np.count_nonzero(self.ana2_seq == "d")
        linear_mass[:ld] = linear_mass[ld]

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                d = d["acceptor"]
                d["acceptor", "mass"] = linear_mass[d.index]

        ana2.calc_apparent_values(a_mass_interp="linear")

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                d = d["donor"]
                assert ("fret", "a_mass") in d
                assert ("fret", "stoi_app") in d
                lm = linear_mass[d.index]
                np.testing.assert_allclose(d["fret", "a_mass"], lm)
                stoi = ((d["donor", "mass"] + d["acceptor", "mass"]) /
                        (d["donor", "mass"] + d["acceptor", "mass"] + lm))
                np.testing.assert_allclose(d["fret", "stoi_app"], stoi)

    def test_calc_apparent_values_stoi_nearest(self, ana2):
        seq_len = len(self.ana2_seq)
        mass_acc1 = 1500
        a_direct1 = np.nonzero(self.ana2_seq == "a")[0]
        mass_acc2 = 2000
        a_direct2 = a_direct1 + len(self.ana2_seq)

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                for sub in d.values():
                    # only two excitation sequence repetitions
                    sub.drop(
                        index=sub.index[sub.index >= 2 * len(self.ana2_seq)],
                        inplace=True)
                d["acceptor"].loc[a_direct1, ("acceptor", "mass")] = mass_acc1
                d["acceptor"].loc[a_direct2, ("acceptor", "mass")] = mass_acc2

        ex_data = ana2.sm_data["k1"][0]
        s = min(e["fret", "frame"].min() for e in ex_data.values())
        near_mass = np.full(len(ex_data["donor"]), mass_acc1)

        last1 = s + a_direct1[-1]
        first2 = s + a_direct1[0] + seq_len
        near2 = (np.abs(ex_data["donor"]["fret", "frame"] - last1) >
                 np.abs(ex_data["donor"]["fret", "frame"] - first2))
        near2up = (np.abs(ex_data["donor"]["fret", "frame"] - last1) >=
                   np.abs(ex_data["donor"]["fret", "frame"] - first2))
        prev2 = ex_data["donor"]["fret", "frame"].to_numpy() >= first2
        next2 = ex_data["donor"]["fret", "frame"].to_numpy() > last1

        t = ana2.sm_data
        st = ana2.special_sm_data
        for n, meth in [(near2, "nearest"), (near2up, "nearest-up"),
                        (prev2, "previous"), (next2, "next")]:
            ana2.sm_data = copy.deepcopy(t)
            ana2.special_sm_data = copy.deepcopy(st)
            ana2.calc_apparent_values(a_mass_interp=meth)

            for dset in itertools.chain(ana2.sm_data.values(),
                                        ana2.special_sm_data.values()):
                for d in dset.values():
                    d = d["donor"]
                    dm = d["donor", "mass"]
                    am = d["acceptor", "mass"]
                    stoi = ((dm + am) / (dm + am + mass_acc1)).copy()
                    stoi2 = (dm + am) / (dm + am + mass_acc2)
                    stoi[n] = stoi2[n]
                    nm = near_mass.copy()
                    nm[n] = mass_acc2

                    assert ("fret", "a_mass") in d
                    assert ("fret", "stoi_app") in d
                    np.testing.assert_allclose(d["fret", "stoi_app"], stoi)
                    np.testing.assert_allclose(d["fret", "a_mass"], nm)

    def test_calc_apparent_values_stoi_single(self, ana2):
        mass_acc = 2000

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                for sub in d.values():
                    # drop everything after one frame after first direct
                    # acceptor excitation
                    sub.drop(
                        index=sub.index[sub.index >= len(self.ana2_seq) + 1],
                        inplace=True)
                d["acceptor"]["acceptor", "mass"] = mass_acc

        ana2.calc_apparent_values()

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                d = d["donor"]
                assert ("fret", "a_mass") in d
                assert ("fret", "stoi_app") in d
                np.testing.assert_allclose(d["fret", "a_mass"], mass_acc)
                stoi = ((d["donor", "mass"] + d["acceptor", "mass"]) /
                        (d["donor", "mass"] + d["acceptor", "mass"] +
                         mass_acc))
                np.testing.assert_allclose(d["fret", "stoi_app"], stoi)

    def test_calc_apparent_values_neighbor(self, ana2):
        sm_data = ana2.sm_data
        ssm_data = ana2.special_sm_data
        for dset in itertools.chain(sm_data.values(), ssm_data.values()):
            for d in dset.values():
                d["acceptor"]["fret", "has_neighbor"] = 0
                idx_2 = d["acceptor"].index[1]
                idx_3 = d["acceptor"].index[2]
                d["acceptor"].loc[idx_3, ("acceptor", "mass")] *= 2
                d["acceptor"].loc[idx_3, ("fret", "has_neighbor")] = 1

        # This will skip the double mass frame, resulting in constant a_mass
        ana2.sm_data = copy.deepcopy(sm_data)
        ana2.special_sm_data = copy.deepcopy(ssm_data)
        ana2.calc_apparent_values(a_mass_interp="next")

        a_mass = d["acceptor"].loc[idx_2, ("acceptor", "mass")]

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                d = d["donor"]
                assert ("fret", "a_mass") in d
                assert ("fret", "stoi_app") in d
                np.testing.assert_allclose(d["fret", "a_mass"], a_mass)
                stoi = ((d["donor", "mass"] + d["acceptor", "mass"]) /
                        (d["donor", "mass"] + d["acceptor", "mass"] +
                         a_mass))
                np.testing.assert_allclose(d["fret", "stoi_app"], stoi)

        ana2.sm_data = copy.deepcopy(sm_data)
        ana2.special_sm_data = copy.deepcopy(ssm_data)
        ana2.calc_apparent_values(a_mass_interp="next", skip_neighbors=False)

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                d = d["donor"]
                am = pd.Series(a_mass, index=d.index)
                am[(am.index > idx_2) & (am.index < idx_3)] *= 2
                assert ("fret", "a_mass") in d
                assert ("fret", "stoi_app") in d
                np.testing.assert_allclose(d["fret", "a_mass"], am)
                stoi = ((d["donor", "mass"] + d["acceptor", "mass"]) /
                        (d["donor", "mass"] + d["acceptor", "mass"] + am))
                np.testing.assert_allclose(d["fret", "stoi_app"], stoi)

    def test_calc_apparent_values_filter(self, ana2):
        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                d["acceptor"]["filter", "test"] = 0
                idx_2 = d["acceptor"].index[1]
                idx_3 = d["acceptor"].index[2]
                d["acceptor"].loc[idx_3, ("acceptor", "mass")] *= 2
                d["acceptor"].loc[idx_3, ("filter", "test")] = 1

        # This will skip the double mass frame, resulting in constant a_mass
        ana2.calc_apparent_values(a_mass_interp="linear")

        a_mass = d["acceptor"].loc[idx_2, ("acceptor", "mass")]

        for dset in itertools.chain(ana2.sm_data.values(),
                                    ana2.special_sm_data.values()):
            for d in dset.values():
                d = d["donor"]
                assert ("fret", "a_mass") in d
                assert ("fret", "stoi_app") in d
                np.testing.assert_allclose(d["fret", "a_mass"], a_mass)
                stoi = ((d["donor", "mass"] + d["acceptor", "mass"]) /
                        (d["donor", "mass"] + d["acceptor", "mass"] +
                         a_mass))
                np.testing.assert_allclose(d["fret", "stoi_app"], stoi)

    def test_mass_changepoints(self):
        # NaNs cause bogus changepoints using Pelt; if segment_a_mass
        # does not ignore acceptor frames, we should see that.
        a_mass = np.array([6000, 6001, 6005] * 5 +
                          [0, 4, 2] * 4 +
                          [5007, 5002, 5003] * 3)
        reps = [15, 12, 9]
        segs = np.repeat([0, 1, 2], reps)
        frames = np.concatenate([np.arange(3) + 5 * i
                                 for i in range(sum(reps) // 3)])
        fd = pd.DataFrame({("fret", "a_mass"): a_mass,
                           ("fret", "frame"): frames})
        # Add NaN to test filtering
        fd.loc[0, ("fret", "a_mass")] = np.NaN
        # multiple particles
        fd["fret", "particle"] = 0
        fd2 = fd.copy()
        fd2["fret", "particle"] = 1
        fd3 = fd.copy()
        fd3["fret", "particle"] = 2

        fret_data = pd.concat([fd, fd2, fd3], ignore_index=True)
        # shuffle
        fret_data = pd.concat([fret_data.iloc[::2], fret_data.iloc[1::2]])

        # set filters
        fret_data["filter", "f1"] = 0
        fret_data.loc[0, ("filter", "f1")] = 1  # filter NaN
        fret_data["filter", "f2"] = 0
        # filter all of a particle
        fret_data.loc[fret_data["fret", "particle"] == 2, ("filter", "f2")] = 1

        fret_data["bla", "blub"] = 1
        fret_data2 = fret_data.copy()
        fret_data2["bla", "blub"] = 2
        fret_data3 = fret_data.copy()
        fret_data3["bla", "blub"] = 3
        fret_data4 = fret_data.copy()
        fret_data4["bla", "blub"] = 4

        cp_det = changepoint.Pelt("l2", min_size=1, jump=1, engine="python")

        ana = base.Analyzer(cp_detector=cp_det)
        ana.sm_data = {"k1": {0: {"donor": fret_data},
                              1: {"donor": fret_data2}},
                       "k2": {2: {"donor": fret_data3}}}
        ana.special_sm_data = {"multi-state": {3: {"donor": fret_data4}}}

        def check_result(tracks, stats, stat_results):
            for dset in tracks.values():
                for i, d in dset.items():
                    sub = d["donor"]
                    assert ("bla", "blub") in sub
                    np.testing.assert_array_equal(sub["bla", "blub"], i + 1)
                    assert ("fret", "a_seg") in sub
                    pd.testing.assert_series_equal(
                        sub["fret", "a_seg"].sort_index(),
                        pd.Series(segs.tolist() + [-1] * len(fd) * 2),
                        check_names=False)
                    for s, r in zip(stats, stat_results):
                        assert ("fret", f"a_seg_{s}") in sub
                        pd.testing.assert_series_equal(
                            sub["fret", f"a_seg_{s}"].sort_index(),
                            pd.Series(r.tolist() + [np.NaN] * len(fd) * 2),
                            check_names=False)

        # Test stats
        mean1_data = np.repeat([6001.923076923077, 2.2, 5003.625],
                               reps)
        min1_data = np.repeat([6000, 0, 5002], reps)
        mean2_data = np.repeat([6002, 1.75, 5003.857142857143], reps)
        min2_data = np.repeat([6000, 0, 5002], reps)
        mean6_data = np.repeat([6002.25, np.NaN, 5004], reps)

        ana.mass_changepoints("acceptor", stats="mean", penalty=1e7)
        check_result(ana.sm_data, ["mean"], [mean1_data])
        check_result(ana.special_sm_data, ["mean"], [mean1_data])
        ana.mass_changepoints("acceptor", stats=np.mean, penalty=1e7)
        check_result(ana.sm_data, ["mean"], [mean1_data])
        check_result(ana.special_sm_data, ["mean"], [mean1_data])
        ana.mass_changepoints("acceptor", stats=np.mean, penalty=1e7,
                              stat_margin=2)
        check_result(ana.sm_data, ["mean"], [mean2_data])
        check_result(ana.special_sm_data, ["mean"], [mean2_data])
        ana.mass_changepoints("acceptor", stats=["min", np.mean], penalty=1e7)
        check_result(ana.sm_data, ["min", "mean"], [min1_data, mean1_data])
        check_result(ana.special_sm_data, ["min", "mean"],
                     [min1_data, mean1_data])
        ana.mass_changepoints("acceptor", stats=["min", np.mean], penalty=1e7,
                              stat_margin=2)
        check_result(ana.sm_data, ["min", "mean"], [min2_data, mean2_data])
        check_result(ana.special_sm_data, ["min", "mean"],
                     [min2_data, mean2_data])
        # Use large stat_margin to have no data for some segments
        ana.mass_changepoints("acceptor", stats="mean", penalty=1e7,
                              stat_margin=6)
        check_result(ana.sm_data, ["mean"], [mean6_data])
        check_result(ana.special_sm_data, ["mean"], [mean6_data])

    @pytest.mark.parametrize(
        "cond,particles",
        [("acceptor", [1, 3, 4]), ("donor", [3, 4, 8]),
         ("donor or acceptor", [1, 3, 4, 8]), ("no partial", [1, 3, 4, 7, 8])])
    def test_bleach_step(self, ana1, cond, particles):
        ana1.bleach_step(cond, stat="mean")

        for dset in itertools.chain(ana1.sm_data.values(),
                                    ana1.special_sm_data.values()):
            for d in dset.values():
                for sub in d.values():
                    assert ("filter", "bleach_step") in sub
                    exp = np.ones(len(sub), dtype=np.intp)
                    exp[sub["fret", "particle"].isin(particles)] = 0
                    np.testing.assert_array_equal(sub["filter", "bleach_step"],
                                                  exp)

    def test_eval(self, ana1):
        trc = pd.concat([d["donor"] for d in
                         (ana1.sm_data["k1"][0], ana1.sm_data["k1"][1],
                          ana1.sm_data["k2"][2],
                          ana1.special_sm_data["multi-state"][0])],
                        ignore_index=True)
        d = trc.copy()

        # Simple test
        res = ana1._eval(trc,
                         "(fret_particle == 1 or acceptor_x == 120) and "
                         "fret_frame > 3")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["fret", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        pd.testing.assert_frame_equal(trc, d)  # data must not change

        # Test expression with error
        with pytest.raises(Exception):
            ana1._eval(trc, "fret_bla == 0")
        pd.testing.assert_frame_equal(trc, d)  # data must not change

        # Test `mi_sep` argument
        res = ana1._eval(trc,
                         "(fret__particle == 1 or acceptor__x == 120) and "
                         "fret__frame > 3",
                         mi_sep="__")
        exp = (((d["fret", "particle"] == 1) | (d["acceptor", "x"] == 120)) &
               (d["fret", "frame"] > 3))
        np.testing.assert_array_equal(res, exp)
        pd.testing.assert_frame_equal(trc, d)  # data must not change

    @staticmethod
    def _check_filter_result(actual_sm, actual_ssm, desired_sm, desired_ssm,
                             desired_transform):
        for dset_a, dset_d in zip(
                itertools.chain(actual_sm.values(), actual_ssm.values()),
                itertools.chain(desired_sm.values(), desired_ssm.values())):
            for d_a, d_d in zip(dset_a.values(), dset_d.values()):
                for sub_a, sub_d in zip(d_a.values(), d_d.values()):
                    desired_transform(sub_d)
                    pd.testing.assert_frame_equal(sub_a, sub_d)

    def test_query(self, ana1):
        sm = copy.deepcopy(ana1.sm_data)
        ssm = copy.deepcopy(ana1.special_sm_data)

        # First query
        ana1.query("(fret_particle == 1 or acceptor_x == 120) and "
                   "fret_frame > 3", reason="q1")

        def q1(d):
            m1 = (((d["fret", "particle"] == 1) |
                   (d["acceptor", "x"] == 120)) &
                  (d["fret", "frame"] > 3))
            m1 = (~m1).astype(np.intp)
            d["filter", "q1"] = m1

        self._check_filter_result(ana1.sm_data, ana1.special_sm_data,
                                  sm, ssm, q1)

        # Second query
        # Make sure that previously filtered entries don't get un-filtered
        ana1.query("fret_frame > 5", reason="q1")

        def q2(d):
            m2 = d["fret", "frame"] > 5
            m2 = (~m2).astype(np.intp) * 2
            old_f = d["filter", "q1"] > 0
            m2[old_f] = d.loc[old_f, ("filter", "q1")]
            d["filter", "q1"] = m2

        self._check_filter_result(ana1.sm_data, ana1.special_sm_data,
                                  sm, ssm, q2)

        # Third query
        # Different reason, should be independent
        ana1.query("fret_frame > 7", reason="q3")

        def q3(d):
            m3 = d["fret", "frame"] > 7
            m3 = (~m3).astype(np.intp)
            d["filter", "q3"] = m3

        self._check_filter_result(ana1.sm_data, ana1.special_sm_data,
                                  sm, ssm, q3)

    def test_query_error(self, ana1):
        sm = copy.deepcopy(ana1.sm_data)
        ssm = copy.deepcopy(ana1.special_sm_data)

        with pytest.raises(Exception):
            ana1.query("fret_bla == 0", reason="err")

        self._check_filter_result(ana1.sm_data, ana1.special_sm_data,
                                  sm, ssm, lambda x: None)

    def test_query_particles(self, ana_query_part):
        for d_set in itertools.chain(ana_query_part.sm_data.values(),
                                     ana_query_part.special_sm_data.values()):
            for d in d_set.values():
                dd = d["donor"]
                dd = dd[dd["fret", "particle"] == 1].copy()
                dd["fret", "particle"] = 4
                dd["fret", "d_mass"] = 3000
                d["donor"] = pd.concat([d["donor"], dd], ignore_index=True)
                da = d["acceptor"]
                da = da[da["fret", "particle"] == 1].copy()
                da["fret", "particle"] = 4
                d["acceptor"] = pd.concat([d["acceptor"], da],
                                          ignore_index=True)

        sm = copy.deepcopy(ana_query_part.sm_data)
        ssm = copy.deepcopy(ana_query_part.special_sm_data)

        # First query
        def q1(d):
            d["filter", "qry"] = (~d["fret", "particle"].isin([1, 4])
                                  ).astype(np.intp)

        ana_query_part.query_particles("fret_a_mass < 0", min_abs=2,
                                       reason="qry")
        self._check_filter_result(ana_query_part.sm_data,
                                  ana_query_part.special_sm_data,
                                  sm, ssm, q1)

        # Second query
        # Make sure that previously filtered particles don't get un-filtered
        def q2(d):
            d.loc[d["fret", "particle"] == 4, ("filter", "qry")] = 2

        ana_query_part.query_particles("fret_d_mass > 3500", min_abs=2,
                                       reason="qry")
        self._check_filter_result(ana_query_part.sm_data,
                                  ana_query_part.special_sm_data,
                                  sm, ssm, q2)

    def test_query_particles_pre_filtered(self, ana_query_part):
        for d_set in itertools.chain(ana_query_part.sm_data.values(),
                                     ana_query_part.special_sm_data.values()):
            for d in d_set.values():
                t = d["donor"]
                t["filter", "f1"] = 0
                t["filter", "f2"] = 0
                t.loc[t["fret", "particle"] == 2, ("filter", "f1")] = 1
                t.loc[[6, 7, 8], ("filter", "f2")] = 1

        sm = copy.deepcopy(ana_query_part.sm_data)
        ssm = copy.deepcopy(ana_query_part.special_sm_data)

        def q(d):
            d["filter", "f3"] = 0
            d.loc[d["fret", "particle"] == 0, ("filter", "f3")] = 1
            d.loc[d["fret", "particle"] == 2, ("filter", "f3")] = -1

        ana_query_part.query_particles("fret_a_mass < 200", min_abs=3,
                                       reason="f3")
        self._check_filter_result(ana_query_part.sm_data,
                                  ana_query_part.special_sm_data,
                                  sm, ssm, q)

    def test_query_particles_neg_min_abs(self, ana_query_part):
        sm = copy.deepcopy(ana_query_part.sm_data)
        ssm = copy.deepcopy(ana_query_part.special_sm_data)

        def q(d):
            d["filter", "qry"] = (~d["fret", "particle"].isin([0, 2])
                                  ).astype(np.intp)

        ana_query_part.query_particles("fret_a_mass > 0", min_abs=-1,
                                       reason="qry")
        self._check_filter_result(ana_query_part.sm_data,
                                  ana_query_part.special_sm_data,
                                  sm, ssm, q)

    def test_query_particles_zero_min_abs(self, ana_query_part):
        sm = copy.deepcopy(ana_query_part.sm_data)
        ssm = copy.deepcopy(ana_query_part.special_sm_data)

        def q(d):
            d["filter", "qry"] = (d["fret", "particle"] != 2).astype(np.intp)

        ana_query_part.query_particles("fret_a_mass > 0", min_abs=0,
                                       reason="qry")
        self._check_filter_result(ana_query_part.sm_data,
                                  ana_query_part.special_sm_data,
                                  sm, ssm, q)

    def test_query_particles_min_rel(self, ana_query_part):
        sm = copy.deepcopy(ana_query_part.sm_data)
        ssm = copy.deepcopy(ana_query_part.special_sm_data)

        def q(d):
            d["filter", "qry"] = (d["fret", "particle"] != 2).astype(np.intp)

        ana_query_part.query_particles("fret_a_mass > 1500", min_rel=0.49,
                                       reason="qry")
        self._check_filter_result(ana_query_part.sm_data,
                                  ana_query_part.special_sm_data,
                                  sm, ssm, q)

    def test_image_mask(self, ana1):
        sm = copy.deepcopy(ana1.sm_data)
        ssm = copy.deepcopy(ana1.special_sm_data)

        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 30:60] = True

        def q1(d):
            d["filter", "img_mask"] = (~d["fret", "particle"].isin([0, 3])
                                       ).astype(np.intp)
        ana1.image_mask(mask, "donor", reason="img_mask")
        self._check_filter_result(ana1.sm_data, ana1.special_sm_data,
                                  sm, ssm, q1)

        # Make sure data does not get un-filtered in second call
        mask2 = np.ones_like(mask, dtype=bool)
        ana1.image_mask(mask2, "donor", reason="img_mask")
        self._check_filter_result(ana1.sm_data, ana1.special_sm_data,
                                  sm, ssm, q1)

    def test_image_mask_list(self, ana1):
        mask = np.zeros((200, 200), dtype=bool)
        mask[50:100, 30:60] = True
        mask_list = {"k1": {0: [{"mask": mask, "start": 1, "end": 7},
                                {"mask": ~mask, "start": 10}],
                            1: [{"mask": mask, "start": 1, "end": 7},
                                {"mask": ~mask, "start": 10}]},
                     "k2": {2: [{"mask": np.zeros_like(mask)}]}}

        k2_3 = copy.deepcopy(ana1.sm_data["k2"][2])
        for sub in k2_3.values():
            sub["fret", "particle"] += 50
        ana1.sm_data["k2"][3] = k2_3

        sm = copy.deepcopy(ana1.sm_data)
        ssm = copy.deepcopy(ana1.special_sm_data)

        def q(d):
            flt = np.full(len(d), -1)
            flt[(d["fret", "particle"].isin([0, 3]) &
                 (d["fret", "frame"] >= 1) & (d["fret", "frame"] < 7)) |
                (~d["fret", "particle"].isin([0, 3]) &
                 (d["fret", "frame"] >= 10) &
                 (d["fret", "particle"] < 50))] = 0
            flt[(~d["fret", "particle"].isin([0, 3]) &
                 (d["fret", "frame"] >= 1) & (d["fret", "frame"] < 7) &
                 (d["fret", "particle"] < 50)) |
                (d["fret", "particle"].isin([0, 3]) &
                 (d["fret", "frame"] >= 10))] = 1
            flt[(d["fret", "particle"] >= 5) &
                (d["fret", "particle"] < 50)] = 1
            d["filter", "img_mask"] = flt

        ana1.image_mask(mask_list, "donor", reason="img_mask")
        self._check_filter_result(ana1.sm_data, {}, sm, {}, q)
        self._check_filter_result({}, ana1.special_sm_data, {}, ssm,
                                  lambda x: None)

    def test_flatfield_correction(self):
        """fret.SmFRETAnalyzer.flatfield_correction"""
        img1 = np.hstack([np.full((4, 2), 1), np.full((4, 2), 2)])
        corr1 = flatfield.Corrector([img1], gaussian_fit=False)
        img2 = np.hstack([np.full((4, 2), 1), np.full((4, 2), 3)]).T
        corr2 = flatfield.Corrector([img2], gaussian_fit=False)

        d = np.array([[1.0, 1.0, 3.0, 3.0], [1.0, 3.0, 3.0, 1.0]]).T
        d = pd.DataFrame(d, columns=["x", "y"])
        d = pd.concat([d, d], axis=1, keys=["donor", "acceptor"])
        d["donor", "mass"] = [10.0, 20.0, 30.0, 40.0]
        d["donor", "signal"] = d["donor", "mass"] / 10
        d["acceptor", "mass"] = [20.0, 30.0, 40.0, 50.0]
        d["acceptor", "signal"] = d["acceptor", "mass"] / 5
        d["fret", "bla"] = 0
        k1_0 = {"donor": d.loc[::2].copy(), "acceptor": d.loc[1::2].copy()}
        d["fret", "bla"] = 1
        k1_1 = {"donor": d.loc[::2].copy(), "acceptor": d.loc[1::2].copy()}
        d["fret", "bla"] = 2
        k2_2 = {"donor": d.loc[::2].copy(), "acceptor": d.loc[1::2].copy()}
        d["fret", "bla"] = 3
        ms = {"donor": d.loc[::2].copy(), "acceptor": d.loc[1::2].copy()}

        ana = base.Analyzer()
        ana.flatfield = {"donor": corr1, "acceptor": corr2}
        ana.sm_data = {"k1": {0: k1_0, 1: k1_1}, "k2": {2: k2_2}}
        ana.special_sm_data = {"multi-state": {3: ms}}

        sm = copy.deepcopy(ana.sm_data)
        ssm = copy.deepcopy(ana.special_sm_data)

        for _ in range(2):
            # Run twice to ensure that flatfield corrections are not applied
            # on top of each other
            ana.flatfield_correction()

            for dset_a, dset_d in zip(
                    itertools.chain(ana.sm_data.values(),
                                    ana.special_sm_data.values()),
                    itertools.chain(sm.values(), ssm.values())):
                for (n, d_a), d_d in zip(dset_a.items(), dset_d.values()):
                    for sub_a, sub_d in zip(d_a.values(), d_d.values()):
                        for chan, col in itertools.product(
                                ["donor", "acceptor"], ["signal", "mass"]):
                            src = f"{col}_pre_flat"
                            assert (chan, src) in sub_a
                            np.testing.assert_allclose(sub_a[chan, src],
                                                       sub_d[chan, col])
                        np.testing.assert_allclose(sub_a["fret", "bla"], n)

                    np.testing.assert_allclose(
                        d_a["donor"]["donor", "mass"], [20, 30])
                    np.testing.assert_allclose(
                        d_a["acceptor"]["donor", "mass"], [20, 120])
                    np.testing.assert_allclose(
                        d_a["donor"]["donor", "signal"], [2, 3])
                    np.testing.assert_allclose(
                        d_a["acceptor"]["donor", "signal"], [2, 12])
                    np.testing.assert_allclose(
                        d_a["donor"]["acceptor", "mass"], [40, 40])
                    np.testing.assert_allclose(
                        d_a["acceptor"]["acceptor", "mass"], [30, 150])
                    np.testing.assert_allclose(
                        d_a["donor"]["acceptor", "signal"], [8, 8])
                    np.testing.assert_allclose(
                        d_a["acceptor"]["acceptor", "signal"], [6, 30])

    def test_calc_leakage(self):
        d = {("fret", "eff_app"): [0.1, 0.1, 0.3, 0.1, 0.1, 0.25],
             ("fret", "has_neighbor"): [0, 0, 1, 0, 0, 0],
             ("fret", "frame"): [0, 1, 2, 3, 4, 5],
             ("filter", "test"): [0, 0, 0, 0, 0, 1]}
        d = pd.DataFrame(d)
        d2 = d.copy()
        d2["fret", "eff_app"] *= 3

        ana = base.Analyzer()
        ana.special_sm_data = {"donor-only": {0: {"donor": d,
                                                  "acceptor": d.iloc[:0]},
                                              1: {"donor": d2,
                                                  "acceptor": d2.iloc[:0]}}}
        ana.calc_leakage()

        assert ana.leakage == pytest.approx(0.25)

    def test_calc_direct_excitation(self):
        d = {("fret", "frame"): [0, 2, 4, 6],
             ("fret", "stoi_app"): [0.03, 0.03, 0.6, 0.5],
             ("fret", "has_neighbor"): [0, 0, 1, 0],
             ("filter", "test"): [0, 0, 0, 1]}
        d = pd.DataFrame(d)
        d2 = d.copy()
        d2["fret", "stoi_app"] /= 3

        ana = base.Analyzer()
        ana.special_sm_data = {"acceptor-only": {0: {"donor": d,
                                                     "acceptor": d.iloc[:0]},
                                                 1: {"donor": d2,
                                                     "acceptor": d2.iloc[:0]}}}
        ana.calc_direct_excitation()
        assert ana.direct_excitation == pytest.approx(0.02 / 0.98)

    def test_calc_detection_eff(self):
        d1 = {("donor", "mass"): [0, 0, 0, 4, np.NaN, 6, 6, 6, 6, 9000],
              ("acceptor", "mass"): [10, 12, 10, 12, 3000, 1, 1, 1, 1, 7000],
              ("fret", "has_neighbor"): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              ("fret", "a_seg"): [0] * 5 + [1] * 5,
              ("filter", "test"): 0}
        d1 = pd.DataFrame(d1)
        d1["fret", "particle"] = 0
        d1["fret", "frame"] = np.arange(len(d1))

        d2 = d1.copy()
        d2["fret", "particle"] = 1
        d2["fret", "a_seg"] = [0] * 2 + [1] * 8  # short pre

        d3 = d1.copy()
        d3["fret", "particle"] = 2
        d3["fret", "a_seg"] = [0] * 8 + [1] * 2  # short post

        d4 = d1.copy()
        d4["fret", "particle"] = 3
        d4["donor", "mass"] = [1] * 5 + [11] * 5
        d4.loc[4, ("acceptor", "mass")] = 11

        d5 = d1.copy()
        d5["fret", "particle"] = 4
        d5["donor", "mass"] = [np.NaN] * 5 + [10] * 5
        d5["acceptor", "mass"] = [10] * 5 + [np.NaN] * 5

        # Test filtering
        d6 = d1.copy()
        d6["fret", "particle"] = 5
        d6.loc[:5, ("donor", "mass")] = 1000
        d6["filter", "test"] = 1

        sm = {"k1": {0: {"donor": pd.concat([d1, d2], ignore_index=True)},
                     1: {"donor": pd.concat([d5, d6], ignore_index=True)}},
              "k2": {2: {"donor": pd.concat([d3, d4], ignore_index=True)}}}

        ana = base.Analyzer()
        ana.sm_data = sm

        des = {"k1": {0: pd.Series([10 / 6, np.NaN], index=[0, 1]),
                      1: pd.Series([np.NaN, np.NaN], index=[4, 5])},
               "k2": {2: pd.Series([np.NaN, 1.0], index=[2, 3])}}

        ana.calc_detection_eff("individual", min_seg_len=3)
        assert ana.detection_eff.keys() == des.keys()
        for k in des.keys():
            a = ana.detection_eff[k]
            d = des[k]
            assert a.keys() == d.keys()
            for k2 in d.keys():
                pd.testing.assert_series_equal(a[k2], d[k2])

        des["k1"][0][0] = 2

        ana.calc_detection_eff("individual", min_seg_len=3, seg_stat=np.mean)
        assert ana.detection_eff.keys() == des.keys()
        for k in des.keys():
            a = ana.detection_eff[k]
            d = des[k]
            assert a.keys() == d.keys()
            for k2 in d.keys():
                pd.testing.assert_series_equal(a[k2], d[k2])

        ana.calc_detection_eff("all", min_seg_len=3, seg_stat=np.mean)
        assert ana.detection_eff == pytest.approx(1.5)

        ana.calc_detection_eff("all", min_seg_len=3, seg_stat=np.mean)
        assert ana.detection_eff == pytest.approx(1.5)

        ana.calc_detection_eff("all", datasets="k1", min_seg_len=3,
                               seg_stat=np.mean)
        assert ana.detection_eff == pytest.approx(2)
        ana.calc_detection_eff("all", datasets=["k1"], min_seg_len=3,
                               seg_stat=np.mean)
        assert ana.detection_eff == pytest.approx(2)
        ana.calc_detection_eff("all", datasets=["k1", "k2"], min_seg_len=3,
                               seg_stat=np.mean)
        assert ana.detection_eff == pytest.approx(1.5)

    def test_calc_excitation_eff(self):
        sz1 = 10
        sz2 = 20
        i_da = np.array([700.] * sz1 + [0.] * sz2)
        i_dd = np.array([300.] * sz1 + [1000.] * sz2)
        i_aa = np.array([1000.] * sz1 + [0.] * sz2)
        a = np.array([0] * sz1 + [1] * sz2)
        data = pd.DataFrame({("donor", "mass"): i_dd,
                             ("acceptor", "mass"): i_da,
                             ("fret", "a_mass"): i_aa,
                             ("fret", "a_seg"): a,
                             ("fret", "d_seg"): 0,
                             ("fret", "particle"): 0,
                             ("filter", "test"): 0})
        data2 = data.copy()
        data2["fret", "particle"] = 1
        data2["donor", "mass"] = 10 * i_dd
        data2["filter", "test"] = 1

        ana = base.Analyzer()
        ana.sm_data["k1"] = {0: {"donor": data}, 1: {"donor": data2}}
        ana.leakage = 0.2
        ana.direct_excitation = 0.1
        ana.detection_eff = 0.5

        ana.calc_excitation_eff("k1")

        f_da = 700 - 300 * 0.2 - 1000 * 0.1
        f_dd = 300 * 0.5
        i_aa = 1000
        assert ana.excitation_eff == pytest.approx(i_aa / (f_dd + f_da))

    def test_calc_excitation_eff_components(self):
        sz1 = 10
        sz2 = 20
        i_da = np.array([700.] * sz1 + [0.] * sz2)
        i_dd = np.array([300.] * sz1 + [1000.] * sz2)
        i_aa = np.array([1000.] * sz1 + [0.] * sz2)
        a = np.array([0] * sz1 + [1] * sz2)
        data = pd.DataFrame({("donor", "mass"): i_dd,
                             ("acceptor", "mass"): i_da,
                             ("fret", "a_mass"): i_aa,
                             ("fret", "a_seg"): a,
                             ("fret", "d_seg"): 0,
                             ("fret", "particle"): 0})

        data2 = data.copy()
        data2["fret", "particle"] = 1
        data2["fret", "a_mass"] = 1e6

        # Construct eff_app and stoi_app such that particle 1 is removed
        rnd = np.random.RandomState(0)
        sz = sz1 + sz2
        c1 = rnd.normal((0.9, 0.5), 0.1, (sz, 2))
        c2 = rnd.normal((0.1, 0.8), 0.1, (sz, 2))
        data["fret", "eff_app"] = c1[:, 0]
        data["fret", "stoi_app"] = c1[:, 1]
        data2["fret", "eff_app"] = c2[:, 0]
        data2["fret", "stoi_app"] = c2[:, 1]

        ana = base.Analyzer()
        ana.sm_data["k1"] = {0: {"donor": data}, 1: {"donor": data2}}
        ana.leakage = 0.2
        ana.direct_excitation = 0.1
        ana.detection_eff = 0.5

        ana.calc_excitation_eff("k1", n_components=2)

        f_da = 700 - 300 * 0.2 - 1000 * 0.1
        f_dd = 300 * 0.5
        i_aa = 1000
        assert ana.excitation_eff == pytest.approx(i_aa / (f_dd + f_da))

    def test_calc_detection_excitation_effs(self):
        rs = np.random.RandomState(0)
        dd1 = rs.normal(2000, 30, 10000)  # 0.3 FRET eff
        da1 = rs.normal(1000, 30, 10000)
        aa1 = rs.normal(3000, 30, 10000)  # 0.5 stoi

        # First component
        d1 = pd.DataFrame({("donor", "frame"): np.arange(len(dd1)),
                           ("donor", "mass"): dd1,
                           ("acceptor", "mass"): da1,
                           ("fret", "a_mass"): aa1,
                           ("fret", "particle"): 0,
                           ("fret", "d_seg"): 0,
                           ("fret", "a_seg"): 0})
        # Second component
        d2 = d1.copy()
        d2["fret", "particle"] = 1
        d2["donor", "mass"] = da1
        d2["acceptor", "mass"] = dd1
        # Bogus component
        d3 = d1.copy()
        d3["fret", "particle"] = 2
        d3["donor", "mass"] /= 2

        leak = 0.05
        direct = 0.1
        det = 0.95
        exc = 1.2

        for t in d1, d2, d3:
            t["donor", "mass"] /= det
            t["fret", "a_mass"] *= exc
            t["acceptor", "mass"] += (leak * t["donor", "mass"] +
                                      direct * t["fret", "a_mass"])

        # Test using the two "good" components
        ana = base.Analyzer()
        ana.special_sm_data = {"multi-state": {0: {"donor": d1},
                                               1: {"donor": d2}}}
        ana.leakage = leak
        ana.direct_excitation = direct
        ana.calc_detection_excitation_effs(2)
        assert ana.detection_eff == pytest.approx(det, abs=0.001)
        assert ana.excitation_eff == pytest.approx(exc, abs=0.001)

        # Test using with all three "good" components, excluding the bad one
        ana = base.Analyzer()
        ana.sm_data = {"bla": {0: {"donor": pd.concat([d1, d3],
                                                      ignore_index=True)},
                               1: {"donor": d2}}}
        ana.leakage = leak
        ana.direct_excitation = direct
        ana.calc_detection_excitation_effs(3, [0, 2], dataset="bla")
        assert ana.detection_eff == pytest.approx(det, abs=0.001)
        assert ana.excitation_eff == pytest.approx(exc, abs=0.001)

    @pytest.mark.parametrize("gamma", ["all", "dataset", "individual"])
    def test_fret_correction(self, gamma):
        d = pd.DataFrame({("donor", "mass"): [1000, 0, 500, 0],
                          ("acceptor", "mass"): [2000, 3000, 2500, 2000],
                          ("fret", "a_mass"): [3000, 3000, 2000, 2000],
                          ("fret", "bla"): 0,
                          ("fret", "particle"): [0, 0, 1, 1]})
        d2 = d.copy()
        d2["fret", "bla"] = 1
        d3 = d.copy()
        d3["fret", "bla"] = 2
        sm = {"k1": {0: {"donor": d}, 1: {"donor": d2}},
              "k2": {2: {"donor": d3}}}

        ana = base.Analyzer()
        ana.sm_data = copy.deepcopy(sm)
        ana.leakage = 0.1
        ana.direct_excitation = 0.2
        ana.excitation_eff = 0.8

        f_da = np.array([1300, 2400, 2050, 1600], dtype=float)
        f_aa = np.array([3750, 3750, 2500, 2500], dtype=float)

        if gamma == "all":
            ana.detection_eff = 0.9
            f = np.array([900, 0, 450, 0], dtype=float)
            f_dd = {"k1": {0: f, 1: f}, "k2": {2: f}}
        elif gamma == "dataset":
            ana.detection_eff = {"k1": 0.9, "k2": 0.8}
            f1 = np.array([900, 0, 450, 0], dtype=float)
            f2 = np.array([800, 0, 400, 0], dtype=float)
            f_dd = {"k1": {0: f1, 1: f1}, "k2": {2: f2}}
        else:
            ana.detection_eff = {"k1": {0: pd.Series([0.9, 0.8]),
                                        1: pd.Series([0.8, 0.7])},
                                 "k2": {2: pd.Series([0.7, 0.6])}}
            f_dd = {"k1": {0: np.array([900.0, 0.0, 400.0, 0.0]),
                           1: np.array([800.0, 0.0, 350.0, 0.0])},
                    "k2": {2: np.array([700.0, 0.0, 300.0, 0.0])}}

        ana.fret_correction()

        assert ana.sm_data.keys() == sm.keys()
        for k, dset in ana.sm_data.items():
            assert dset.keys() == sm[k].keys()
            for fid, d in dset.items():
                t = d["donor"]
                for c in ("f_da", "f_dd", "f_aa", "eff", "stoi"):
                    assert ("fret", c) in t

                f = f_dd[k][fid]
                np.testing.assert_allclose(t["fret", "f_dd"], f)
                np.testing.assert_allclose(t["fret", "f_aa"], f_aa)
                np.testing.assert_allclose(t["fret", "eff"],
                                           f_da / (f_da + f))
                np.testing.assert_allclose(
                    t["fret", "stoi"], (f_da + f) / (f_da + f + f_aa))
                np.testing.assert_array_equal(t["fret", "bla"], fid)


def test_gaussian_mixture_split():
    """fret.gaussian_mixture_split"""
    rnd = np.random.RandomState(0)
    c1 = rnd.normal((0.1, 0.8), 0.1, (2000, 2))
    c2 = rnd.normal((0.9, 0.5), 0.1, (2000, 2))
    d = np.concatenate([c1[:1500], c2[:500], c1[1500:], c2[500:]])
    d = pd.DataFrame({("fret", "particle"): [0] * len(c1) + [1] * len(c2),
                      ("fret", "eff_app"): d[:, 0],
                      ("fret", "stoi_app"): d[:, 1]})

    labels, means = base.analyzer.gaussian_mixture_split(d, 2)
    np.testing.assert_array_equal(
        labels, [1] * 1500 + [0] * 500 + [1] * 500 + [0] * 1500)
    np.testing.assert_allclose(means, [[0.9, 0.5], [0.1, 0.8]], atol=0.005)
