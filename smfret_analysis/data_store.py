# SPDX-FileCopyrightText: 2021 Lukas Schrangl <lukas.schrangl@tuwien.ac.at>
#
# SPDX-License-Identifier: BSD-3-Clause

from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, Iterable, Optional, Union
import warnings

import numpy as np
import pandas as pd
from sdt import flatfield as _flatfield, io


class DataStore:
    """Save and load data

    Data is stored as attributes (e.g. :py:attr:`tracks`). Arbitrary
    attributes can be added and will be serialized to YAML when saving. Some
    attributes (:py:attr:`localizations`, :py:attr:`tracks`,
    :py:attr:`flatfield`, :py:attr:`segment_images`) receive special treatment
    and are saved to binary files.

    Examples
    --------
    If ``loc`` is a dictionary mapping dataset ids to localization data, and
    ``trc`` maps dataset ids to tracking data, save to disk using

    >>> ds = DataStore(localizations=loc, tracks=trc)
    >>> ds.save()

    Load data using the :py:meth:`load` method, access data as attributes:

    >>> ld = DataStore.load()
    >>> trc = ld.tracks
    """
    localizations: Dict[str, pd.DataFrame]
    """Maps dataset id -> localization data as created by
    :py:meth:`Tracker.locate`
    """
    tracks: Dict[str, pd.DataFrame]
    """Maps dataset id -> tracking data as created by
    :py:meth:`Tracker.track`
    """
    flatfield: Dict[str, _flatfield.Corrector]
    """Maps excitation channel ("donor" or "acceptor") to flatfield Corrector
    instance.
    """
    segment_images: Dict[str, Dict[str, np.ndarray]]
    """Maps dataset id, file id -> sequence of images used for segmentation"""
    data_dir: Path
    """Path to raw data"""

    def __init__(self, **data):
        """Parameters
        ----------
        **data
            Data to add to class instance as attributes
        """
        if "data_dir" in data:
            self.data_dir = Path(data.pop("data_dir"))
        self.__dict__.update(data)

    @staticmethod
    def copy_files(old_prefix: str, new_prefix: str, version: int = 14):
        """Copy save files to a new prefix

        Parameters
        ----------
        old_prefix
            File prefix to copy from. See also :py:meth:`save` and :py:meth:`load`.
        new_prefix
            File prefix to copy to.
        version
            Which file version to copy. Currently supports only v014.
        """
        if version != 14:
            raise ValueError("only v014 files can be copied for now")

        import shutil

        for suffix in (
            "loc.h5",
            "tracks.h5",
            "seg_img.npz",
            "flat_donor.npz",
            "flat_acceptor.npz",
            "yaml"
        ):
            src = Path(f"{old_prefix}-v014.{suffix}")
            if not src.exists():
                continue
            dst = f"{new_prefix}-v014.{suffix}"
            shutil.copy(src, dst)

    def save(self, file_prefix: str = "tracking", mode: str = "write"):
        """Save data to disk

        This will save attributes to disk.

        Parameters
        ----------
        file_prefix
            Common file_prefix for all files written by this method. It will be
            suffixed by the output format version (v{version}) and file
            extensions corresponding to what is saved in each file.
        mode
            Use "write" to delete previously existing files (which contain
            localization and tracking data) and write a new ones. As a result,
            only the current data will end up in the file. Use "update" to
            add or modify data without deleting anything not present in this
            instance.
            """
        outfile = Path(f"{file_prefix}-v014")
        data = self.__dict__.copy()
        file_mode = "w" if mode == "write" else "a"

        with warnings.catch_warnings():
            import tables
            warnings.simplefilter("ignore", tables.NaturalNameWarning)

            with pd.HDFStore(outfile.with_suffix(".loc.h5"), file_mode) as s:
                for key, loc in data.pop("localizations", {}).items():
                    s.put(f"/locs/{key}", loc)
                for key, loc in data.pop("special_localizations", {}).items():
                    s.put(f"/special_locs/{key}", loc)
            with pd.HDFStore(outfile.with_suffix(".tracks.h5"),
                             file_mode) as s:
                for key, trc in data.pop("tracks", {}).items():
                    # Categorical exc_type does not allow for storing in fixed
                    # format while multiindex for both rows and columns does
                    # not work with table format…
                    s.put(f"/tracks/{key}",
                          trc.astype({("fret", "exc_type"): str}))
                for key, trc in data.pop("special_tracks", {}).items():
                    s.put(f"/special_tracks/{key}",
                          trc.astype({("fret", "exc_type"): str}))

        if mode == "write":
            old = {}
            seg = {}
        else:
            old = self.load(file_prefix, loc=False, tracks=False,
                            segment_images="segment_images" in data,
                            flatfield=False).__dict__.copy()
            seg = old.pop("segment_images", {})

        if "segment_images" in data:
            for k, v in data.pop("segment_images").items():
                for k2, v2 in v.items():
                    seg[f"{k}/{k2}"] = v2
            np.savez_compressed(outfile.with_suffix(".seg_img.npz"), **seg)
        if "flatfield" in data:
            if mode == "write":
                ffiles = io.get_files(
                    fr"^{outfile}\.flat_([\w\s-]+)\.npz$")[0]
                for f in ffiles:
                    Path(f).unlink()
            for k, ff in data.pop("flatfield").items():
                ff.save(outfile.with_suffix(f".flat_{k}.npz"))
        old.update(data)
        if "data_dir" in old:
            old["data_dir"] = str(old["data_dir"])
        with outfile.with_suffix(".yaml").open("w") as f:
            io.yaml.safe_dump(old, f)

    @classmethod
    def load(cls, file_prefix: str = "tracking",
             loc: Union[bool, Iterable[str]] = True,
             tracks: Union[bool, Iterable[str]] = True,
             segment_images: bool = True,
             flatfield: bool = True, version: Optional[int] = None,
             filtered_file_prefix: str = "filtered") -> "DataStore":
        """Load data to a new class instance

        Parameters
        ----------
        file_prefix
            Prefix used for saving via :py:meth:`save`.
        loc
            Whether to load localization data. If `True`, load all data. If
            `False`, don't load any. To load only certain datasets, specify
            their keys.
        tracks
            Whether to load tracking data. If `True`, load all data. If
            `False`, don't load any. To load only certain datasets, specify
            their keys.
        cell_images
            Whether to load cell images.
        flatfield
            Whether to load flatfield corrections.
        filtered_file_prefix
            Prefix used for saving analyzed and filtered data in v013 format,
            i.e. prefix passed to the ``save()`` method in the Analysis
            notebook. v013 only.

        Returns
        -------
            Dictionary of loaded data and settings.
        """
        if version in (14, None):
            try:
                return cls(**cls._load_v14(
                    file_prefix, loc, tracks, segment_images, flatfield))
            except FileNotFoundError:
                if version is not None:
                    raise
        if version in (13, None):
            try:
                return cls(**cls._load_v13(
                    file_prefix, loc, tracks, segment_images, flatfield,
                    filtered_file_prefix))
            except FileNotFoundError:
                if version is not None:
                    raise
        if version is None:
            raise ValueError("cannot find supported save files")
        else:
            raise ValueError(f"cannot load version {version} save files")

    @staticmethod
    def _load_v13(file_prefix: str, loc: Union[bool, Iterable[str]],
                  tracks: Union[bool, Iterable[str]], segment_images: bool,
                  flatfield: bool, filtered_file_prefix: str) -> Dict:
        """Load data to a dictionary (v013 format)

        Parameters
        ----------
        file_prefix
            Prefix used for saving via :py:meth:`save`.
        loc
            Whether to load localization data. If `True`, load all data. If
            `False`, don't load any. To load only certain datasets, specify
            their keys.
        tracks
            Whether to load tracking data. If `True`, load all data. If
            `False`, don't load any. To load only certain datasets, specify
            their keys.
        cell_images
            Whether to load cell images.
        flatfield
            Whether to load flatfield corrections.
        filtered_file_prefix
            Prefix used for saving analyzed and filtered data in v013 format,
            i.e. prefix passed to the ``save()`` method in the Analysis
            notebook.

        Returns
        -------
            Dictionary of loaded data and settings.
        """
        infile = Path(f"{file_prefix}-v013")
        with infile.with_suffix(".yaml").open() as f:
            ret = io.yaml.safe_load(f)

        ret["data_dir"] = Path(ret.get("data_dir", ""))
        ret["localizations"] = {}
        ret["tracks"] = {}
        ret["segment_images"] = defaultdict(dict)
        ret["flatfield"] = {}

        for src in ret["sources"], ret["special_sources"]:
            for k, v in src.items():
                if isinstance(v, list):
                    src[k] = {n: i if isinstance(i, str) else tuple(i)
                              for n, i in enumerate(v)}

        all_src = {**ret["sources"], **ret["special_sources"]}

        do_load = []
        if loc:
            do_load.append((ret["localizations"], "_loc", loc))
        if tracks:
            do_load.append((ret["tracks"], "_trc", tracks))
        if len(do_load):
            with pd.HDFStore(infile.with_suffix(".h5"), "r") as s:
                for sink, suffix, user_keys in do_load:
                    if isinstance(user_keys, bool):
                        # was set to `True` to load all data
                        keys = (k for k in s.keys() if k.endswith(suffix))
                    else:
                        # keys were specified
                        keys = (f"/{k}{suffix}" for k in user_keys)
                    for k in keys:
                        new_key = k[1:-len(suffix)]
                        loaded = s[k]
                        src = all_src[new_key]
                        fname_map = pd.Series(src.keys(),
                                              index=list(src.values()))
                        loaded.index = loaded.index.set_levels(
                            fname_map[loaded.index.levels[0]], level=0)
                        if ("fret", "exc_type") in loaded:
                            # Restore categorical exc_type. See comment in
                            # `save` method for details.
                            loaded = loaded.astype(
                                {("fret", "exc_type"): "category"}, copy=False)
                        sink[new_key] = loaded

        if tracks and filtered_file_prefix:
            filtered_file = Path(f"{filtered_file_prefix}-v013.h5")
            if filtered_file.exists():
                with pd.HDFStore(filtered_file, "r") as s:
                    for k, v in ret["tracks"].items():
                        try:
                            loaded = s[f"/{k}_trc"]
                        except KeyError:
                            # TODO: Warn?
                            continue
                        src = all_src[k]
                        fname_map = pd.Series(src.keys(), index=src.values())
                        loaded.index = loaded.index.set_levels(
                            fname_map[loaded.index.levels[0]], level=0)
                        v = loaded.combine_first(v)
                        v["filter", "load_v013"] = 1
                        v.loc[loaded.index, ("filter", "load_v013")] = 0
                        ret["tracks"][k] = v.astype(
                            {("fret", "exc_type"): "category"}, copy=False)

        if segment_images:
            seg_img_file = infile.with_suffix(".cell_img.npz")
            # Map file names to dataset IDs
            fname_map = {fname: (did, fid)
                         for did, src in all_src.items()
                         for fid, fname in src.items()}
            try:
                with np.load(seg_img_file) as data:
                    ci = dict(data)
                for k, v in ci.items():
                    k_split = k.split("\n")
                    if len(k_split) == 1:
                        new_k = k_split[0]
                    else:
                        new_k = tuple(k_split)
                    did, fid = fname_map[new_k]
                    ret["segment_images"][did][fid] = v
            except Exception as e:
                warnings.warn("Could not load segmentation images from file "
                              f"\"{str(seg_img_file)}\" ({e}).")
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

    @staticmethod
    def _load_v14(file_prefix: str, loc: Union[bool, Iterable[str]],
                  tracks: Union[bool, Iterable[str]], segment_images: bool,
                  flatfield: bool) -> Dict:
        """Load data to a dictionary (v014 format)

        Parameters
        ----------
        file_prefix
            Prefix used for saving via :py:meth:`save`.
        loc
            Whether to load localization data. If `True`, load all data. If
            `False`, don't load any. To load only certain datasets, specify
            their keys.
        tracks
            Whether to load tracking data. If `True`, load all data. If
            `False`, don't load any. To load only certain datasets, specify
            their keys.
        segment_images
            Whether to load segmentation images.
        flatfield
            Whether to load flatfield corrections.

        Returns
        -------
            Dictionary of loaded data and settings.
        """
        infile = Path(f"{file_prefix}-v014")
        with infile.with_suffix(".yaml").open() as f:
            ret = io.yaml.safe_load(f)

        if "data_dir" in ret:
            ret["data_dir"] = Path(ret["data_dir"])

        if loc:
            ret["localizations"] = {}
            ret["special_localizations"] = {}
            with pd.HDFStore(infile.with_suffix(".loc.h5"), "r") as s:
                if isinstance(loc, bool):
                    # was set to `True` to load all data
                    keys = s.keys()
                else:
                    # keys were specified
                    keys = (f"/locs/{k}" for k in tracks)
                for k in keys:
                    loaded = s.get(k)
                    if k.startswith("/locs/"):
                        ret["localizations"][k[6:]] = loaded
                    elif k.startswith("/special_locs/"):
                        ret["special_localizations"][k[14:]] = loaded
        if tracks:
            ret["tracks"] = {}
            ret["special_tracks"] = {}
            with pd.HDFStore(infile.with_suffix(".tracks.h5"), "r") as s:
                if isinstance(tracks, bool):
                    # was set to `True` to load all data
                    keys = s.keys()
                else:
                    # keys were specified
                    keys = (f"/tracks/{k}" for k in tracks)
                for k in keys:
                    # Restore categorical exc_type. See comment in
                    # `save` method for details.
                    loaded = s.get(k).astype(
                        {("fret", "exc_type"): "category"}, copy=False)
                    if k.startswith("/tracks/"):
                        ret["tracks"][k[8:]] = loaded
                    elif k.startswith("/special_tracks/"):
                        ret["special_tracks"][k[16:]] = loaded
        if segment_images:
            ret["segment_images"] = defaultdict(dict)
            seg_img_file = infile.with_suffix(".seg_img.npz")
            try:
                with np.load(seg_img_file) as data:
                    ci = dict(data)
                for k, v in ci.items():
                    split_idx = k.rfind("/")
                    k1 = k[:split_idx]
                    k2 = int(k[split_idx+1:])
                    ret["segment_images"][k1][k2] = v
            except Exception as e:
                warnings.warn("Could not load segmentation images from file "
                              f"\"{str(seg_img_file)}\" ({e}).")
        if flatfield:
            ret["flatfield"] = {}
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
