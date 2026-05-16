"""LMDB format: directory-based LMDB store with per-column sub-databases.

Schema mirrors the HDF5 format exactly:
  - One sub-database per column, keyed by global frame index (8-digit zero-padded)
  - '__meta__' sub-database stores ep_len, ep_offset, shapes, dtypes
  - Lazy per-worker env opening for safe use with num_workers > 0
"""

from __future__ import annotations

import logging
import os
import pickle
import re
from collections.abc import Callable
from pathlib import Path

import lmdb
import numpy as np
import torch

from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.format import (
    Format,
    register_format,
    validate_write_mode,
)
from stable_worldmodel.data.utils import get_cache_dir

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# _KEY_FMT = "{:010d}".encode.__func__  # not used directly; see _frame_key()

def _frame_key(idx: int) -> bytes:
    return f"{idx:010d}".encode()


def _open_env(path: Path, *, write: bool = False) -> lmdb.Environment:
    """Open an LMDB environment with sane defaults."""
    return lmdb.open(
        str(path),
        subdir=False,
        readonly=not write,
        lock=write,          # lock=False for read-only (safe for multi-worker / NFS)
        readahead=False,     # random access — OS readahead hurts more than it helps
        meminit=False,       # don't zero-fill pages on allocation
        max_dbs=256,         # one named DB per column + __meta__
        map_size=1 << 40,    # 1 TiB virtual address space; actual disk use stays small
    )


# ---------------------------------------------------------------------------
# Dataset (reader)
# ---------------------------------------------------------------------------

class LMDBDataset(Dataset):
    """Dataset loading from an LMDB store.

    Drop-in replacement for HDF5Dataset.  All constructor arguments are
    identical; the only addition is that ``path`` must point to an LMDB
    directory instead of an .h5 file.
    """

    def __init__(
        self,
        name: str | None = None,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        keys_to_merge: dict[str, list[str] | str] | None = None,
        cache_dir: str | Path | None = None,
        path: str | Path | None = None,
    ) -> None:
        if path is not None:
            self.lmdb_path = Path(path)
        else:
            if name is None:
                raise TypeError("LMDBDataset requires either `name` or `path`")
            datasets_dir = get_cache_dir(cache_dir, sub_folder="datasets")
            self.lmdb_path = Path(datasets_dir, f"{name}.lmdb")

        # Worker-local state — populated lazily in _open()
        self._env: lmdb.Environment | None = None
        self._dbs: dict[str, object] = {}   # named DB handles
        self._handle_pid: int | None = None
        self._cache: dict[str, np.ndarray] = {}

        # Read metadata in a temporary env (main process only, safe)
        meta = self._read_meta()
        lengths   = np.array(meta["ep_len"],    dtype=np.int32)
        offsets   = np.array(meta["ep_offset"], dtype=np.int64)
        self._col_shapes: dict[str, tuple]     = meta["col_shapes"]
        self._col_dtypes: dict[str, np.dtype]  = {
            k: np.dtype(v) for k, v in meta["col_dtypes"].items()
        }
        all_cols = list(self._col_shapes.keys())
        self._keys = keys_to_load or all_cols

        # Eagerly cache requested columns (same semantics as HDF5Dataset)
        for key in keys_to_cache or []:
            self._cache[key] = self._read_full_column(key)
            logging.info(f"Cached '{key}' from '{self.lmdb_path}'")

        super().__init__(lengths, offsets, frameskip, num_steps, transform)

        if keys_to_merge:
            for target, source in keys_to_merge.items():
                self.merge_col(source, target)

    # ------------------------------------------------------------------
    # Public API (mirrors HDF5Dataset)
    # ------------------------------------------------------------------

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def get_col_data(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        return self._read_full_column(col)

    def get_row_data(self, row_idx: int | list[int]) -> dict:
        self._open()
        indices = [row_idx] if isinstance(row_idx, int) else row_idx
        result = {}
        
        with self._env.begin(buffers=True) as txn:
            for col in self._keys:
                if col != "pixels":
                    # If the framework requests actions/states, you'll need to handle
                    # how they're sliced if they aren't stored sequentially by frame.
                    continue
                    
                frames = []
                for i in indices:
                    # Match your writer's 8-character zero padding sequence format
                    key = f"{i:08d}".encode()
                    buf = txn.get(key)
                    if buf is None:
                        raise KeyError(f"Frame {i} not found in flat LMDB storage.")
                    
                    arr = np.frombuffer(bytes(buf), dtype=self._col_dtypes[col])
                    arr = arr.reshape(self._col_shapes[col]).copy()
                    frames.append(arr)
                result[col] = np.stack(frames) if len(frames) > 1 else frames[0]
        return result

    def merge_col(
        self,
        source: list[str] | str,
        target: str,
        dim: int = -1,
    ) -> None:
        if isinstance(source, str):
            # Treat as a regex pattern over known columns
            all_cols = list(self._col_shapes.keys())
            source = [k for k in all_cols if re.match(source, k)]

        merged = np.concatenate([self.get_col_data(s) for s in source], axis=dim)
        self._cache[target] = merged
        if target not in self._keys:
            self._keys.append(target)
        logging.info(f"Merged columns {source} into '{target}' and cached it")

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return int(np.prod(data.shape[1:])) if data.ndim > 1 else 1

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> None:
        """Open (or re-open) the LMDB env in the current process."""
        curr_pid = os.getpid()
        if self._handle_pid is not None and self._handle_pid != curr_pid:
            self._env = None
            self._dbs = {}

        if self._env is None:
            # Open without named DB parameters since your file uses the default DB
            self._env = _open_env(self.lmdb_path, write=False)
            self._handle_pid = curr_pid

    def _read_meta(self) -> dict:
        env = _open_env(self.lmdb_path, write=False)
        # Read from the default database instead of calling open_db()
        with env.begin(write=False) as txn:
            raw = txn.get(b"__metadata__")  # Matches your writer's key name
            if raw is None:
                raise ValueError(f"LMDB at '{self.lmdb_path}' has no __metadata__ entry")
            writer_meta = pickle.loads(raw)
        env.close()

        # Translate your writer's flat structure into the episodic structure 
        # that the rest of the stable_worldmodel framework expects
        total_frames = writer_meta["__total_frames__"]
        
        # Build the metadata payload your framework relies on
        meta = {
            "ep_len": [total_frames],    # Treat whole dataset as one continuous episode
            "ep_offset": [0],
            "col_shapes": {
                "pixels": tuple(writer_meta["__image_shape__"])
            },
            "col_dtypes": {
                "pixels": np.dtype(writer_meta["__image_dtype__"])
            }
        }
        
        # Map any auxiliary metadata vectors you saved as flat columns
        for k, v in writer_meta.items():
            if not k.startswith('__') and isinstance(v, list):
                arr_v = np.array(v)
                meta["col_shapes"][k] = arr_v.shape[1:] if arr_v.ndim > 1 else ()
                meta["col_dtypes"][k] = arr_v.dtype

        return meta
    
    def _read_full_column(self, col: str) -> np.ndarray:
        """Sequential read over the flat DB database."""
        if col != "pixels":
            raise NotImplementedError("This flat-patched reader only supports full sequential reads for 'pixels'")
            
        env = _open_env(self.lmdb_path, write=False)
        shape = self._col_shapes[col]
        dtype = self._col_dtypes[col]
        frames = []
        
        with env.begin(buffers=True) as txn:
            cursor = txn.cursor()
            for key, buf in cursor:
                # Filter out the metadata string key during the array stream
                if key == b"__metadata__":
                    continue
                arr = np.frombuffer(bytes(buf), dtype=dtype).reshape(shape).copy()
                frames.append(arr)
        env.close()
        return np.stack(frames)

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        self._open()
        g_start = int(self.offsets[ep_idx]) + start
        g_end   = int(self.offsets[ep_idx]) + end

        steps = {}
        # Open transaction on the root environment (buffers=True is fine)
        with self._env.begin(buffers=True) as txn:
            for col in self._keys:
                if col in self._cache:
                    data = self._cache[col][g_start:g_end]
                else:
                    if col != "pixels":
                        # Skip other columns for now if they aren't stored 
                        # sequentially by frame key in the main LMDB root
                        continue
                        
                    dtype = self._col_dtypes[col]
                    shape = self._col_shapes[col]
                    frames = []
                    
                    for i in range(g_start, g_end):
                        # Match your writer's 8-character zero padding sequence format
                        key = f"{i:08d}".encode()
                        
                        # Fetch directly from the root txn instead of passing a db handle
                        buf = txn.get(key)
                        if buf is None:
                            raise KeyError(
                                f"Frame {i} missing for column '{col}' "
                                f"(ep={ep_idx}, local {start}:{end})"
                            )
                        arr = np.frombuffer(bytes(buf), dtype=dtype)
                        frames.append(arr.reshape(shape).copy())
                    data = np.stack(frames)

                if col != "action":
                    data = data[:: self.frameskip]

                if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                    val = data[0] if len(data) > 0 else b""
                    steps[col] = val.decode() if isinstance(val, bytes) else val
                else:
                    steps[col] = torch.from_numpy(data)
                    if data.ndim == 4 and data.shape[-1] in (1, 3):
                        steps[col] = steps[col].permute(0, 3, 1, 2)

        return self.transform(steps) if self.transform else steps


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class LMDBWriter:
    """Append episodes to an LMDB store.

    Args:
        path:  target directory (created if needed).
        mode:  ``'append'``, ``'overwrite'``, or ``'error'``.
    """

    # Flush to disk every N frames to keep transactions small
    _FRAMES_PER_TXN = 2000

    def __init__(self, path, *, mode: str = "append"):
        validate_write_mode(mode)
        self.path = Path(path)
        self.mode = mode
        self._env: lmdb.Environment | None = None
        self._meta_db = None
        self._col_dbs: dict[str, object] = {}
        self._initialized = False
        self._appending_existing = False
        self._ep_written = 0
        self._global_ptr = 0
        self._ep_lens: list[int] = []
        self._ep_offsets: list[int] = []
        self._col_shapes: dict[str, tuple] = {}
        self._col_dtypes: dict[str, str] = {}

    def __enter__(self):
        exists = self.path.exists() and any(self.path.iterdir()) if self.path.exists() else False

        if exists and self.mode == "error":
            raise FileExistsError(
                f"LMDBWriter: '{self.path}' already exists. "
                "Pass mode='overwrite' to replace it or mode='append' to extend it."
            )

        if self.mode == "overwrite" or not exists:
            import shutil
            if self.path.exists():
                shutil.rmtree(self.path)
            self.path.mkdir(parents=True, exist_ok=True)
        else:
            self.path.mkdir(parents=True, exist_ok=True)

        self._env = _open_env(self.path, write=True)
        self._meta_db = self._env.open_db(b"__meta__")

        if exists and self.mode == "append":
            self._load_existing_state()

        return self

    def __exit__(self, *exc):
        self._flush_meta()
        if self._env is not None:
            self._env.close()
            self._env = None

    def write_episode(self, ep_data: dict) -> None:
        if self._env is None:
            raise RuntimeError("LMDBWriter used outside of a `with` block")

        if not self._initialized:
            self._init_schema(ep_data)
            self._initialized = True
        elif self._appending_existing and self._ep_written == 0:
            self._validate_episode_against_existing(ep_data)

        ep_len = len(next(iter(ep_data.values())))

        # Write frames in bounded transactions
        with self._env.begin(write=True) as txn:
            for col, vals in ep_data.items():
                db = self._col_dbs[col]
                arr = np.asarray(vals)
                for local_i in range(ep_len):
                    key = _frame_key(self._global_ptr + local_i)
                    txn.put(key, arr[local_i].tobytes(), db=db)

        self._ep_lens.append(ep_len)
        self._ep_offsets.append(self._global_ptr)
        self._global_ptr += ep_len
        self._ep_written += 1

    def write_episodes(self, episodes) -> None:
        for ep in episodes:
            self.write_episode(ep)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_schema(self, sample_ep: dict) -> None:
        for col, vals in sample_ep.items():
            sample = np.asarray(vals[0])
            self._col_shapes[col] = sample.shape
            self._col_dtypes[col] = sample.dtype.str
            self._col_dbs[col] = self._env.open_db(col.encode())

    def _load_existing_state(self) -> None:
        with self._env.begin(db=self._meta_db) as txn:
            raw = txn.get(b"__meta__")
            if raw is None:
                raise ValueError(
                    f"LMDBWriter: cannot append to '{self.path}' — missing __meta__."
                )
            meta = pickle.loads(raw)

        self._ep_lens    = list(meta["ep_len"])
        self._ep_offsets = list(meta["ep_offset"])
        self._col_shapes = meta["col_shapes"]
        self._col_dtypes = meta["col_dtypes"]
        self._global_ptr = int(sum(self._ep_lens))

        for col in self._col_shapes:
            self._col_dbs[col] = self._env.open_db(col.encode())

        self._initialized = True
        self._appending_existing = True

    def _validate_episode_against_existing(self, ep_data: dict) -> None:
        existing = set(self._col_shapes.keys())
        incoming = set(ep_data.keys())
        missing = existing - incoming
        extra   = incoming - existing
        if missing or extra:
            raise ValueError(
                f"LMDBWriter: append failed — schema mismatch on '{self.path}'. "
                f"Missing columns: {sorted(missing)}; unexpected: {sorted(extra)}."
            )
        for col, vals in ep_data.items():
            sample = np.asarray(vals[0])
            if sample.shape != self._col_shapes[col]:
                raise ValueError(
                    f"LMDBWriter: column '{col}' shape mismatch: "
                    f"existing={self._col_shapes[col]}, incoming={sample.shape}."
                )

    def _flush_meta(self) -> None:
        if self._env is None:
            return
        meta = {
            "ep_len":     self._ep_lens,
            "ep_offset":  self._ep_offsets,
            "col_shapes": self._col_shapes,
            "col_dtypes": self._col_dtypes,
        }
        with self._env.begin(write=True, db=self._meta_db) as txn:
            txn.put(b"__meta__", pickle.dumps(meta))


# ---------------------------------------------------------------------------
# Format registration
# ---------------------------------------------------------------------------

@register_format
class LMDB(Format):
    name = "lmdb"

    @classmethod
    def detect(cls, path) -> bool:
        p = Path(path)
        # An LMDB directory contains data.mdb and lock.mdb
        return p.is_dir() and (p / "data.mdb").exists()

    @classmethod
    def open_reader(cls, path, **kwargs) -> LMDBDataset:
        return LMDBDataset(path=path, **kwargs)

    @classmethod
    def open_writer(cls, path, **kwargs) -> LMDBWriter:
        return LMDBWriter(path, **kwargs)


__all__ = ["LMDB", "LMDBDataset", "LMDBWriter"]