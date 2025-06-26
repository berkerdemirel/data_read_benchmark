import os
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


# ───────────────────────── helper ──────────────────────────
class _ParquetShard:
    """
    Open-once wrapper around pyarrow.ParquetFile.
    Provides fast, row-level access to the binary 'image' column.
    """

    def __init__(self, path: str):
        self.file = pq.ParquetFile(path)
        self.rows_per_rg = self.file.metadata.row_group(0).num_rows

    @lru_cache(maxsize=None)                      # cache Arrow RecordBatch objects
    def _row_group(self, rg_idx: int) -> pa.RecordBatch:
        return self.file.read_row_group(rg_idx, columns=["image"])

    def get_image_flat(self, global_row: int) -> memoryview:
        rg_idx = global_row // self.rows_per_rg
        offset = global_row % self.rows_per_rg
        col = self._row_group(rg_idx).column(0)[offset]  # pyarrow.BinaryScalar
        return col.as_buffer()                           # zero-copy memoryview
# ───────────────────────────────────────────────────────────


def get_image_folder_dataset(
    data_dir: str, transform: Optional[Callable] = None
) -> ImageFolder:
    """Returns a standard ImageFolder dataset for PNG images."""
    return ImageFolder(root=os.path.join(data_dir, "png"), transform=transform)


class MultiFormatDataset(Dataset):
    """
    Unified dataset that can read png / npy / arrow / parquet samples
    according to the 'format' column in metadata CSV.
    """

    def __init__(
        self,
        metadata_csv: str,
        data_format: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int, int] | None = None,   # (C,H,W) if parquet
    ):
        super().__init__()
        self.transform = transform
        self.metadata = (
            pd.read_csv(metadata_csv)
            .query("format == @data_format")
            .reset_index(drop=True)
        )

        self._shard_cache: dict[str, _ParquetShard] = {}      # for parquet
        self._C, self._H, self._W = (
            image_size if image_size else (None, None, None)
        )

        loaders: Dict[str, Callable[[int], Any]] = {
            "png": self._load_png,
            "npy": self._load_npy,
            "arrow": self._load_arrow,
            "parquet": self._load_parquet,
        }
        if data_format not in loaders:
            raise ValueError(f"Unsupported format: {data_format}")
        self._loader = loaders[data_format]

    # ─────────────── individual format loaders ───────────────
    def _load_png(self, idx: int):
        path = self.metadata.at[idx, "file_path"]
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img

    def _load_npy(self, idx: int):
        path = self.metadata.at[idx, "file_path"]
        arr = np.load(path).transpose(1, 2, 0)  # CHW → HWC
        return self.transform(arr) if self.transform else arr

    def _load_arrow(self, idx: int):
        path = self.metadata.at[idx, "file_path"]
        with pa.OSFile(path, "rb") as f:
            rb = pa.ipc.open_file(f).read_all()
            buf = rb.column("image")[0].as_py()
            shape = rb.column("shape")[0].as_py()
        arr = np.frombuffer(buf, np.uint8).reshape(shape).transpose(1, 2, 0)
        return self.transform(arr) if self.transform else arr

    # ★ fast parquet loader (row-level, cached) ★
    def _load_parquet(self, idx: int):
        meta = self.metadata.iloc[idx]
        shard = self._shard_cache.get(meta.file_path)
        if shard is None:
            shard = _ParquetShard(meta.file_path)
            self._shard_cache[meta.file_path] = shard

        flat = np.frombuffer(
            shard.get_image_flat(int(meta.index_in_batch)), dtype=np.uint8
        )

        if None in (self._C, self._H, self._W):
            raise RuntimeError(
                "image_size (C,H,W) must be passed to MultiFormatDataset"
                " when using parquet format."
            )
        img = flat.reshape(self._C, self._H, self._W).transpose(1, 2, 0)
        return self.transform(img) if self.transform else img
    # ──────────────────────────────────────────────────────────

    # Torch-style boilerplate
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        label = self.metadata.at[idx, "label"]
        return self._loader(idx), label
