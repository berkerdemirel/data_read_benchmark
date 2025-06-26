import os
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def get_image_folder_dataset(
    data_dir: str, transform: Optional[Callable] = None
) -> ImageFolder:
    """Returns a standard ImageFolder dataset for PNG images."""
    return ImageFolder(root=os.path.join(data_dir, "png"), transform=transform)


class MultiFormatDataset(Dataset):
    """
    A custom dataset that loads images from multiple formats based on a metadata file.

    This dataset uses a metadata CSV file to find the paths to the data files
    and their corresponding labels. It can handle several data formats by using
    a dedicated loader function for each format.

    Args:
        metadata_csv (str): Path to the metadata CSV file. The CSV should
            contain 'file_path', 'label', and 'format' columns.
        data_format (str): The data format to load (e.g., 'png', 'npy', 'arrow', 'parquet').
        transform (Optional[Callable]): A function/transform to apply to the data.
            Defaults to None.
    """

    def __init__(
        self, metadata_csv: str, data_format: str, transform: Optional[Callable] = None
    ):
        """Initializes the dataset by loading metadata and setting up the correct loader."""
        self.transform = transform
        self.metadata = (
            pd.read_csv(metadata_csv)
            .query("format == @data_format")
            .reset_index(drop=True)
        )
        # Cache for the currently loaded parquet shard. This is designed to work
        # with a sampler that iterates through shards sequentially.
        self._cached_shard_path: Optional[str] = None
        self._cached_shard_data: Optional[pa.Table] = None

        loaders: Dict[str, Callable[[int], Any]] = {
            "png": self._load_png,
            "npy": self._load_npy,
            "arrow": self._load_arrow,
            "parquet": self._load_parquet,
        }
        if data_format not in loaders:
            raise ValueError(f"Unsupported format: {data_format}")
        self._loader = loaders[data_format]

    def _load_png(self, idx: int) -> Any:
        """Loads a PNG image and applies transformations."""
        path = self.metadata.at[idx, "file_path"]
        img = Image.open(path).convert("RGB")
        return self.transform(img) if self.transform else img

    def _load_npy(self, idx: int) -> Any:
        """Loads a NumPy array and applies transformations."""
        path = self.metadata.at[idx, "file_path"]
        arr = np.load(path)
        # Transpose from (C, H, W) to (H, W, C) for ToTensor transform
        arr = arr.transpose(1, 2, 0)
        return self.transform(arr) if self.transform else arr

    def _load_arrow(self, idx: int) -> Any:
        """Loads an Arrow file, reconstructs the image, and applies transformations."""
        path = self.metadata.at[idx, "file_path"]
        with pa.OSFile(path, "rb") as f:
            rb = pa.ipc.open_file(f).read_all()
            buf = rb.column("image")[0].as_py()
            shape = rb.column("shape")[0].as_py()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(shape)
        # Transpose from (C, H, W) to (H, W, C) for ToTensor transform
        arr = arr.transpose(1, 2, 0)
        return self.transform(arr) if self.transform else arr

    def _load_parquet(self, idx: int) -> Any:
        """
        Loads an image from a batched Parquet file and applies transformations.
        This method caches one Parquet file (shard) in memory at a time to speed up
        access. It is designed to be used with a ShardSampler.
        """
        row_metadata = self.metadata.loc[idx]
        path = row_metadata["file_path"]
        index_in_batch = int(row_metadata["index_in_batch"])

        # If the requested sample is not in the cached shard, load the new shard.
        if path != self._cached_shard_path:
            self._cached_shard_data = pq.read_table(path)
            self._cached_shard_path = path

        # Retrieve the shard from the cache.
        shard_table = self._cached_shard_data

        # Access the specific row from the cached Arrow Table.
        image_row = shard_table.slice(index_in_batch, 1).to_pandas()

        image_flat = image_row["image"].iloc[0]
        channels = image_row["channels"].iloc[0]
        height = image_row["height"].iloc[0]
        width = image_row["width"].iloc[0]

        arr = np.array(image_flat).reshape(channels, height, width).astype(np.uint8)
        # Transpose from (C, H, W) to (H, W, C) for ToTensor transform
        arr = arr.transpose(1, 2, 0)
        return self.transform(arr) if self.transform else arr

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """
        Fetches a sample (data and label) at the given index.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A tuple containing the data and its corresponding label.
        """
        label = self.metadata.at[idx, "label"]
        return self._loader(idx), label
