import logging
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from PIL import Image

# Add the project root to the Python path to enable imports from `src`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generate_data import do_generate_data

log = logging.getLogger(__name__)


def validate_generated_data(cfg: OmegaConf):
    """Validates the integrity of the generated data."""
    log.info("Running validation on generated data...")
    metadata_path = os.path.join(cfg.data.root_dir, "train_metadata.csv")
    metadata_df = pd.read_csv(metadata_path)
    image_size = tuple(cfg.data.image_size)

    for data_format in cfg.data.formats:
        log.info(f"Validating format: {data_format}")
        format_df = metadata_df[metadata_df["format"] == data_format]

        for _, row in format_df.iterrows():
            file_path = row["file_path"]
            log.info(f"Validating file: {file_path}")

            if data_format == "png":
                image = Image.open(file_path)
                image_np = np.array(image).transpose(2, 0, 1)
                assert image_np.shape == image_size
            elif data_format == "npy":
                image_np = np.load(file_path)
                assert image_np.shape == image_size
            elif data_format == "arrow":
                with pa.OSFile(file_path, "rb") as source:
                    reader = pa.ipc.open_file(source)
                    rb = reader.read_all()
                    image_bytes = rb.column("image")[0].as_py()
                    shape = rb.column("shape")[0].as_py()
                    image_np = np.frombuffer(image_bytes, dtype=np.uint8).reshape(shape)
                    assert image_np.shape == image_size
            elif data_format == "parquet":
                table = pq.read_table(file_path)
                df = table.to_pandas()
                image_flat = df["image"].iloc[0]
                channels = df["channels"].iloc[0]
                height = df["height"].iloc[0]
                width = df["width"].iloc[0]
                image_np = np.array(image_flat).reshape(channels, height, width)
                assert image_np.shape == image_size

    log.info("All data validation checks passed!")


class TestDataGeneration(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory and config for tests."""
        self.test_dir = "test_data_temp"
        os.makedirs(self.test_dir, exist_ok=True)
        self.cfg = OmegaConf.create(
            {
                "data": {
                    "root_dir": self.test_dir,
                    "num_samples": 20,
                    "image_size": [3, 32, 32],
                    "num_classes": 2,
                    "formats": ["png", "npy", "arrow", "parquet"],
                    "parquet_batch_size": 10,
                },
                "seed": 42,
            }
        )
        do_generate_data(self.cfg)

    def tearDown(self):
        """Remove the temporary directory after tests."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_parquet_file_content(self):
        """Test that Parquet files are created and content is correct."""
        metadata_path = os.path.join(self.test_dir, "train_metadata.csv")
        self.assertTrue(os.path.exists(metadata_path))
        metadata_df = pd.read_csv(metadata_path)

        parquet_meta = metadata_df[metadata_df["format"] == "parquet"].iloc[0]
        parquet_file_path = parquet_meta["file_path"]
        self.assertTrue(os.path.exists(parquet_file_path))

        # Read the Parquet file
        table = pq.read_table(parquet_file_path)
        df = table.to_pandas()

        # Get the specific row for the image
        index_in_batch = int(parquet_meta["index_in_batch"])
        image_row = df.iloc[index_in_batch]

        # Reconstruct the image
        image_flat = image_row["image"]
        channels = image_row["channels"]
        height = image_row["height"]
        width = image_row["width"]
        image_reconstructed = (
            np.array(image_flat).reshape(channels, height, width).astype(np.uint8)
        )

        # For this test, we'll just check the shape and type
        self.assertEqual(image_reconstructed.shape, tuple(self.cfg.data.image_size))
        self.assertEqual(image_reconstructed.dtype, np.uint8)

        # To do a more thorough check, we'd need to regenerate the original
        # data with the same seed and compare the exact pixel values.
        # This is a good first step to ensure the pipeline is working.


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
