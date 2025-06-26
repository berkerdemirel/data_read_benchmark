import logging
import os

import hydra
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

log = logging.getLogger(__name__)


def do_generate_data(cfg: DictConfig) -> None:
    """
    Core logic to generates and saves a random dataset in specified formats.

    This script creates a dataset of random images and saves them in multiple
    formats (PNG, Arrow, Parquet, NPY). The data is organized into class folders,
    and metadata is saved to a CSV file.

    Args:
        cfg (DictConfig): Hydra configuration object.
    """
    # Set random seed for reproducibility
    if hasattr(cfg, "seed"):
        np.random.seed(cfg.seed)

    # Create the root directory for the generated data
    os.makedirs(cfg.data.root_dir, exist_ok=True)

    # Generate the full dataset
    num_samples: int = cfg.data.num_samples
    image_size: tuple = tuple(cfg.data.image_size)
    num_classes: int = cfg.data.num_classes

    # Create balanced class labels
    samples_per_class: int = num_samples // num_classes
    labels: np.ndarray = np.repeat(np.arange(num_classes), samples_per_class)

    # If num_samples is not perfectly divisible by num_classes, distribute the remainder
    remainder: int = num_samples % num_classes
    if remainder > 0:
        labels = np.concatenate([labels, np.arange(remainder)])

    np.random.shuffle(labels)

    images: np.ndarray = np.random.randint(
        0, 256, size=(num_samples, *image_size), dtype=np.uint8
    )

    metadata = []

    # Generate formats that are one-file-per-image
    one_file_formats = [f for f in cfg.data.formats if f not in ["parquet"]]
    if one_file_formats:
        log.info(f"Generating {', '.join(one_file_formats)} files...")
        for i in tqdm(range(num_samples), desc="Generating individual files"):
            class_label: int = int(labels[i])
            image_data: np.ndarray = images[i]

            for data_format in one_file_formats:
                format_dir: str = os.path.join(cfg.data.root_dir, data_format)
                class_dir: str = os.path.join(format_dir, f"class_{class_label}")
                os.makedirs(class_dir, exist_ok=True)

                file_path: str = os.path.join(
                    class_dir,
                    f"image_{i}.{data_format if data_format != 'arrow' else 'arrow'}",
                )

                if data_format == "png":
                    # Transpose from (C, H, W) to (H, W, C) for PIL
                    Image.fromarray(image_data.transpose(1, 2, 0)).save(file_path)
                elif data_format == "npy":
                    np.save(file_path, image_data)
                elif data_format == "arrow":
                    # For a single image, writing pa.py_buffer is more idiomatic.
                    # We'll also save the shape to reconstruct the array.
                    schema = pa.schema(
                        [
                            pa.field("image", pa.binary()),
                            pa.field("shape", pa.list_(pa.int64())),
                        ]
                    )
                    rb = pa.RecordBatch.from_arrays(
                        [
                            pa.array([image_data.tobytes()]),
                            pa.array([list(image_data.shape)]),
                        ],
                        schema=schema,
                    )
                    with pa.OSFile(file_path, "wb") as sink:
                        with pa.ipc.new_file(sink, schema=schema) as writer:
                            writer.write(rb)

                metadata.append(
                    {
                        "file_path": file_path,
                        "label": class_label,
                        "format": data_format,
                    }
                )

    # Generate Parquet format in batches
    if "parquet" in cfg.data.formats:
        log.info("Generating Parquet files in batches…")
        format_dir: str = os.path.join(cfg.data.root_dir, "parquet")
        os.makedirs(format_dir, exist_ok=True)

        BATCH = cfg.data.parquet_batch_size          # e.g. 4_096
        C, H, W = image_size                         # channels, height, width

        for i in tqdm(range(0, num_samples, BATCH), desc="Parquet batches"):
            batch_images = images[i : i + BATCH]         # (B,C,H,W)
            batch_labels = labels[i : i + BATCH]         # (B,)
            file_idx     = i // BATCH
            file_path    = os.path.join(format_dir, f"batch_{file_idx:04d}.parquet")

            # ---- build Arrow table ------------------------------------------------
            # 1) one binary blob per image (fastest for row-level reads)
            blobs  = [img.tobytes() for img in batch_images]          # zero copy later
            table  = pa.Table.from_arrays(
                [
                    pa.array(blobs, pa.binary()),                     # image column
                    pa.array(batch_labels.astype(np.int16)),          # label column
                ],
                names=["image", "label"],
            )

            # 2) write once, 4 096-row row-groups, light (zstd-1) compression
            pq.write_table(
                table,
                file_path,
                row_group_size=4096,          # ≈128 MB per row-group
                compression="zstd",
                compression_level=1,
            )
            # -----------------------------------------------------------------------

            # update metadata — row index *inside the file* is just offset in batch
            for k in range(len(batch_images)):
                metadata.append(
                    {
                        "file_path": file_path,
                        "label":    int(batch_labels[k]),
                        "format":   "parquet",
                        "index_in_batch": k,          # 0 … BATCH-1
                    }
                )

    # Save metadata
    metadata_df = pd.DataFrame(metadata)

    # # Split metadata into train and validation sets
    # train_df, val_df = train_test_split(
    #     metadata_df,
    #     test_size=0.2,
    #     random_state=cfg.seed if hasattr(cfg, "seed") else None,
    #     stratify=metadata_df["label"].astype(str) + metadata_df["format"].astype(str),
    # )

    train_df = metadata_df

    train_df.to_csv(os.path.join(cfg.data.root_dir, "train_metadata.csv"), index=False)
    # val_df.to_csv(os.path.join(cfg.data.root_dir, "val_metadata.csv"), index=False)

    log.info(f"Dataset successfully generated at: {cfg.data.root_dir}")


@hydra.main(config_path="..", config_name="config", version_base=None)
def generate_data(cfg: DictConfig) -> None:
    do_generate_data(cfg)


if __name__ == "__main__":
    generate_data()
