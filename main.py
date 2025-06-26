import os

import hydra
import pytorch_lightning
import torch
import wandb
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from src.datasets import MultiFormatDataset, get_image_folder_dataset
from src.generate_data import do_generate_data
from src.lightning_module import BenchmarkModule, DataloaderTimer
from src.sampler import DistributedShardSampler, ShardSampler
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


@rank_zero_only
def build_wandb_logger(
    cfg: DictConfig,
    data_format: str,
    batch_size: int,
    num_workers: int,
    gpus: list[int] = [],
    gpu_str: str = "cpu",
) -> WandbLogger:
    """
    Builds a Weights & Biases logger for tracking experiments.

    Returns:
        WandbLogger: Configured Weights & Biases logger.
    """
    return WandbLogger(
        project=cfg.wandb.project,
        name=f"{data_format}-bs{batch_size}-nw{num_workers}-{gpu_str}",
        config={
            "data_format": data_format,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "gpus": list(gpus),
            **dict(cfg),
        },
    )


@hydra.main(config_path=".", config_name="config", version_base="1.1")
def main(cfg: DictConfig) -> None:
    """
    Main script to run the data loading benchmark.

    This script uses Hydra to manage configurations and iterates through different
    data formats, batch sizes, and worker numbers to benchmark data loading
    performance. Results are logged to Weights & Biases.

    Args:
        cfg (DictConfig): The Hydra configuration object.
    """
    # Check if data exists, if not, generate it
    metadata_path = os.path.join(cfg.data.root_dir, "train_metadata.csv")
    if not os.path.exists(metadata_path):
        print("Metadata file not found. Generating data...")
        do_generate_data(cfg)
    else:
        print("Data already exists. Skipping generation.")

    # Define image transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Iterate over all combinations of benchmark parameters
    for gpus in cfg.trainer.gpus_configs:
        for data_format in cfg.data.formats:
            for batch_size in cfg.dataloader.batch_size:
                for num_workers in cfg.dataloader.num_workers:
                    # Initialize a new wandb run for each experiment
                    gpu_str = f"gpus-{'-'.join(map(str, gpus))}" if gpus else "cpu"
                    wandb_logger = build_wandb_logger(
                        cfg, data_format, batch_size, num_workers, gpus, gpu_str
                    )

                    # Load the appropriate dataset
                    if data_format == "png":
                        train_dataset = get_image_folder_dataset(
                            data_dir=cfg.data.root_dir, transform=transform
                        )
                    else:
                        train_dataset = MultiFormatDataset(
                            metadata_csv=f"{cfg.data.root_dir}/train_metadata.csv",
                            data_format=data_format,
                            transform=transform,
                        )

                    # Create the dataloader
                    is_ddp = len(gpus) > 1
                    if data_format == "parquet":
                        sampler = DistributedShardSampler(
                            train_dataset.metadata,
                            shuffle=True,
                            seed=cfg.seed,
                        )
                        shuffle_flag = False
                    else:
                        sampler = None
                        shuffle_flag = True

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        sampler=sampler,
                        shuffle=shuffle_flag,
                        pin_memory=True,
                        persistent_workers=(num_workers > 0),
                    )

                    timer_callback = DataloaderTimer()
                    # initialize the model
                    model = BenchmarkModule(
                        image_size=tuple(cfg.data.image_size),
                        num_classes=cfg.data.num_classes,
                    ).to("cuda" if gpus else "cpu")
                    # Run a dummy forward pass to initialize lazy modules before DDP setup
                    with torch.no_grad():
                        dummy = torch.randn(
                            2, *cfg.data.image_size, device=model.device
                        )
                        model(dummy)
                    trainer = Trainer(
                        max_epochs=cfg.trainer.max_epochs,
                        accelerator="gpu" if gpus else "cpu",
                        devices=list(gpus) if gpus else 1,
                        strategy="ddp" if is_ddp else "auto",
                        use_distributed_sampler=False if sampler is not None else True,
                        callbacks=[timer_callback],
                        logger=wandb_logger,
                    )

                    # Start the training and benchmarking
                    trainer.fit(model, train_loader)

                    # Finish the wandb run
                    rank_zero_only(wandb.finish)()  # clean finish only on rank-0


if __name__ == "__main__":
    main()
