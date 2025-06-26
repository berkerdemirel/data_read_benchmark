import time
from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import Callback


class BenchmarkModule(pl.LightningModule):
    """
    A PyTorch Lightning module for benchmarking data loading speeds.

    This module includes a dummy network to process the data. The actual time
    measurement is handled by the DataloaderTimer callback.

    Args:
        image_size (tuple): The size of the input images (C, H, W).
        num_classes (int): The number of classes in the dataset.
    """

    def __init__(self, image_size: tuple, num_classes: int):
        super().__init__()
        self.save_hyperparameters()

        # A simple dummy model
        self.model = nn.Sequential(
            nn.Conv2d(image_size[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Processes a single training batch and computes the loss."""
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return {"loss": loss}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for the model."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class DataloaderTimer(Callback):
    """
    A PyTorch Lightning callback to measure and log data loading performance.

    This callback records the time taken for each training batch and logs the
    average batch time at the end of each epoch.
    """

    def __init__(self):
        super().__init__()
        self.batch_times: List[float] = []
        self.start_time: float = 0

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Records the start time of a training batch."""
        self.start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Records the end time of a training batch and logs the duration."""
        end_time = time.time()
        batch_time = end_time - self.start_time
        self.batch_times.append(batch_time)
        pl_module.log(
            "batch_time", batch_time, on_step=True, on_epoch=False, prog_bar=True
        )

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Records the start time of a training epoch."""
        self.start_time_epoch = time.time()

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """Calculates and logs the average batch time for the epoch."""
        if not self.batch_times:
            return
        total_time = sum(self.batch_times)
        avg_time = total_time / len(self.batch_times)
        pl_module.log("avg_batch_time_per_epoch", avg_time, on_epoch=True)
        epoch_time = time.time() - self.start_time_epoch
        pl_module.log("total_epoch_time", epoch_time, on_epoch=True)
        self.batch_times.clear()
