import math
from typing import Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class ShardSampler(Sampler[int]):
    """
    A custom sampler for shard-based datasets.

    This sampler first shuffles the shards (e.g., Parquet files) and then
    yields indices from one shard at a time until it's exhausted. This is
    useful for large datasets where loading an entire shard into memory is
    more efficient than random access across different files.

    The sampler yields the original DataFrame indices, which can be used with
    `df.loc[]`.

    Args:
        metadata (pd.DataFrame): A DataFrame containing at least a 'file_path'
            column to identify the shards.
        shuffle (bool): If True, the order of shards and indices within each
            shard are shuffled at the beginning of each epoch. Defaults to True.
    """

    def __init__(self, metadata: pd.DataFrame, shuffle: bool = True):
        super().__init__(metadata)
        self.metadata = metadata
        self.shuffle = shuffle

        # Group indices by shard (file_path)
        self.shards = self.metadata.groupby("file_path").indices
        self.shard_keys = list(self.shards.keys())

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over indices, grouped and shuffled by shard.
        """
        shard_keys = self.shard_keys[:]
        if self.shuffle:
            # Shuffle the order of shards for the current epoch
            np.random.shuffle(shard_keys)

        # Concatenate the indices from all shards in the new shuffled order
        all_indices: List[int] = []
        for shard_key in shard_keys:
            # Get the original indices for the current shard
            shard_indices = self.shards[shard_key].copy()
            if self.shuffle:
                # Shuffle indices within the shard
                np.random.shuffle(shard_indices)
            all_indices.extend(shard_indices)

        return iter(all_indices)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.metadata)


class DistributedShardSampler(torch.utils.data.DistributedSampler):
    """
    A distributed sampler for shard-based datasets.

    This sampler assigns a non-overlapping subset of shards to each process.
    This is more efficient for sharded datasets (like multiple Parquet files)
    in a distributed setting because it minimizes the number of files each
    process needs to open and cache.

    It also ensures that each process gets roughly the same number of samples
    by padding or truncating the number of samples.

    Args:
        metadata (pd.DataFrame): DataFrame with a 'file_path' column.
        num_replicas (int, optional): Number of processes. Defaults to world size.
        rank (int, optional): Rank of the current process. Defaults to current rank.
        shuffle (bool): If True, shuffle shards and samples within shards.
        seed (int): Random seed for shuffling.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if num_replicas is None:
            if not dist.is_available() or not dist.is_initialized():
                num_replicas = 1
            else:
                num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available() or not dist.is_initialized():
                rank = 0
            else:
                rank = dist.get_rank()

        self.metadata = metadata
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed

        self.shards = self.metadata.groupby("file_path").indices
        self.shard_keys = sorted(list(self.shards.keys()))
        super().__init__(metadata, num_replicas, rank, shuffle=shuffle, drop_last=False)

        self.total_size = len(self.metadata)
        self.num_samples = math.ceil(self.total_size / self.num_replicas)
        self.total_sampled_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            shard_order = torch.randperm(len(self.shard_keys), generator=g).tolist()
        else:
            shard_order = list(range(len(self.shard_keys)))

        # Distribute shards to replicas
        shards_for_rank_indices = shard_order[self.rank :: self.num_replicas]
        shards_for_rank_keys = [self.shard_keys[i] for i in shards_for_rank_indices]

        # Get all indices for this rank's shards
        indices = []
        for key in shards_for_rank_keys:
            shard_indices = self.shards[key].copy()
            if self.shuffle:
                np.random.shuffle(shard_indices)
            indices.extend(shard_indices)

        # Pad or truncate to ensure all ranks have the same number of samples
        if len(indices) > self.num_samples:
            indices = indices[: self.num_samples]
        else:
            padding_size = self.num_samples - len(indices)
            if padding_size > 0 and len(indices) > 0:
                # indices += np.random.choice(indices, size=padding_size).tolist()
                indices += (
                    np.random.RandomState(self.seed + self.epoch)
                    .choice(indices, size=padding_size)
                    .tolist()
                )
        # self.set_epoch(self.epoch)  # PL will update epoch for us
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int):
        self.epoch = epoch
