import math
from typing import Iterator, List, Optional

import numpy as np
import torch.utils.data.sampler
from torch_geometric.data import Dataset


class OrderedSampler(torch.utils.data.sampler.Sampler):
    """
    Deterministic sampler for DDP training with resume support.
    Work both for single / multi GPU.
    For single GPU, use world_size=1, rank=0 (default).
    """

    def __init__(
        self,
        dataset,
        sequence: Optional[List[int]] = None,
        shuffle: bool = False,
        seed: int = 777,
        world_size: int = 1,
        rank: int = 0,
    ):
        if sequence is None:
            self.sequence = np.arange(len(dataset))
        else:
            self.sequence = np.array(sequence)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        assert world_size > 0 and rank < world_size
        self.world_size = world_size
        self.rank = rank

        self.total_samples_per_rank = math.ceil(
            len(self.sequence) / self.world_size
        )
        self.total_size = self.total_samples_per_rank * self.world_size

        self._start_index = 0

    def continue_from_data_progress(
        self,
        numpy_rng_state: dict,
        total_data_num: int = -1,
        current_data_index: int = 0,
    ):
        if numpy_rng_state is not None:
            self.rng.bit_generator.state = numpy_rng_state
        if total_data_num < 0:  # Nothing to continue
            return
        elif total_data_num != len(self.sequence):
            raise ValueError(
                'data_progress not compatible'
                + 'set reset_data_progress: True'
            )
        self._start_index = current_data_index

    def get_rng_state(self):
        return self.rng.bit_generator.state

    def permutate_sequence(self):
        return self.rng.permutation(self.sequence)

    def refresh_sequence(self):
        self._start_index = 0

    def __iter__(self) -> Iterator[int]:
        indices = self.sequence.copy()
        if self.shuffle:
            # deterministically shuffle based on numpy rng state
            indices = self.permutate_sequence()

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            padding_sequence = indices[:padding_size]
        else:
            rep_num = math.ceil(padding_size / len(indices))
            padding_sequence = np.tile(indices, rep_num)[:padding_size]
        indices = np.concatenate((indices, padding_sequence))
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self._start_index + self.rank : self.total_size : self.world_size
        ]
        assert len(indices) == len(self)
        self.refresh_sequence()  # after one epoch, it initializes to 0.

        return iter(indices)

    def __len__(self) -> int:
        current_idx_per_rank = int(self._start_index / self.world_size)
        return self.total_samples_per_rank - current_idx_per_rank


class StratifiedBatchSampler(torch.utils.data.sampler.Sampler):
    """
    Homogeneous batches over groups of a concatenated dataset, each group
    with its own batch size; batch order is shuffled across groups every
    epoch. Built for joint EFS + BEC training: BEC batches stay small (the
    BEC double-backward costs ~3.8x the memory of an E/F/s step) while EFS
    batches stay large.

    DDP-aware: batches are sharded round-robin over ranks, padded so every
    rank yields the same count. The RNG advances on each __iter__, so epochs
    reshuffle deterministically without needing set_epoch() (all ranks share
    the seed and call __iter__ in lockstep).

    Implements the OrderedSampler resume protocol (get_rng_state /
    continue_from_data_progress / permutate_sequence / refresh_sequence) so
    train_by_batch works, with data progress counted in BATCHES, not samples
    (batch sizes differ per group, so sample-count arithmetic cannot locate
    a batch boundary).
    """

    def __init__(
        self,
        group_sizes: List[int],
        batch_sizes: List[int],
        shuffle: bool = True,
        seed: int = 777,
        world_size: int = 1,
        rank: int = 0,
    ):
        assert len(group_sizes) == len(batch_sizes)
        assert all(n > 0 for n in group_sizes)
        assert all(b > 0 for b in batch_sizes)
        assert world_size > 0 and rank < world_size
        offsets = np.cumsum([0] + list(group_sizes))[:-1]
        self.groups = [
            np.arange(n) + off for n, off in zip(group_sizes, offsets)
        ]
        self.batch_sizes = list(batch_sizes)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.world_size = world_size
        self.rank = rank
        n_batches = sum(
            math.ceil(len(g) / b)
            for g, b in zip(self.groups, self.batch_sizes)
        )
        # progress bookkeeping in batch units (analogous to
        # OrderedSampler.sequence, which is in sample units)
        self.sequence = np.arange(n_batches)
        self.batches_per_rank = math.ceil(n_batches / world_size)
        self.total_size = self.batches_per_rank * world_size
        self._start_index = 0

    def _build_batches(self) -> List[np.ndarray]:
        # consumes rng draws when shuffle; every resume of the same epoch
        # must restore the epoch-start rng state before calling
        batches = []
        for group, bsize in zip(self.groups, self.batch_sizes):
            idx = self.rng.permutation(group) if self.shuffle else group
            batches.extend(
                idx[i : i + bsize] for i in range(0, len(idx), bsize)
            )
        if self.shuffle:
            batches = [batches[i] for i in self.rng.permutation(len(batches))]
        while len(batches) < self.total_size:  # pad for even DDP sharding
            batches.extend(batches[: self.total_size - len(batches)])
        return batches

    def get_rng_state(self):
        return self.rng.bit_generator.state

    def continue_from_data_progress(
        self,
        numpy_rng_state: dict,
        total_data_num: int = -1,
        current_data_index: int = 0,
    ):
        if numpy_rng_state is not None:
            self.rng.bit_generator.state = numpy_rng_state
        if total_data_num < 0:  # Nothing to continue
            return
        elif total_data_num != len(self.sequence):
            raise ValueError(
                'data_progress not compatible (batch counts differ; changed '
                'batch_size/bec_batch_size or datasets?) '
                'set reset_data_progress: True'
            )
        self._start_index = current_data_index

    def permutate_sequence(self):
        # advance rng state by exactly one epoch's worth of draws
        return self._build_batches()

    def refresh_sequence(self):
        self._start_index = 0

    def __len__(self) -> int:
        return self.batches_per_rank - self._start_index // self.world_size

    def __iter__(self) -> Iterator[List[int]]:
        batches = self._build_batches()
        selected = batches[
            self._start_index + self.rank : self.total_size : self.world_size
        ]
        self.refresh_sequence()  # after one epoch, it initializes to 0.
        return iter([b.tolist() for b in selected])
