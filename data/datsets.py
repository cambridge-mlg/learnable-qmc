from typing import Tuple, Optional

import tensorflow as tf
from ucimlrepo import fetch_ucirepo

from lqmc.random import Seed, randperm
from lqmc.utils import to_tensor, f64, i32


def _shuffle_tensor_along_first_dim(
    seed: Seed, tensor: tf.Tensor
) -> tf.Tensor:
    seed, idx = randperm(seed=seed, shape=(), maxval=tf.shape(tensor)[0] - 1)
    return seed, tf.gather(tensor, idx, axis=0)


def _split_train_test(
    tensor: tf.Tensor, split_id: int, num_splits: int
) -> Tuple[tf.Tensor, tf.Tensor]:
    # Compute total size of tensor and the size of each split
    total_size = tf.shape(tensor)[0]
    split_size = total_size // num_splits

    # Determine the start and end indices of the test split
    test_start_idx = split_id * split_size
    test_end_idx = test_start_idx + split_size

    # Create the train split and then the test split
    train_split = tf.concat(
        [tensor[:test_start_idx], tensor[test_end_idx:]], axis=0
    )
    test_split = tensor[test_start_idx:test_end_idx]

    return train_split, test_split


def _normalise(
    tensor: tf.Tensor,
    mean: Optional[tf.Tensor] = None,
    stddev: Optional[tf.Tensor] = None,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

    mean = tf.reduce_mean(tensor) if mean is None else mean
    std = tf.math.reduce_std(tensor) if stddev is None else stddev
    normalised_tensor = (tensor - mean) / std

    return normalised_tensor, mean, std


class UCIDataset:

    x_train: tf.Tensor
    y_train: tf.Tensor

    x_mean: tf.Tensor
    y_mean: tf.Tensor

    x_test: tf.Tensor
    y_test: tf.Tensor

    uci_id: int

    def __init__(
        self,
        seed: Seed,
        split_id: int,
        num_splits: int,
    ):

        self.data = fetch_ucirepo(id=self.uci_id).data
        seed, self._x = _shuffle_tensor_along_first_dim(
            seed,
            to_tensor(self.data.features, f64),
        )
        seed, self._y = _shuffle_tensor_along_first_dim(
            seed,
            to_tensor(self.data.targets, f64),
        )

        self.x_train, self.x_test = _split_train_test(
            self._x,
            split_id=split_id,
            num_splits=num_splits,
        )

        self.y_train, self.y_test = _split_train_test(
            self._y,
            split_id=split_id,
            num_splits=num_splits,
        )

        self.x_train, self.x_mean, self.x_std = _normalise(self.x_train)
        self.y_train, self.y_mean, self.y_std = _normalise(self.y_train)

        self.x_test, _, _ = _normalise(
            self.x_test,
            mean=self.x_mean,
            stddev=self.x_std,
        )

        self.y_test, _, _ = _normalise(
            self.y_test,
            mean=self.y_mean,
            stddev=self.y_std,
        )


class ConcreteCompressiveStrength(UCIDataset):
    uci_id: int = 165
