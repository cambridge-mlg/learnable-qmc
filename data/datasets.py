from typing import Tuple, Optional
import os

import wget
import pandas as pd
import tensorflow as tf
from ucimlrepo import fetch_ucirepo

from lqmc.random import Seed, randperm
from lqmc.utils import to_tensor, f64


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

    mean = (
        tf.reduce_mean(tensor, axis=0, keepdims=True) if mean is None else mean
    )
    std = (
        tf.math.reduce_std(tensor, axis=0, keepdims=True)
        if stddev is None
        else stddev
    )
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
        dtype: tf.DType,
    ):

        features, targets = self.get_raw_inputs_and_outputs()
        seed, self._x = _shuffle_tensor_along_first_dim(
            seed,
            to_tensor(features, dtype),
        )
        seed, self._y = _shuffle_tensor_along_first_dim(
            seed,
            to_tensor(targets, dtype),
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

    def get_raw_inputs_and_outputs(self) -> Tuple[tf.Tensor, tf.Tensor]:
        data = fetch_ucirepo(id=self.uci_id).data
        return data.features, data.targets


class UCICSVDataset(UCIDataset):
    url_to_tar: str
    name: str

    def get_raw_inputs_and_outputs(self) -> Tuple[tf.Tensor, tf.Tensor]:

        if not os.path.exists(f"_data"):
            raise RuntimeError(
                f"Expected directory '_data' to exist where this script is "
                f"run from, but this was not found. Please run 'mkdir _data'."
            )

        if not os.path.exists(f"_data/zip"):
            os.makedirs("_data/zip", exist_ok=True)

        if not os.path.exists(f"_data/zip/{self.name}"):
            file = wget.download(
                self.url_to_tar, out=f"_data/zip/{self.name}.zip"
            )

        if not os.path.exists(f"_data/{self.name}"):
            os.system(f"unzip _data/zip/{self.name}.zip -d _data/{self.name}")

        return self.csv_to_features_and_targets()

    def csv_to_features_and_targets(self) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError


class ConcreteCompressiveStrength(UCIDataset):
    uci_id: int = 165
    name: str = "concrete"


class WineRed(UCIDataset):
    uci_id: int = 186
    name: str = "wine"

    def get_raw_inputs_and_outputs(self) -> Tuple[tf.Tensor, tf.Tensor]:
        data = fetch_ucirepo(id=self.uci_id).data
        is_red = data["original"]["color"] == "red"
        return data.features[is_red], data.targets[is_red]


class PowerPlant(UCIDataset):
    uci_id: int = 294
    name: str = "power"


class Superconductivity(UCICSVDataset):
    url_to_tar: str = (
        "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
    )
    name: str = "superconductivity"

    def csv_to_features_and_targets(self) -> Tuple[tf.Tensor, tf.Tensor]:
        df = pd.read_csv(f"_data/{self.name}/train.csv")
        return df.iloc[:, :-1].values, df.iloc[:, -1:].values


class Yacht(UCICSVDataset):
    url_to_tar: str = (
        "https://archive.ics.uci.edu/static/public/243/yacht+hydrodynamics.zip"
    )
    name: str = "yacht"

    def csv_to_features_and_targets(self) -> Tuple[tf.Tensor, tf.Tensor]:
        os.listdir(f"_data/{self.name}")
        df = pd.read_csv(
            f"_data/{self.name}/yacht_hydrodynamics.data",
            header=None,
            sep="\s+",
        )
        return df.iloc[:, :-1].values, df.iloc[:, -1:].values


class Airfoil(UCICSVDataset):
    url_to_tar: str = (
        "https://archive.ics.uci.edu/static/public/291/airfoil+self+noise.zip"
    )
    name: str = "airfoil"

    def csv_to_features_and_targets(self) -> Tuple[tf.Tensor, tf.Tensor]:
        df = pd.read_csv(
            f"_data/{self.name}/airfoil_self_noise.dat",
            header=None,
            sep="\t",
        )
        return df.iloc[:, :-1].values, df.iloc[:, -1:].values


class BostonHousing(UCIDataset):
    url_to_csv: str = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
    )
    name: str = "boston"

    def get_raw_inputs_and_outputs(self) -> Tuple[tf.Tensor, tf.Tensor]:

        if not os.path.exists(f"_data/{self.name}"):
            os.mkdir(f"_data/{self.name}")
            wget.download(self.url_to_csv, out=f"_data/{self.name}/boston.csv")

        return self.csv_to_features_and_targets()

    def csv_to_features_and_targets(self) -> Tuple[tf.Tensor, tf.Tensor]:
        df = pd.read_csv(
            f"_data/{self.name}/boston.csv",
            header=None,
            sep="\s+",
        )
        return df.iloc[:, :-1].values, df.iloc[:, -1:].values
