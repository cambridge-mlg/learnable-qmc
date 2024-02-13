from abc import ABC, abstractmethod
from typing import Optional, List

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfd = tfp.distributions

from lqmc.utils import to_tensor
from lqmc.random import Seed


class Kernel(ABC, tfk.Model):
    def __init__(
        self,
        dim: int,
        name="kernel",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        self.dim = dim

    def call(
        self, *, x1: tf.Tensor, x2: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Computes the value of the kernel on the pair `x1`, `x2`.
        If `x2` is `None`, the kernel is evaluated at `x1` with itself.

        Arguments:
            x1: tensor, shape `(..., n, dim)`.
            x2: tensor, shape `(..., m, dim)` or `None`.

        Returns:
            tensor, shape `(..., n, m)`.
        """
        return self.k(x1, x2 if x2 is not None else x1)

    @abstractmethod
    def k(self, *, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        """Computes the value of the kernel on the pair `x1`, `x2`.

        Arguments:
            x1: tensor, shape `(..., n, dim)`.
            x2: tensor, shape `(..., m, dim)`.

        Returns:
            tensor, shape `(..., n, m)`.
        """
        pass

    @abstractmethod
    def make_features(self, *, x: tf.Tensor, **kwargs) -> tf.Tensor:
        """Computes the features of `x`.

        Arguments:
            x: tensor, shape `(..., n, dim)`.
            kwargs: additional keyword arguments.

        Returns:
            phi: tensor, shape `(..., n, n_features)`.
        """
        pass

    def rmse_loss(
        self,
        *,
        x1: tf.Tensor,
        x2: Optional[tf.Tensor] = None,
        **kwargs,
    ) -> tf.Tensor:
        """Computes the loss between the pair `x1`, `x2`.

        Arguments:
            x1: tensor, shape `(..., n, dim)`.
            x2: tensor, shape `(..., m, dim)`.
            kwargs: additional keyword arguments.

        Returns:
            tensor, shape `(..., n, m)`.
        """

        x2 = x2 if x2 is not None else x1
        assert x1.shape[:-2] == x2.shape[:-2] and x1.shape[-1] == x2.shape[-1]

        f1 = self.make_features(x=x1, **kwargs)
        f2 = self.make_features(x=x2, **kwargs)

        k_approx = tf.matmul(f1, f2, transpose_b=True)
        k_true = self.k(x1=x1, x2=x2)

        return tf.reduce_mean(tf.square(k_approx - k_true))


class StationaryKernel(Kernel):
    def __init__(
        self,
        dim: int,
        lengthscales: List[float],
        name="eq_kernel",
        **kwargs,
    ):
        super().__init__(dim=dim, name=name, **kwargs)

        # Check length scales are positive
        assert all([l > 0 for l in lengthscales])

        # Initialise log length scales
        self.log10_lengthscales = tf.Variable(
            tf.math.log(to_tensor(lengthscales, dtype=self.dtype)),
            dtype=float,
        )

    @property
    def lengthscales(self) -> tf.Tensor:
        return tf.exp(self.log10_lengthscales)

    @abstractmethod
    def rbf(self, *, r: tf.Tensor) -> tf.Tensor:
        pass

    def k(self, *, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        """Computes the value of the kernel on the pair `x1`, `x2`.

        Arguments:
            x1: tensor, shape `(..., n, dim)`.
            x2: tensor, shape `(..., m, dim)`.

        Returns:
            tensor, shape `(..., n, m)`.
        """
        lengthscale = tf.reshape(self.lengthscales, len(x1.shape) * [1] + [-1])
        diff = (x1[..., :, None, :] - x2[..., None, :, :]) / lengthscale
        return self.rbf(tf.reduce_sum(tf.square(diff), axis=-1) ** 0.5)

    def make_features(
        self,
        *,
        x: tf.Tensor,
        omega: tf.Tensor,
        rotation: tf.Tensor,
    ) -> tf.Tensor:
        """Computes the features of `x`.

        Arguments:
            x: tensor, shape `(..., n, dim)`.
            omega: tensor, shape `(..., dim, n_features)`.
            rotation: tensor, shape `(..., dim, dim)`.

        Returns:
            phi: tensor, shape `(..., n, 2*n_features)`.
        """
        lengthscale = tf.reshape(self.lengthscales, len(x.shape) * [1] + [-1])
        x = tf.matmul(rotation, x / lengthscale)

        inner_prod = tf.einsum("...nd,...df->...nf", x, omega)

        features = tf.stack(
            [
                tf.cos(inner_prod),
                tf.sin(inner_prod),
            ],
            axis=-1,
        )

        features = tf.reshape(features, features.shape[:-2] + [-1])

        return features


class ExponentiatedQuadraticKernel(StationaryKernel):

    def rbf(self, *, r: tf.Tensor) -> tf.Tensor:
        return tf.exp(-0.5 * r**2)

    @classmethod
    def rff_inverse_cdf(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.sqrt(tfd.Chi2(df=self.dim).quantile(tensor))
