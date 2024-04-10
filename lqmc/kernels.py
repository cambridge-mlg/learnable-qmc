from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfd = tfp.distributions

from lqmc.utils import to_tensor, cast, orthonormal_frame
from lqmc.random import Seed, rand_unitary


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
        self,
        x1: tf.Tensor,
        x2: Optional[tf.Tensor] = None,
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
    def k(
        self,
        x1: tf.Tensor,
        x2: tf.Tensor,
    ) -> tf.Tensor:
        """Computes the value of the kernel on the pair `x1`, `x2`.

        Arguments:
            x1: tensor, shape `(..., n, dim)`.
            x2: tensor, shape `(..., m, dim)`.

        Returns:
            tensor, shape `(..., n, m)`.
        """
        pass

    @abstractmethod
    def make_features(
        self,
        x: tf.Tensor,
        **kwargs,
    ) -> tf.Tensor:
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
        seed: Seed,
        omega: tf.Tensor,
        x1: tf.Tensor,
        x2: Optional[tf.Tensor] = None,
        apply_rotation: bool = True,
    ) -> tf.Tensor:
        """Computes the loss between the pair `x1`, `x2`.

        Arguments:
            x1: tensor, shape `(..., n, dim)`.
            x2: tensor, shape `(..., m, dim)`.
            kwargs: additional keyword arguments.

        Returns:
            tensor, shape `(,)`.
        """

        x2 = x2 if x2 is not None else x1

        if apply_rotation:
            seed, rotation = rand_unitary(
                seed=seed,
                shape=(omega.shape[0],),
                dim=self.dim,
                dtype=self.dtype,
            )

        else:
            rotation = tf.eye(self.dim, dtype=self.dtype)[None, :, :]
            rotation = tf.tile(rotation, (omega.shape[0], 1, 1))
        
        f1 = self.make_features(x=x1, omega=omega, rotation=rotation)
        f2 = self.make_features(x=x2, omega=omega, rotation=rotation)

        k_approx = tf.reduce_mean(tf.matmul(f1, f2, transpose_b=True), axis=0)
        k_true = self.k(x1=x1, x2=x2)

        return seed, tf.reduce_mean((k_approx - k_true) ** 2.0) ** 0.5


class StationaryKernel(Kernel):
    def __init__(
        self,
        dim: int,
        lengthscales: List[float],
        output_scale: float = 1.0,
        name: str = "eq_kernel",
        **kwargs,
    ):
        super().__init__(dim=dim, name=name, **kwargs)

        # Check length scales are positive
        assert all([l > 0 for l in lengthscales])

        # Initialise log length scales
        self.log_lengthscales = tf.Variable(
            tf.math.log(
                to_tensor(
                    lengthscales,
                    dtype=self.dtype,
                )
            ),
        )

        self.log_output_scale = tf.Variable(
            tf.math.log(to_tensor(output_scale, dtype=self.dtype)),
        )

    @property
    def lengthscales(self) -> tf.Tensor:
        return tf.exp(self.log_lengthscales)

    @property
    def output_scale(self) -> tf.Tensor:
        return tf.exp(self.log_output_scale)

    @abstractmethod
    def rbf(self, r: tf.Tensor) -> tf.Tensor:
        pass

    def k(self, x1: tf.Tensor, x2: tf.Tensor) -> tf.Tensor:
        """Computes the value of the kernel on the pair `x1`, `x2`.

        Arguments:
            x1: tensor, shape `(..., n, dim)`.
            x2: tensor, shape `(..., m, dim)`.

        Returns:
            tensor, shape `(..., n, m)`.
        """
        assert x1.shape[-1] == self.dim and x2.shape[-1] == self.dim
        lengthscales = tf.reshape(self.lengthscales, len(x1.shape) * [1] + [-1])
        diff = (x1[..., :, None, :] - x2[..., None, :, :]) / (lengthscales + 1e-6)
        r2 = tf.reduce_sum(diff ** 2., axis=-1)
        return self.output_scale**2. * self.rbf(r2=r2)

    def make_features(
        self,
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

        lengthscale = tf.reshape(
            self.lengthscales, (len(x.shape) - 1) * [1] + [-1]
        )
        x = x / lengthscale
        x = tf.einsum("sij, nj -> sni", rotation, x)

        inner_prod = tf.einsum("sfi, sni -> snf", omega, x)

        features = tf.stack(
            [
                tf.cos(inner_prod),
                tf.sin(inner_prod),
            ],
            axis=-1,
        )

        s = features.shape
        features = tf.reshape(features, (s[0], s[1], -1))

        return (
            features
            / tf.sqrt(cast(features.shape[-1] // 2, features.dtype))
            * self.output_scale
        )


class ExponentiatedQuadraticKernel(StationaryKernel):

    def rbf(self, r2: tf.Tensor) -> tf.Tensor:
        return tf.exp(-0.5 * r2)

    def rff_inverse_cdf(self, tensor: tf.Tensor) -> tf.Tensor:
        df = to_tensor(self.dim, dtype=self.dtype)
        return tf.sqrt(tfd.Chi2(df=df).quantile(tensor))
