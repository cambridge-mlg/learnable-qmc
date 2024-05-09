from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfd = tfp.distributions

from lqmc.utils import to_tensor, cast
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
        """
        Arguments:
            x1: tensor, shape `(..., n, dim)`.
            x2: tensor, shape `(..., m, dim)`.
            omega: tensor, shape `(batch_size, 2 * dim, dim)`.
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

        f1 = self.make_features(
            x=x1, omega=omega, rotation=rotation
        )  # (batch_size, 2 * n_features * nx1)
        f2 = self.make_features(
            x=x2, omega=omega, rotation=rotation
        )  # (batch_size, 2 * n_features * nx2)

        k_approx = tf.matmul(f1, f2, transpose_b=True)
        k_true = self.k(x1=x1, x2=x2)

        return seed, tf.reduce_mean((k_approx - k_true) ** 2.0) ** 0.5

    def inverse_rmse_loss(
        self,
        seed: Seed,
        omega: tf.Tensor,
        noise_std: tf.Tensor,
        x: tf.Tensor,
        apply_rotation: bool = True,
    ) -> tf.Tensor:
        """
        Arguments:
            x: tensor, shape `(..., n, dim)`.
            omega: tensor, shape `(batch_size, 2 * dim, dim)`.
            kwargs: additional keyword arguments.

        Returns:
            tensor, shape `(,)`.
        """

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

        f = self.make_features(
            x=x, omega=omega, rotation=rotation
        )  # (batch_size, 2 * n_features * nx)

        k_approx = tf.matmul(f, f, transpose_b=True)
        k_true = self.k(x1=x, x2=x)

        noise = tf.eye(
            tf.shape(x)[0],
            dtype=self.dtype,
        ) * noise_std**2.0

        # Invert k_approx + noise by backsolving with a Cholesky decomposition
        k_approx_inv = tf.linalg.cholesky_solve(
            tf.linalg.cholesky(k_approx + noise),
            tf.eye(tf.shape(x)[0], dtype=self.dtype),
        )

        # Invert k_true + noise by backsolving with a Cholesky decomposition
        k_true_inv = tf.linalg.cholesky_solve(
            tf.linalg.cholesky(k_true + noise),
            tf.eye(tf.shape(x)[0], dtype=self.dtype),
        )

        return seed, tf.reduce_mean((k_approx_inv - k_true_inv) ** 2.0) ** 0.5


class StationaryKernel(Kernel):
    def __init__(
        self,
        dim: int,
        lengthscale: float,
        output_scale: float = 1.0,
        name: str = "eq_kernel",
        **kwargs,
    ):
        super().__init__(dim=dim, name=name, **kwargs)

        # Check length scale is positive
        assert lengthscale > 0.

        # Initialise log length scale
        self.log_lengthscale = tf.Variable(
            tf.math.log(
                to_tensor(
                    lengthscale,
                    dtype=self.dtype,
                )
            ),
        )

        self.log_output_scale = tf.Variable(
            tf.math.log(to_tensor(output_scale, dtype=self.dtype)),
        )

    @property
    def lengthscales(self) -> tf.Tensor:
        return tf.exp(tf.stack(self.dim * [self.log_lengthscale]))

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
        lengthscales = tf.reshape(
            self.lengthscales, len(x1.shape) * [1] + [-1]
        )
        diff = (x1[..., :, None, :] - x2[..., None, :, :]) / (
            lengthscales + 1e-6
        )
        r2 = tf.reduce_sum(diff**2.0, axis=-1)
        return self.output_scale**2.0 * self.rbf(r2=r2)

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

        x = x / self.lengthscales[None, :]
        x = tf.einsum("sij, nj -> sni", rotation, x)

        inner_prod = tf.einsum("sfi, sni -> snf", omega, x)

        features = tf.concat(
            [
                tf.cos(inner_prod),
                tf.sin(inner_prod),
            ],
            axis=-1,
        )  # (s, n, 2f)

        features = tf.transpose(
            features,
            (1, 0, 2),
        )  # (n, s, 2f)

        features = tf.reshape(
            features,
            (features.shape[0], -1),
        )  # (n, 2sf)

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
