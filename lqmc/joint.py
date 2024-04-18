from abc import abstractmethod
from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors

from lqmc.random import Seed, randn, randu, mvn_chol, rand_halton
from lqmc.utils import ortho_frame, ortho_anti_frame


class JointDistribution(tfk.Model):
    def __init__(
        self,
        dim: int,
        num_points: int,
        name: str = "joint_distribution",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.num_points = num_points

    @abstractmethod
    def __call__(self, seed: Seed, batch_size: int) -> tf.Tensor:
        pass


class IndependentUniform(JointDistribution):
    def __init__(
        self,
        dim: int,
        num_points: int,
        name: str = "independent_uniform",
        **kwargs,
    ):
        super().__init__(dim=dim, name=name, num_points=num_points, **kwargs)

    def __call__(self, seed: Seed, batch_size: int) -> tf.Tensor:
        return randn(
            seed=seed,
            shape=(batch_size, self.num_points, self.dim),
            mean=tf.zeros((), dtype=self.dtype),
            stddev=tf.ones((), dtype=self.dtype),
        )


class HaltonSequence(JointDistribution):
    def __init__(
        self,
        dim: int,
        num_points: int,
        name: str = "halton_sequence",
        **kwargs,
    ):
        super().__init__(name=name, num_points=num_points, dim=dim, **kwargs)

    def __call__(self, seed: Seed, batch_size: int) -> tf.Tensor:
        seed, omega_uniform = rand_halton(
            seed=seed,
            shape=(batch_size,),
            num_samples=self.num_points,
            dim=self.dim,
            dtype=self.dtype,
        )
        omega = tfd.Normal(
            loc=tf.zeros((), dtype=self.dtype),
            scale=tf.ones((), dtype=self.dtype),
        ).quantile(omega_uniform)

        return seed, omega


class GaussianCopula(JointDistribution):
    def __init__(
        self,
        dim: int,
        inverse_cdf: Callable,
        frame_type: str,
        num_points: int,
        name="gaussian_copula",
        **kwargs,
    ):

        super().__init__(dim=dim, name=name, num_points=num_points, **kwargs)

        if frame_type == "ortho":
            self.get_frame = ortho_frame

        elif frame_type == "ortho_anti":
            self.get_frame = ortho_anti_frame

        else:
            raise ValueError(
                f"frame_type must be one of ortho, ortho_anti, "
                f"got {frame_type}."
            )

        self.inverse_cdf = inverse_cdf

    @property
    def cholesky(self) -> tf.Tensor:
        raise NotImplementedError

    @property
    def covariance(self) -> tf.Tensor:
        return tf.matmul(self.cholesky, self.cholesky, transpose_b=True)

    def __call__(self, seed: Seed, batch_size: int) -> tf.Tensor:

        seed, samples = mvn_chol(
            seed=seed,
            mean=tf.zeros((batch_size, self.num_points), dtype=self.dtype),
            cov_chol=self.cholesky,
        )

        omega_norms = self.inverse_cdf(
            tfd.Normal(
                loc=tf.zeros((), dtype=self.dtype),
                scale=tf.ones((), dtype=self.dtype),
            ).cdf(samples)
        )

        frame = self.get_frame(
            dim=self.dim,
            num_points=self.num_points,
            dtype=self.dtype,
        )
        omega = omega_norms[:, :, None] * frame[None, :, :]

        return seed, omega


class GaussianCopulaParametrised(GaussianCopula):
    def __init__(
        self,
        seed: Seed,
        dim: int,
        inverse_cdf: Callable,
        frame_type: str,
        num_points: int,
        trainable: bool = True,
        name="gaussian_copula_parameterised",
        **kwargs,
    ):

        super().__init__(
            dim=dim,
            name=name,
            inverse_cdf=inverse_cdf,
            num_points=num_points,
            frame_type=frame_type,
            **kwargs,
        )

        num_thetas = num_points * (num_points - 1) // 2
        _, thetas_init = randn(
            (num_thetas,),
            seed=seed,
            mean=tf.zeros((num_thetas), dtype=self.dtype),
            stddev=1e-1 * tf.ones((num_thetas), dtype=self.dtype),
        )
        self.thetas = tf.Variable(
            thetas_init,
            dtype=self.dtype,
            trainable=trainable,
        )
        self.bijector = tfp.bijectors.CorrelationCholesky()

    @property
    def cholesky(self) -> tf.Tensor:
        return self.bijector(self.thetas)


class GaussianCopulaAntiparallelCoupled(GaussianCopula):
    def __init__(
        self,
        dim: int,
        correlation_factor: float,
        inverse_cdf: Callable,
        frame_type: str,
        num_points: int,
        name="gaussian_copula_antiparallel_coupled",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            name=name,
            inverse_cdf=inverse_cdf,
            frame_type=frame_type,
            num_points=num_points,
            **kwargs,
        )

        self.num_pairs = self.num_points // 2
        self.num_spares = self.num_points % 2
        eye = tf.eye(self.num_pairs, dtype=self.dtype)
        cov_block1 = tf.concat([eye, correlation_factor * eye], axis=0)
        cov_block2 = tf.concat([correlation_factor * eye, eye], axis=0)

        # Covariance matrix between pairs
        self._covariance = tf.concat([cov_block1, cov_block2], axis=1)

        # Zeros for spare in covariance matrix
        if self.num_spares == 1:

            spare_row = tf.zeros(
                (self.num_spares, 2 * self.num_pairs), dtype=self.dtype
            )
            spare_col = tf.concat(
                [
                    tf.zeros((2 * self.num_pairs, self.num_spares), dtype=self.dtype),
                    tf.ones((self.num_spares, self.num_spares), dtype=self.dtype),
                ],
                axis=0,
            )
            self._covariance = tf.concat([self._covariance, spare_row], axis=0)
            self._covariance = tf.concat([self._covariance, spare_col], axis=1)
            

        self._cholesky = tf.linalg.cholesky(self._covariance)

    @property
    def cholesky(self) -> tf.Tensor:
        return self._cholesky

    @property
    def covariance(self) -> tf.Tensor:
        return self._covariance


class GaussianCopulaAntiparallelCorrelated(GaussianCopulaAntiparallelCoupled):
    def __init__(
        self,
        dim: int,
        inverse_cdf: Callable,
        frame_type: str,
        num_points: int,
        name="gaussian_copula_antiparallel_correlated",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            correlation_factor=1 - 1e-2,
            inverse_cdf=inverse_cdf,
            frame_type=frame_type,
            num_points=num_points,
            name=name,
            **kwargs,
        )


class GaussianCopulaAntiparallelUncorrelated(
    GaussianCopulaAntiparallelCoupled
):
    def __init__(
        self,
        dim: int,
        inverse_cdf: Callable,
        frame_type: str,
        num_points: int,
        name="gaussian_copula_antiparallel_uncorrelated",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            correlation_factor=0.0,
            inverse_cdf=inverse_cdf,
            frame_type=frame_type,
            num_points=num_points,
            name=name,
            **kwargs,
        )


class GaussianCopulaAntiparallelAnticorrelated(
    GaussianCopulaAntiparallelCoupled
):
    def __init__(
        self,
        dim: int,
        inverse_cdf: Callable,
        frame_type: str,
        num_points: int,
        name="gaussian_copula_antiparallel_anticorrelated",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            correlation_factor=-(1 - 1e-2),
            inverse_cdf=inverse_cdf,
            frame_type=frame_type,
            num_points=num_points,
            name=name,
            **kwargs,
        )
