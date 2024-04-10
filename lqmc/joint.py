from abc import abstractmethod
from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors

from lqmc.random import Seed, randn, randu, mvn_chol, rand_halton
from lqmc.utils import orthonormal_frame


class JointDistribution(tfk.Model):
    def __init__(
        self,
        dim: int,
        name: str = "joint_distribution",
        **kwargs,
    ):
        self.dim = dim
        super().__init__(name=name, **kwargs)

    @abstractmethod
    def __call__(self, seed: Seed, batch_size: int) -> tf.Tensor:
        pass


class IndependentUniform(JointDistribution):
    def __init__(self, dim: int, name: str = "independent_uniform", **kwargs):
        super().__init__(dim=dim, name=name, **kwargs)

    def __call__(self, seed: Seed, batch_size: int) -> tf.Tensor:
        return randn(
            seed=seed,
            shape=(batch_size, 2 * self.dim, self.dim),
            mean=tf.zeros((), dtype=self.dtype),
            stddev=tf.ones((), dtype=self.dtype),
        )

class HaltonSequence(tfk.Model):
    def __init__(self, dim: int, name: str = "halton_sequence", **kwargs):
        self.dim = dim
        super().__init__(name=name, **kwargs)

    def __call__(self, seed: Seed, batch_size: int) -> tf.Tensor:
        seed, omega_uniform = rand_halton(
            seed=seed,
            shape=(batch_size,),
            num_samples=2 * self.dim,
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
        name="gaussian_copula",
        **kwargs,
    ):
        super().__init__(dim=dim, name=name, **kwargs)
        self.inverse_cdf = inverse_cdf

    @property
    def cholesky(self) -> tf.Tensor:
        return self.bijector(self.thetas)

    @property
    def covariance(self) -> tf.Tensor:
        return tf.matmul(self.cholesky, self.cholesky, transpose_b=True)

    def __call__(self, seed: Seed, batch_size: int) -> tf.Tensor:

        seed, samples = mvn_chol(
            seed=seed,
            mean=tf.zeros((batch_size, 2 * self.dim), dtype=self.dtype),
            cov_chol=self.cholesky,
        )

        omega_norms = self.inverse_cdf(
            tfd.Normal(
                loc=tf.zeros((), dtype=self.dtype),
                scale=tf.ones((), dtype=self.dtype),
            ).cdf(samples)
        )

        frame = orthonormal_frame(
            dim=self.dim,
            num_pairs=self.dim,
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
        trainable: bool = True,
        name="gaussian_copula_parameterised",
        **kwargs,
    ):

        super().__init__(dim=dim, name=name, inverse_cdf=inverse_cdf, **kwargs)

        num_thetas = 2 * dim * (2 * dim - 1) // 2
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

    @property
    def covariance(self) -> tf.Tensor:
        return tf.matmul(self.cholesky, self.cholesky, transpose_b=True)


class GaussianCopulaAntiparallelCoupled(GaussianCopula):
    def __init__(
        self,
        dim: int,
        correlation_factor: float,
        inverse_cdf: Callable,
        name="gaussian_copula_antiparallel_coupled",
        **kwargs,
    ):
        super().__init__(dim=dim, name=name, inverse_cdf=inverse_cdf, **kwargs)

        eye = tf.eye(dim, dtype=self.dtype)
        cov_block1 = tf.concat([eye, correlation_factor * eye], axis=0)
        cov_block2 = tf.concat([correlation_factor * eye, eye], axis=0)

        self._covariance = tf.concat([cov_block1, cov_block2], axis=1)
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
        name="gaussian_copula_antiparallel_correlated",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            correlation_factor=1 - 1e-2,
            inverse_cdf=inverse_cdf,
            name=name,
            **kwargs,
        )


class GaussianCopulaAntiparallelUncorrelated(GaussianCopulaAntiparallelCoupled):
    def __init__(
        self,
        dim: int,
        inverse_cdf: Callable,
        name="gaussian_copula_antiparallel_uncorrelated",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            correlation_factor=0.0,
            inverse_cdf=inverse_cdf,
            name=name,
            **kwargs,
        )


class GaussianCopulaAntiparallelAnticorrelated(GaussianCopulaAntiparallelCoupled):
    def __init__(
        self,
        dim: int,
        inverse_cdf: Callable,
        name="gaussian_copula_antiparallel_anticorrelated",
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            correlation_factor=-(1 - 1e-2),
            inverse_cdf=inverse_cdf,
            name=name,
            **kwargs,
        )
