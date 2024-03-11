from typing import Callable

import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors

from lqmc.random import Seed, randn, mvn_chol


class GaussianCopula(tfk.Model):
    def __init__(
        self,
        seed: Seed,
        dim: int,
        target_inverse_cdf: Callable,
        name="gaussian_copula",
        **kwargs,
    ):

        super().__init__(name=name, **kwargs)

        self.dim = dim
        num_thetas = 2 * dim * (2 * dim - 1) // 2
        _, thetas_init = randn(
            (num_thetas,),
            seed=seed,
            mean=tf.zeros((num_thetas), dtype=self.dtype),
            stddev=1e-2 * tf.ones((num_thetas), dtype=self.dtype),
        )
        self.thetas = tf.Variable(thetas_init, dtype=self.dtype)
        self.bijector = tfp.bijectors.CorrelationCholesky()
        self.target_inverse_cdf = target_inverse_cdf

    @property
    def cholesky(self) -> tf.Tensor:
        return self.bijector(self.thetas)

    @property
    def covariance(self) -> tf.Tensor:
        return tf.matmul(self.cholesky, self.cholesky, transpose_b=True)

    def call(
        self,
        seed: Seed,
        batch_size: int,
    ) -> tf.Tensor:

        seed, samples = mvn_chol(
            seed=seed,
            mean=tf.ones((batch_size, 2 * self.dim), dtype=self.dtype),
            cov_chol=self.cholesky,
        )

        norm = tfd.Normal(
            loc=tf.zeros((), dtype=self.dtype),
            scale=tf.ones((), dtype=self.dtype),
        )
        samples = self.target_inverse_cdf(norm.cdf(samples))

        return seed, samples
