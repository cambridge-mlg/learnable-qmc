import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors

from lqmc.random import Seed, randn, mvn_chol


class GaussianCopula(tfk.Model):
    def __init__(
        self,
        dim: int,
        name="gaussian_copula",
        **kwargs,
    ):

        super().__init__(name=name, **kwargs)

        self.dim = dim

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
            mean=tf.zeros((batch_size, 2 * self.dim), dtype=self.dtype),
            cov_chol=self.cholesky,
        )

        norm = tfd.Normal(
            loc=tf.zeros((), dtype=self.dtype),
            scale=tf.ones((), dtype=self.dtype),
        )

        return seed, norm.cdf(samples)


class GaussianCopulaParametrised(GaussianCopula):
    def __init__(
        self,
        seed: Seed,
        dim: int,
        trainable: bool,
        name="gaussian_copula_parameterised",
        **kwargs,
    ):

        super().__init__(dim=dim, name=name, **kwargs)

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


class GaussianCopulaAntiparallelCorrelated(GaussianCopula):
    def __init__(
        self,
        dim: int,
        correlation_factor: float,
        name="gaussian_copula_antiparallel_correlated",
        **kwargs,
    ):
        super().__init__(dim=dim, name=name, **kwargs)

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
