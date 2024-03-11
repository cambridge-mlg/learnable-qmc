from typing import Callable, Tuple, Optional

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfd = tfp.distributions

from lqmc.kernels import Kernel
from lqmc.utils import to_tensor


def zero_mean(x: tf.Tensor) -> tf.Tensor:
    return tf.zeros(x.shape[:-1], dtype=x.dtype)


class GaussianProcess(tfk.Model):
    def __init__(
        self,
        *,
        kernel: Kernel,
        noise_std: float,
        x: tf.Tensor,
        y: tf.Tensor,
        mean_function: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        name="gaussian_process",
        **kwargs,
    ):

        super().__init__(name=name, **kwargs)

        assert tf.rank(x) == 2
        assert tf.rank(y) == 1
        assert x.shape[0] == y.shape[0]

        self.kernel = kernel
        self.noise_variance = tf.Variable(
            to_tensor(noise_std, dtype=self.dtype) ** 2.0
        )
        self.x_train = x
        self.y_train = y
        self.mean_function = mean_function if mean_function else zero_mean

    def call(
        self,
        x_pred: tf.Tensor,
        noiseless: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        ktp = self.kernel(self.x_train, x_pred)
        kpp = self.kernel(x_pred, x_pred)
        ktt = self.kernel(self.x_train, self.x_train)
        kttn = (
            ktt
            + tf.eye(
                tf.shape(self.x_train)[0],
                dtype=ktt.dtype,
            )
            * self.noise_variance
        )

        kttn_chol = tf.linalg.cholesky(kttn)

        # Compute posterior mean
        mean = tf.linalg.matmul(
            ktp,
            tf.linalg.cholesky_solve(kttn_chol, self.y_train[:, None]),
            transpose_a=True,
        )[:, 0]

        # Compute posterior covariance
        cov = kpp - tf.linalg.matmul(
            ktp,
            tf.linalg.cholesky_solve(kttn_chol, ktp),
            transpose_a=True,
        )
        if not noiseless:
            cov += self.noise_variance * tf.eye(
                tf.shape(x_pred)[0],
                dtype=cov.dtype,
            )

        return mean, cov

    def loss(self) -> tf.Tensor:
        # Compute prior mean and covariance
        mean = self.mean_function(self.x_train)
        cov = self.kernel(self.x_train, self.x_train)

        # Compute negative log marginal likelihood
        predictive = tfd.MultivariateNormalFullCovariance(
            loc=mean,
            covariance_matrix=cov
            + self.noise_variance
            * tf.eye(
                tf.shape(self.x_train)[0],
                dtype=cov.dtype,
            ),
        )

        return -predictive.log_prob(self.y_train)



class RandomFeatureGaussianProcess(GaussianProcess):
    def __init__(
        self,
        *,
        kernel: Kernel,
        noise_std: float,
        x: tf.Tensor,
        y: tf.Tensor,
        mean_function: Optional[Callable[[tf.Tensor], tf.Tensor]] = None,
        name="random_feature_gaussian_process",
        **kwargs,
    ):

        super().__init__(
            kernel=kernel,
            noise_std=noise_std,
            x=x,
            y=y,
            mean_function=mean_function,
            name=name,
            **kwargs,
        )
