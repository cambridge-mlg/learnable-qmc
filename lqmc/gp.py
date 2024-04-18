from typing import Callable, Tuple, Optional

import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfd = tfp.distributions

from lqmc.kernels import Kernel
from lqmc.utils import to_tensor, ortho_frame
from lqmc.random import Seed, rand_unitary


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
        self.noise_log_std = tf.Variable(
            tf.math.log(to_tensor(noise_std, dtype=self.dtype))
        )
        self.x_train = x
        self.y_train = y
        self.mean_function = mean_function if mean_function else zero_mean

    @property
    def noise_std(self) -> tf.Tensor:
        return tf.exp(self.noise_log_std)

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
            * self.noise_std**2.0
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
            cov += self.noise_std**2.0 * tf.eye(
                tf.shape(x_pred)[0],
                dtype=cov.dtype,
            )

        return mean, cov

    def loss(self) -> tf.Tensor:
        # Compute prior mean and covariance
        mean = self.mean_function(self.x_train)
        cov = self.kernel(self.x_train, self.x_train)

        scale_tril = tf.linalg.cholesky(
            cov
            + self.noise_std**2.0
            * tf.eye(
                tf.shape(self.x_train)[0],
                dtype=cov.dtype,
            )
        )

        # Compute negative log marginal likelihood
        predictive = tfd.MultivariateNormalTriL(loc=mean, scale_tril=scale_tril)

        return -predictive.log_prob(self.y_train) / self.x_train.shape[0]

    def pred_nll(self, x_pred: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mean, cov = self(x_pred, noiseless=False)
        predictive = tfd.MultivariateNormalTriL(loc=mean, scale_tril=tf.linalg.cholesky(cov))

        return -predictive.log_prob(y_pred) / x_pred.shape[0]
    

class RandomFeatureGaussianProcess(GaussianProcess):
    def __init__(
        self,
        *,
        joint_sampler: Callable,
        name="random_feature_gaussian_process",
        **kwargs,
    ):

        super().__init__(name=name, **kwargs)
        self.joint_sampler = joint_sampler

    def call(
        self,
        seed: Seed,
        x_pred: tf.Tensor,
        num_ensembles: int,
        noiseless: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        seed, omega = self.joint_sampler(seed=seed, batch_size=num_ensembles)
        omega = self.kernel.rff_inverse_cdf(omega)

        frame = ortho_frame(
            dim=self.kernel.dim,
            num_pairs=self.kernel.dim,
            dtype=self.dtype,
        )

        omega = omega[:, :, None] * frame[None, :, :]
        seed, rotation = rand_unitary(
            seed=seed,
            shape=(num_ensembles,),
            dim=self.kernel.dim,
            dtype=self.dtype,
        )

        num_data = tf.shape(self.x_train)[0]
        x_full = tf.concat([self.x_train, x_pred], axis=0)

        features = self.kernel.make_features(
            x=x_full,
            omega=omega,
            rotation=rotation,
        )
        features = tf.einsum("bnd -> bdn", features) / num_ensembles**0.5
        features = tf.reshape(features, (-1, features.shape[-1]))

        features_data = features[:, :num_data]
        features_pred = features[:, num_data:]

        iS = (
            tf.eye(features_data.shape[0], dtype=self.dtype)
            + tf.linalg.matmul(
                features_data,
                features_data,
                transpose_b=True,
            )
            / self.noise_std**2.0
        )

        # Compute predictive mean
        mean_pred = (
            self.noise_std**-2.0
            * tf.linalg.matmul(
                features_pred,
                tf.linalg.solve(iS, features_data @ self.y_train[:, None]),
                transpose_a=True,
            )[:, 0]
        )

        # Compute predictive covariance
        cov_pred = tf.einsum(
            "fi, fj -> ij",
            features_pred,
            tf.linalg.solve(iS, features_pred),
        )
        if not noiseless:
            cov_pred += (
                tf.eye(
                    tf.shape(x_pred)[0],
                    dtype=cov_pred.dtype,
                )
                * self.noise_std**2.0
            )

        return seed, mean_pred, cov_pred

    @tf.function
    def loss(
        self,
        seed: Seed,
        num_ensembles: int,
    ) -> tf.Tensor:

        seed, omega = self.joint_sampler(seed=seed, batch_size=num_ensembles)
        omega = self.kernel.rff_inverse_cdf(omega)

        frame = ortho_frame(
            dim=self.kernel.dim,
            num_pairs=self.kernel.dim,
            dtype=self.dtype,
        )

        omega = omega[:, :, None] * frame[None, :, :]
        seed, rotation = rand_unitary(
            seed=seed,
            shape=(num_ensembles,),
            dim=self.kernel.dim,
            dtype=self.dtype,
        )

        features = self.kernel.make_features(
            x=self.x_train,
            omega=omega,
            rotation=rotation,
        )
        features = tf.einsum("bnd -> bdn", features) / num_ensembles**0.5
        features = tf.reshape(features, (-1, features.shape[-1]))

        mean = self.mean_function(self.x_train)
        diag_noise = (
            tf.eye(
                tf.shape(self.x_train)[0],
                dtype=self.dtype,
            )
            * self.noise_std**2.0
        )

        predictive = tfd.MultivariateNormalDiagPlusLowRankCovariance(
            loc=mean,
            cov_diag_factor=tf.linalg.diag_part(diag_noise),
            cov_perturb_factor=tf.transpose(features, (1, 0)),
        )

        log_prob = predictive.log_prob(self.y_train)

        return seed, -log_prob / self.x_train.shape[0]
