import argparse
import os

from tqdm import trange
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

from lqmc.joint import (
    IndependentUniform,
    HaltonSequence,
    GaussianCopulaAntiparallelUncorrelated,
    GaussianCopulaAntiparallelAnticorrelated,
    GaussianCopulaParametrised,
)

from lqmc.gp import GaussianProcess, RandomFeatureGaussianProcess
from lqmc.kernels import ExponentiatedQuadraticKernel
from data.datasets import make_dataset


DTYPE = tf.float64
tfd = tfp.distributions


def parse_args():

    # Make argument parser with just the config argument
    parser = argparse.ArgumentParser()

    # Results arguments
    parser.add_argument("--experiment-name", type=str)

    # Dataset arguments
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed-dataset", type=int)
    parser.add_argument("--num-splits", type=int)
    parser.add_argument("--split-id", type=int)
    parser.add_argument("--max-datapoints", type=int)

    # Kernel arguments
    parser.add_argument("--lengthscale", type=float)
    parser.add_argument("--output-scale", type=float)

    # Exact GP arguments
    parser.add_argument("--noise-std", type=float)

    # Joint sampler arguments
    parser.add_argument("--num-ensembles", type=int)
    parser.add_argument("--num-trials", type=int)
    parser.add_argument(
        "--frame",
        type=str,
        choices=["ortho", "ortho_anti"],
    )
    parser.add_argument("--num-features", type=int)

    # Training arguments
    parser.add_argument("--seed-training", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--sampler-num-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--sampler-learning-rate", type=float)

    return parser.parse_args()


def kld(
    exact_mean: tf.Tensor,
    exact_cov: tf.Tensor,
    approx_mean: tf.Tensor,
    approx_cov: tf.Tensor,
):
    exact_dist = tfd.MultivariateNormalTriL(
        loc=exact_mean,
        scale_tril=tf.linalg.cholesky(exact_cov),
    )

    approx_dist = tfd.MultivariateNormalTriL(
        loc=approx_mean,
        scale_tril=tf.linalg.cholesky(approx_cov),
    )

    return tfd.kl_divergence(exact_dist, approx_dist)


@tf.function
def gradient_step(
    optimizer: tf.keras.optimizers.Optimizer, model: tf.keras.Model
):

    with tf.GradientTape() as tape:
        loss = model.loss()

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


@tf.function
def joint_sampler_gradient_step(
    seed: tf.Tensor,
    optimizer: tf.keras.optimizers.Optimizer,
    joint_sampler: tf.keras.Model,
    kernel: tf.keras.Model,
    x: tf.Tensor,
    num_ensembles: int,
):

    with tf.GradientTape() as tape:

        seed, omega = joint_sampler(seed=seed, batch_size=num_ensembles)
        seed, loss = kernel.rmse_loss(seed=seed, omega=omega, x1=x)

    gradients = tape.gradient(loss, joint_sampler.trainable_variables)
    optimizer.apply_gradients(
        zip(gradients, joint_sampler.trainable_variables)
    )

    return seed, loss


def run_single_trial(
    seed: tf.Tensor,
    joint: tf.keras.Model,
    num_ensembles: int,
    name: str,
    kernel: tf.keras.Model,
    dataset,
    rfgp: tf.keras.Model,
    exact_mean: tf.Tensor,
    exact_cov: tf.Tensor,
):
    seed, omega = joint(seed, batch_size=num_ensembles)
    seed, rmse_loss = kernel.rmse_loss(
        seed=seed,
        omega=omega,
        x1=dataset.x_test,
        apply_rotation=True,
    )

    seed, nll_pred_loss = rfgp.pred_loss(
        seed=seed,
        x_pred=dataset.x_test,
        y_pred=dataset.y_test,
        num_ensembles=num_ensembles,
    )
    nll_pred_loss = nll_pred_loss + tf.math.log(dataset.y_std[0])

    seed, pred_mean, pred_cov = rfgp(
        seed,
        dataset.x_test,
        num_ensembles=num_ensembles,
    )
    pred_rmse_loss = tf.reduce_mean((pred_mean - dataset.y_test) ** 2.0) ** 0.5
    pred_rmse_loss = pred_rmse_loss * dataset.y_std[0]

    kl_divergence = (
        kld(
            exact_mean=exact_mean,
            exact_cov=exact_cov,
            approx_mean=pred_mean,
            approx_cov=pred_cov,
        )
        / dataset.y_test.shape[0]
    )

    return seed, rmse_loss, nll_pred_loss, pred_rmse_loss, kl_divergence


def main():

    # Parser arguments
    args = parse_args()

    # Create directories if they don't exist
    results_path = os.path.join(
        "_results",
        args.experiment_name,
        args.dataset,
        f"split_{args.split_id}_{args.num_splits}",
    )
    os.makedirs(results_path, exist_ok=True)

    # Create dataset
    dataset = make_dataset(
        name=args.dataset,
        seed=[args.seed_dataset, args.seed_dataset],
        num_splits=args.num_splits,
        split_id=args.split_id,
        max_datapoints=args.max_datapoints,
        dtype=DTYPE,
    )

    # Create kernel
    kernel = ExponentiatedQuadraticKernel(
        lengthscale=args.lengthscale,
        output_scale=args.output_scale,
        dim=dataset.dim,
        dtype=DTYPE,
    )

    # Create GP
    gp = GaussianProcess(
        kernel=kernel,
        noise_std=args.noise_std,
        x=dataset.x_train,
        y=dataset.y_train,
        dtype=DTYPE,
    )

    copula_seed = [args.seed_training, args.seed_training]
    num_points = (
        args.num_features if args.num_features is not None else dataset.dim
    )
    num_points = num_points if args.frame == "ortho" else 2 * num_points
    joint_samplers = {
        "iid": IndependentUniform(
            dim=dataset.dim,
            num_points=num_points,
            dtype=DTYPE,
        ),
        "halton": HaltonSequence(
            dim=dataset.dim,
            num_points=num_points,
            dtype=DTYPE,
        ),
        "ortho": GaussianCopulaAntiparallelUncorrelated(
            dim=dataset.dim,
            inverse_cdf=kernel.rff_inverse_cdf,
            frame_type=args.frame,
            num_points=num_points,
            dtype=DTYPE,
        ),
        "anti": GaussianCopulaAntiparallelAnticorrelated(
            dim=dataset.dim,
            inverse_cdf=kernel.rff_inverse_cdf,
            frame_type=args.frame,
            num_points=num_points,
            dtype=DTYPE,
        ),
        "learnt": GaussianCopulaParametrised(
            seed=copula_seed,
            dim=dataset.dim,
            inverse_cdf=kernel.rff_inverse_cdf,
            frame_type=args.frame,
            num_points=num_points,
            dtype=DTYPE,
        ),
    }

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Train GP hyperparameters
    pbar = trange(args.num_steps)
    for _ in pbar:
        loss = gradient_step(optimizer, gp)
        pbar.set_description(f"GP loss: {loss.numpy():.4f}")

    # Evaluate RMSE and NLL on test set
    exact_mean, exact_cov = gp(dataset.x_test)
    rmse = tf.reduce_mean((exact_mean - dataset.y_test) ** 2.0) ** 0.5
    rmse = rmse * dataset.y_std[0]
    nll = gp.pred_loss(dataset.x_test, dataset.y_test)
    nll = nll + tf.math.log(dataset.y_std[0])

    # Print prediction metrics and save them into a text file
    print(f"RMSE: {rmse.numpy()}")
    print(f"NLL: {nll.numpy()}")
    with open(os.path.join(results_path, "prediction-metrics"), "w") as file:
        file.write(f"RMSE: {rmse.numpy()}\n")
        file.write(f"NLL: {nll.numpy()}\n")

    # Print trained parameters and save them into a text file
    print(f"GP lengthscales: {gp.kernel.lengthscales.numpy()}")
    print(f"GP output scale: {gp.kernel.output_scale.numpy()}")
    print(f"GP noise std: {gp.noise_std.numpy()}")

    with open(os.path.join(results_path, "trained-hypers"), "w") as file:
        file.write(f"lengthscales: {gp.kernel.lengthscales.numpy()}\n")
        file.write(f"output_scale: {gp.kernel.output_scale.numpy()}\n")
        file.write(f"noise_std: {gp.noise_std.numpy()}\n")

    if os.path.exists(os.path.join(results_path, f"joint-metrics")):
        os.remove(os.path.join(results_path, f"joint-metrics"))

    # Re-create optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.sampler_learning_rate
    )

    # Train learnt joint sampler
    pbar = trange(args.sampler_num_steps)
    seed = [args.seed_training, args.seed_training]
    sampler_losses = []
    for _ in pbar:
        seed, loss = joint_sampler_gradient_step(
            optimizer=optimizer,
            joint_sampler=joint_samplers["learnt"],
            kernel=kernel,
            x=dataset.x_train,
            num_ensembles=args.num_ensembles,
            seed=seed,
        )
        sampler_losses.append(loss)
        pbar.set_description(
            f"Learnt sampler loss: {tf.reduce_mean(sampler_losses).numpy():.4f}"
        )

    # Loop over joints
    for name, joint in joint_samplers.items():

        rfgp = RandomFeatureGaussianProcess(
            kernel=kernel,
            noise_std=float(gp.noise_std),
            x=dataset.x_train,
            y=dataset.y_train,
            joint_sampler=joint,
            dtype=DTYPE,
        )

        rmse_losses = []
        nll_losses = []
        pred_rmse_losses = []
        pred_kld = []
        tf_run_single_trial = tf.function(run_single_trial)
        for _ in trange(args.num_trials):
            seed, rmse_loss, nll_loss, pred_rmse_loss, kld_loss = (
                tf_run_single_trial(
                    seed=seed,
                    joint=joint,
                    num_ensembles=args.num_ensembles,
                    name=name,
                    kernel=kernel,
                    dataset=dataset,
                    rfgp=rfgp,
                    exact_mean=exact_mean,
                    exact_cov=exact_cov,
                )
            )
            rmse_losses.append(rmse_loss)
            nll_losses.append(nll_loss)
            pred_rmse_losses.append(pred_rmse_loss)
            pred_kld.append(kld_loss)

        rmse_losses = tf.stack(rmse_losses)
        mean_rmse_loss = tf.reduce_mean(rmse_losses)
        stderr_rmse_loss = (
            tf.math.reduce_std(rmse_losses) / args.num_trials**0.5
        )

        nll_losses = tf.stack(nll_losses)
        mean_nll_loss = tf.reduce_mean(nll_losses)
        stderr_nll_loss = tf.math.reduce_std(nll_losses) / args.num_trials**0.5

        pred_rmse_losses = tf.stack(pred_rmse_losses)
        mean_pred_rmse_loss = tf.reduce_mean(pred_rmse_losses)
        stderr_pred_rmse_loss = (
            tf.math.reduce_std(pred_rmse_losses) / args.num_trials**0.5
        )

        pred_kld = tf.stack(pred_kld)
        mean_pred_kld = tf.reduce_mean(pred_kld)
        stderr_pred_kld = tf.math.reduce_std(pred_kld) / args.num_trials**0.5

        print(
            f"\n{name} Kernel RMSE loss: {mean_rmse_loss.numpy()} +/- {2. * stderr_rmse_loss.numpy()}"
        )

        print(
            f"{name} Pred NLL loss: {mean_nll_loss.numpy()} +/- {2. * stderr_nll_loss.numpy()}"
        )

        print(
            f"{name} Pred RMSE loss: {mean_pred_rmse_loss.numpy()} +/- {2. * stderr_pred_rmse_loss.numpy()}"
        )

        print(
            f"{name} Pred KLD: {mean_pred_kld.numpy()} +/- {2. * stderr_pred_kld.numpy()}"
        )

        # Save mean and stderr of MSE losses into a text file
        with open(
            os.path.join(results_path, f"kernel-rmse-losses"), "a"
        ) as file:
            file.write(
                f"{name} Kernel MSE: {mean_rmse_loss.numpy()} +/- {2. * stderr_rmse_loss.numpy()}\n"
            )

        # Save mean and stderr of NLL losses into a text file
        with open(os.path.join(results_path, f"pred-nll-losses"), "a") as file:
            file.write(
                f"{name} Pred NLL: {mean_nll_loss.numpy()} +/- {2. * stderr_nll_loss.numpy()}\n"
            )

        with open(
            os.path.join(results_path, f"pred-rmse-losses"), "a"
        ) as file:
            file.write(
                f"{name} Pred RMSE: {mean_pred_rmse_loss.numpy()} +/- {2. * stderr_pred_rmse_loss.numpy()}\n"
            )

        with open(os.path.join(results_path, f"pred-kld"), "a") as file:
            file.write(
                f"{name} Pred KLD: {mean_pred_kld.numpy()} +/- {2. * stderr_pred_kld.numpy()}\n"
            )

        # Save learnt sampler covariance as numpy and imshow
        if name == "learnt":
            cov = joint.covariance.numpy()
            np.save(os.path.join(results_path, "learnt-covariance.npy"), cov)

            plt.imshow(cov)
            plt.axis("off")
            plt.savefig(os.path.join(results_path, "learnt-covariance.png"))


if __name__ == "__main__":
    main()
