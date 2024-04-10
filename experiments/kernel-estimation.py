import argparse
import os

from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from lqmc.joint import (
    IndependentUniform,
    HaltonSequence,
    GaussianCopulaAntiparallelCorrelated,
    GaussianCopulaAntiparallelUncorrelated,
    GaussianCopulaAntiparallelAnticorrelated,
    GaussianCopulaParametrised,
)

from lqmc.gp import GaussianProcess, RandomFeatureGaussianProcess
from lqmc.kernels import ExponentiatedQuadraticKernel
from data.datasets import make_dataset


DTYPE = tf.float64


def parse_args():

    # Make argument parser with just the config argument
    parser = argparse.ArgumentParser()

    # Results arguments
    parser.add_argument("--results-path", type=str, default="_results")

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

    # Training arguments
    parser.add_argument("--seed-training", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--sampler-num-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--sampler-learning-rate", type=float)

    return parser.parse_args()


@tf.function
def gradient_step(optimizer: tf.keras.optimizers.Optimizer, model: tf.keras.Model):

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
    optimizer.apply_gradients(zip(gradients, joint_sampler.trainable_variables))

    return seed, loss


def main():

    # Parser arguments
    args = parse_args()

    # Create directories if they don't exist
    results_path = os.path.join(
        args.results_path,
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
        lengthscales=dataset.dim * [args.lengthscale],
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
    joint_samplers = {
        "iid": IndependentUniform(dim=dataset.dim, dtype=DTYPE),
        "halton": HaltonSequence(dim=dataset.dim, dtype=DTYPE),
        "ortho": GaussianCopulaAntiparallelUncorrelated(
            dim=dataset.dim,
            inverse_cdf=kernel.rff_inverse_cdf,
            dtype=DTYPE,
        ),
        "anti": GaussianCopulaAntiparallelAnticorrelated(
            dim=dataset.dim,
            inverse_cdf=kernel.rff_inverse_cdf,
            dtype=DTYPE,
        ),
        "learnt": GaussianCopulaParametrised(
            seed=copula_seed,
            dim=dataset.dim,
            inverse_cdf=kernel.rff_inverse_cdf,
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
    mean, _ = gp(dataset.x_test, noiseless=False)
    rmse = tf.reduce_mean((mean - dataset.y_test) ** 2.0) ** 0.5
    nll = gp.pred_nll(dataset.x_test, dataset.y_test)

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.sampler_learning_rate)

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
        mse_losses = []
        for _ in trange(args.num_trials):
            seed, omega = joint(seed, batch_size=args.num_ensembles)
            apply_rotation = name != "halton"
            seed, rmse_loss = kernel.rmse_loss(
                seed=seed,
                omega=omega,
                x1=dataset.x_train,
                apply_rotation=apply_rotation,
            )
            mse_losses.append(rmse_loss**2.)

        # Print mean and stderr of MSE losses
        mse_losses = tf.stack(mse_losses)
        mean_mse_loss = tf.reduce_mean(mse_losses)
        stderr_mse_loss = tf.math.reduce_std(mse_losses) / args.num_trials**0.5

        print(
            f"\n{name} MSE loss: {mean_mse_loss.numpy()} +/- {stderr_mse_loss.numpy()}\n"
        )

        # Save mean and stderr of MSE losses into a text file
        with open(os.path.join(results_path, f"joint-metrics"), "a") as file:
            file.write(
                f"{name} MSE: {mean_mse_loss.numpy()} +/- {stderr_mse_loss.numpy()}\n"
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
