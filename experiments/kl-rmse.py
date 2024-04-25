import argparse
import os

from tqdm import trange
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from lqmc.joint import (
    IndependentUniform,
    HaltonSequence,
    GaussianCopulaAntiparallelUncorrelated,
    GaussianCopulaAntiparallelAnticorrelated,
    GaussianCopulaParametrised,
)

from lqmc.gp import RandomFeatureGaussianProcess
from lqmc.kernels import ExponentiatedQuadraticKernel
from lqmc.random import Seed
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
    parser.add_argument("--max-datapoints", type=int, default=2**16)

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

    # Training arguments
    parser.add_argument("--seed-training", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--learning-rate", type=float)

    return parser.parse_args()


def gradient_step(
    seed: Seed,
    optimizer: tf.keras.optimizers.Optimizer,
    gp: tf.keras.Model,
    num_ensembles: int = 1,
    y_std: tf.Tensor = None,
):

    with tf.GradientTape() as tape:
        seed, loss = gp.loss(seed=seed, num_ensembles=num_ensembles)
        if y_std is not None:
            loss = loss + tf.math.log(y_std)

    gradients = tape.gradient(loss, gp.trainable_variables)
    optimizer.apply_gradients(zip(gradients, gp.trainable_variables))

    return seed, loss


def validation_step(
    seed: Seed,
    gp: tf.keras.Model,
    x_pred: tf.Tensor,
    y_pred: tf.Tensor,
    num_ensembles: int = 1,
    num_trials: int = 100,
    y_std: tf.Tensor = None,
):

    losses = []
    for i in tf.range(num_trials):
        seed, loss = gp.pred_loss(
            seed=seed,
            num_ensembles=num_ensembles,
            x_pred=x_pred,
            y_pred=y_pred,
        )
        if y_std is not None:
            loss = loss + tf.math.log(y_std)
        losses.append(loss)

    mean_loss = tf.reduce_mean(losses)
    stderr_loss = tf.math.reduce_std(losses) / num_trials**0.5

    return seed, mean_loss, stderr_loss


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

    copula_seed = [args.seed_training, args.seed_training]
    num_points = dataset.dim if args.frame == "ortho" else 2 * dataset.dim

    for joint_sampler_name in ["iid", "halton", "ortho", "anti", "learnt"]:

        tf_gradient_step = tf.function(gradient_step)

        # Create kernel
        kernel = ExponentiatedQuadraticKernel(
            lengthscales=dataset.dim * [args.lengthscale],
            output_scale=args.output_scale,
            dim=dataset.dim,
            dtype=DTYPE,
        )

        if joint_sampler_name == "iid":
            joint_sampler = IndependentUniform(
                dim=dataset.dim,
                num_points=num_points,
                dtype=DTYPE,
            )
        elif joint_sampler_name == "halton":
            joint_sampler = HaltonSequence(
                dim=dataset.dim,
                num_points=num_points,
                dtype=DTYPE,
            )

        elif joint_sampler_name == "ortho":
            joint_sampler = GaussianCopulaAntiparallelUncorrelated(
                dim=dataset.dim,
                inverse_cdf=kernel.rff_inverse_cdf,
                frame_type=args.frame,
                num_points=num_points,
                dtype=DTYPE,
            )

        elif joint_sampler_name == "anti":
            joint_sampler = GaussianCopulaAntiparallelAnticorrelated(
                dim=dataset.dim,
                inverse_cdf=kernel.rff_inverse_cdf,
                frame_type=args.frame,
                num_points=num_points,
                dtype=DTYPE,
            )

        elif joint_sampler_name == "learnt":
            joint_sampler = GaussianCopulaParametrised(
                seed=copula_seed,
                dim=dataset.dim,
                inverse_cdf=kernel.rff_inverse_cdf,
                frame_type=args.frame,
                num_points=num_points,
                dtype=DTYPE,
            )

        # Create RFF GP
        gp = RandomFeatureGaussianProcess(
            kernel=kernel,
            noise_std=args.noise_std,
            x=dataset.x_train,
            y=dataset.y_train,
            joint_sampler=joint_sampler,
            dtype=DTYPE,
        )

        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        # Train GP hyperparameters
        pbar = trange(args.num_steps)
        seed = [args.seed_training, args.seed_training]
        train_losses = []
        for i in pbar:
            seed, loss = tf_gradient_step(
                seed=seed,
                optimizer=optimizer,
                gp=gp,
                num_ensembles=args.num_ensembles,
                y_std=tf.reshape(dataset.y_std, (-1,)),
            )
            train_losses.append(loss)
            pbar.set_description(
                f"RFF GP ({joint_sampler_name}) loss: "
                f"{tf.reduce_mean(train_losses).numpy():.4f}"
            )

            if i % 500 == 0:
                # Print lengthscales
                print(f"Lengthscales: {gp.kernel.lengthscales.numpy()}")
                # Print noise std and output scale
                print(f"Noise std: {gp.noise_std.numpy()}")
                print(f"Output scale: {gp.kernel.output_scale.numpy()}")
                seed, mean_val_loss, mean_stderr_loss = validation_step(
                    seed=seed,
                    gp=gp,
                    num_ensembles=args.num_ensembles,
                    x_pred=dataset.x_test,
                    y_pred=dataset.y_test,
                    y_std=tf.reshape(dataset.y_std, (-1,)),
                )
                print(
                    f"Validation loss: "
                    f"{mean_val_loss.numpy():.3f} +/- "
                    f"{2 * mean_stderr_loss.numpy():.3f}"
                )

    quit()
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
        mse_losses = []
        for _ in trange(args.num_trials):
            seed, omega = joint(seed, batch_size=args.num_ensembles)
            apply_rotation = name not in ["iid", "halton"]
            seed, rmse_loss = kernel.rmse_loss(
                seed=seed,
                omega=omega,
                x1=dataset.x_train,
                apply_rotation=apply_rotation,
            )
            mse_losses.append(rmse_loss**2.0)

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
