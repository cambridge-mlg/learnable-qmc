import argparse

from tqdm import trange
import tensorflow as tf

from lqmc.joint import (
    IndependentUniform,
    HaltonSequence,
    GaussianCopulaAntiparallelCorrelated,
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

    # Training arguments
    parser.add_argument("--seed-training", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--learning-rate", type=float)

    return parser.parse_args()


@tf.function
def gradient_step(optimizer: tf.keras.optimizers.Optimizer, model: tf.keras.Model):
    
    with tf.GradientTape() as tape:
        loss = model.loss()
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss


def main():

    # Parser arguments
    args = parse_args()

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

    # Create random feature GP
    rfgp = RandomFeatureGaussianProcess(
        joint_sampler=GaussianCopulaParametrised(
            seed=args.seed_training,
            dim=dataset.dim,
            trainable=True,
            dtype=DTYPE,
        ),
        kernel=kernel,
        noise_std=args.noise_std,
        x=dataset.x_train,
        y=dataset.y_train,
        dtype=DTYPE,
    )

    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    # Train GP hyperparameters -- add a progress bar with the loss
    pbar = trange(args.num_steps)
    for i in pbar:
        loss = gradient_step(optimizer, gp)
        pbar.set_description(f"GP loss: {loss.numpy():.4f}")



if __name__ == "__main__":
    main()
