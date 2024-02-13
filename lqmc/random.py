from typing import Tuple, Union
import tensorflow as tf
import tensorflow_probability as tfp

from lqmc.utils import to_tensor

Seed = Union[tf.Tensor, Tuple[tf.Tensor], Tuple[int]]


def randint(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    minval: tf.Tensor,
    maxval: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random integers in the range `[minval, maxval]`,
    uniformly distributed, and propagate a new random seed.

    Arguments:
        shape: Shape of the output tensor.
        seed: Random seed for random number generator.
        minval: Lower bound of the range of random integers to generate.
        maxval: Upper bound of the range of random integers to generate.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random integers in the range `[minval, maxval]`.
    """

    minval = tf.cast(minval, tf.int32) if type(minval) == int else minval
    maxval = tf.cast(maxval, tf.int32) if type(maxval) == int else maxval

    assert (
        minval.dtype == maxval.dtype
    ), "minval and maxval must have the same dtype"
    assert minval.dtype in [
        tf.int32,
        tf.int64,
    ], f"Invalid dtype: {minval.dtype=}"

    seed, next_seed = tfp.random.split_seed(seed, num=2)
    return next_seed, tf.random.stateless_uniform(
        shape=shape,
        seed=seed,
        minval=minval,
        maxval=maxval + 1,
        dtype=minval.dtype,
    )


def randu(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    minval: tf.Tensor,
    maxval: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random uniforms in the range `[minval, maxval]`,
    uniformly distributed, and propagate a new random seed.

    Arguments:
        shape: Shape of the output tensor.
        seed: Random seed for random number generator.
        minval: Lower bound of the range of random uniforms to generate.
        maxval: Upper bound of the range of random uniforms to generate.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random uniforms in the range `[minval, maxval]`.
    """

    assert (
        minval.dtype == maxval.dtype
    ), "minval and maxval must have the same dtype"
    assert minval.dtype in [
        tf.float32,
        tf.float64,
    ], f"Invalid dtype: {minval.dtype=}"

    seed, next_seed = tfp.random.split_seed(seed, n=2)

    return next_seed, tf.random.stateless_uniform(
        shape=shape,
        seed=seed,
        minval=minval,
        maxval=maxval,
        dtype=minval.dtype,
    )

def randperm(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    maxval: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random permutations of integers in the range `[0, maxval]`,
    uniformly distributed, and propagate a new random seed.
    
    Arguments:
        shape: Shape of the output tensor.
        seed: Random seed for random number generator.
        maxval: Upper bound of the range of random permutations to generate.
        dtype: Data type of the output tensor.
    
    Returns:
        seed: New random seed produced by splitting.
        rand: Random permutations of integers in the range `[0, maxval]`.
    """
    
    assert maxval.dtype in [
        tf.int32,
        tf.int64,
    ], f"Invalid dtype: {maxval.dtype=}"

    seed, next_seed = tfp.random.split_seed(seed, n=2)
    uniform = tf.random.stateless_uniform(
        shape=shape + (maxval + 1,),
        seed=seed,
        dtype=tf.float32,
    ) 
    return next_seed, tf.argsort(uniform, axis=-1)


def randn(
    shape: tf.TensorShape,
    seed: tf.Tensor,
    mean: tf.Tensor,
    stddev: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random normals with mean `mean` and standard deviation `stddev`,
    and propagate a new random seed.

    Arguments:
        shape: Shape of the output tensor.
        seed: Random seed for random number generator.
        mean: Mean of the normal distribution.
        stddev: Standard deviation of the normal distribution.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Random normals with mean `loc` and standard deviation `scale`.
    """

    assert (
        mean.dtype == stddev.dtype
    ), "mean and stddev must have the same dtype"

    assert mean.dtype in [
        tf.float32,
        tf.float64,
    ], f"Invalid dtype: {mean.dtype=}"

    split = tfp.random.split_seed(seed, n=2)
    seed = split[0]
    next_seed = split[1]

    return next_seed, tf.random.stateless_normal(
        shape=shape,
        seed=seed,
        mean=mean,
        stddev=stddev,
        dtype=mean.dtype,
    )


def mvn(
    seed: tf.Tensor,
    mean: tf.Tensor,
    cov: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate multivariate normals with mean `mean` and covariance `cov`,
    and propagate a new random seed.

    Arguments:
        seed: Random seed for random number generator.
        mean: Mean of the multivariate normal.
        cov: Covariance of the multivariate normal.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Multivariate normals with mean `loc` and covariance `scale`.
    """

    return mvn_chol(
        seed=seed,
        mean=mean,
        cov_chol=tf.linalg.cholesky(cov),
    )


def mvn_chol(
    seed: tf.Tensor,
    mean: tf.Tensor,
    cov_chol: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random multivariate normals with mean `mean` and cholesky
    factor `cov_chol`, and propagate a new random seed.

    Arguments:
        seed: Random seed for random number generator.
        mean: Mean of the multivariate normal.
        cov_chol: Cholesky of the covariance of the multivariate.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Multivariate normals with mean `loc` and covariance `scale`.
    """

    assert (
        mean.dtype == cov_chol.dtype
    ), "mean and chol must have the same dtype"
    assert mean.dtype in [
        tf.float32,
        tf.float64,
    ], f"Invalid dtype: {mean.dtype=}"

    # Split seed
    split = tfp.random.split_seed(seed, n=2)
    seed = split[0]
    next_seed = split[1]

    # Generate random normals
    seed, rand = randn(
        shape=tf.shape(mean),
        seed=seed,
        mean=to_tensor(0.0, mean.dtype),
        stddev=to_tensor(1.0, mean.dtype),
    )

    # Compute multiply noise by Cholesky factor and add mean
    samples = mean + tf.einsum(
        "...ij, ...j -> ...i",
        cov_chol,
        rand,
    )

    return next_seed, samples


def zero_mean_mvn(
    seed: tf.Tensor,
    cov: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """Generate random multivariate normals with mean zero and covariance `cov`,
    and propagate a new random seed.

    Arguments:
        seed: Random seed for random number generator.
        cov: Covariance of the multivariate normal.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Multivariate normals with mean `loc` and covariance `scale`.
    """

    # Create mean of zeroes
    mean = tf.zeros(shape=tf.shape(cov)[:-1], dtype=cov.dtype)

    return zero_mean_mvn_chol(seed=seed, cov_chol=tf.linalg.cholesky(cov))


def zero_mean_mvn_chol(
    seed: tf.Tensor,
    cov_chol: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:
    """
    Arguments:
        seed: Random seed for random number generator.
        cov_chol: Choleksy of the covariance of the multivariate normal.
        dtype: Data type of the output tensor.

    Returns:
        seed: New random seed produced by splitting.
        rand: Multivariate normals with mean `loc` and cov cholesky `scale`.
    """

    # Create mean of zeroes
    mean = tf.zeros(shape=tf.shape(cov_chol)[:-1], dtype=cov_chol.dtype)

    return mvn_chol(seed=seed, mean=mean, cov_chol=cov_chol)


def zero_mean_mvn_on_grid_from_chol(
    seed: tf.Tensor,
    cov_chols: tf.Tensor,
) -> Tuple[Seed, tf.Tensor]:

    # Get data type
    dtype = cov_chols[0].dtype

    # Get grid shape
    batch_shape = tf.shape(cov_chols[0])[0]
    grid_shape = [tf.shape(cov_chol)[-1] for cov_chol in cov_chols]
    shape = [batch_shape, *grid_shape]

    # Draw random noise for each entry of the grid
    seed, noise = randn(
        shape=shape,
        seed=seed,
        mean=tf.zeros(shape=(), dtype=dtype),
        stddev=tf.ones(shape=(), dtype=dtype),
    )

    for d, chol in enumerate(cov_chols):
        noise = tf.experimental.numpy.swapaxes(noise, d+1, -1)
        noise = tf.einsum("bij, b...j -> b...i", chol, noise)
        noise = tf.experimental.numpy.swapaxes(noise, d+1, -1)

    return seed, noise