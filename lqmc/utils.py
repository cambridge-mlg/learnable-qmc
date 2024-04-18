import tensorflow as tf

i32 = tf.int32
i64 = tf.int64
f32 = tf.float32
f64 = tf.float64

to_tensor = lambda x, dtype: tf.convert_to_tensor(x, dtype=dtype)
cast = lambda x, dtype: tf.cast(x, dtype=dtype)

def ortho_frame(dim: int, num_points: int, dtype: tf.DType) -> tf.Tensor:
    """Given a dimension `dim` and a number of pairs `num_pairs`, returns a
    matrix of shape `(num_pairs, dim)` of the form

        [  1,  0,  0,  0,  0,  0, ... ]
        [  0,  1,  0,  0,  0,  0, ... ]
        [  0,  0,  1,  0,  0,  0, ... ]
        [  0,  0,  0,  1,  0,  0, ... ]
    
    Arguments:
        dim: int, the dimension of the space.
        num_pairs: int, the number of pairs of vectors.
        dtype: tf.DType, the data type of the matrix.

    Returns:
        frame: tensor of shape `(2 * num_pairs, dim)`.
    """
    assert num_points <= dim
    return tf.eye(dim, dtype=dtype)[:num_points]

def ortho_anti_frame(dim: int, num_points: int, dtype: tf.DType) -> tf.Tensor:
    """Given a dimension `dim` and a number of pairs `num_pairs`, returns a
    matrix of shape `(2 * num_pairs, dim)` of the form

        [  1,  0,  0,  0,  0,  0, ... ]
        [  0,  1,  0,  0,  0,  0, ... ]
        [  0,  0,  1,  0,  0,  0, ... ]
        [  0,  0,  0,  1,  0,  0, ... ]
        [ -1,  0,  0,  0,  0,  0, ... ]
        [  0, -1,  0,  0,  0,  0, ... ]
        [  0,  0, -1,  0,  0,  0, ... ]
        [  0,  0,  0, -1,  0,  0, ... ]
    
    Arguments:
        dim: int, the dimension of the space.
        num_pairs: int, the number of pairs of vectors.
        dtype: tf.DType, the data type of the matrix.

    Returns:
        frame: tensor of shape `(2 * num_pairs, dim)`.
    """
    assert num_points % 2 == 0
    assert num_points // 2 <= dim
    frame = tf.eye(dim, dtype=dtype)[:num_points // 2]
    return tf.concat([frame, -frame], axis=0)
