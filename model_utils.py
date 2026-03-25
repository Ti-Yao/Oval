"""Array manipulation operations."""
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.python.ops.numpy_ops import np_array_ops


def broadcast_static_shapes(*shapes):
  """Computes the shape of a broadcast given known shapes.

  Like `tf.broadcast_static_shape`, but accepts any number of shapes.

  Args:
    *shapes: Two or more `TensorShapes`.

  Returns:
    A `TensorShape` representing the broadcasted shape.
  """
  bcast_shape = shapes[0]
  for shape in shapes[1:]:
    bcast_shape = tf.broadcast_static_shape(bcast_shape, shape)
  return bcast_shape


def broadcast_dynamic_shapes(*shapes):
  """Computes the shape of a broadcast given symbolic shapes.

  Like `tf.broadcast_dynamic_shape`, but accepts any number of shapes.

  Args:
    shapes: Two or more rank-1 integer `Tensors` representing the input shapes.

  Returns:
    A rank-1 integer `Tensor` representing the broadcasted shape.
  """
  bcast_shape = shapes[0]
  for shape in shapes[1:]:
    bcast_shape = tf.broadcast_dynamic_shape(bcast_shape, shape)
  return bcast_shape


def cartesian_product(*args):
  """Cartesian product of input tensors.

  Args:
    *args: `Tensors` with rank 1.

  Returns:
    A `Tensor` of shape `[M, N]`, where `N` is the number of tensors in `args`
    and `M` is the product of the sizes of all the tensors in `args`.
  """
  return tf.reshape(meshgrid(*args), [-1, len(args)])


def meshgrid(*args):
  """Return coordinate matrices from coordinate vectors.

  Make N-D coordinate arrays for vectorized evaluations of N-D scalar/vector
  fields over N-D grids, given one-dimensional coordinate arrays
  `x1, x2, ..., xn`.

  .. note::
    Similar to `tf.meshgrid`, but uses matrix indexing and returns a stacked
    tensor (along axis -1) instead of a list of tensors.

  Args:
    *args: `Tensors` with rank 1.

  Returns:
    A `Tensor` of shape `[M1, M2, ..., Mn, N]`, where `N` is the number of
    tensors in `args` and `Mi = tf.size(args[i])`.
  """
  return tf.stack(tf.meshgrid(*args, indexing='ij'), axis=-1)


def ravel_multi_index(multi_indices, dims):
  """Converts an array of multi-indices into an array of flat indices.

  Args:
    multi_indices: A `Tensor` of shape `[..., N]` containing multi-indices into
      an `N`-dimensional tensor.
    dims: A `Tensor` of shape `[N]`. The shape of the tensor that
      `multi_indices` indexes into.

  Returns:
    A `Tensor` of shape `[...]` containing flat indices equivalent to
    `multi_indices`.
  """
  strides = tf.math.cumprod(dims, exclusive=True, reverse=True) # pylint:disable=no-value-for-parameter
  return tf.math.reduce_sum(multi_indices * strides, axis=-1)


def unravel_index(indices, dims):
  """Converts an array of flat indices into an array of multi-indices.

  Args:
    indices: A `Tensor` of shape `[...]` containing flat indices into an
      `N`-dimensional tensor.
    dims: A `Tensor` of shape `[N]`. The shape of the tensor that
      `indices` indexes into.

  Returns:
    A `Tensor` of shape `[..., N]` containing multi-indices equivalent to flat
    indices.
  """
  return tf.transpose(tf.unravel_index(indices, dims))


def central_crop(tensor, shape):
  """Crop the central region of a tensor.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the region to crop. The length of `shape`
      must be equal to or less than the rank of `tensor`. If the length of
      `shape` is less than the rank of tensor, the operation is applied along
      the last `len(shape)` dimensions of `tensor`. Any component of `shape` can
      be set to the special value -1 to leave the corresponding dimension
      unchanged.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The centrally cropped tensor.

  Raises:
    ValueError: If `shape` has a rank other than 1.
  """
  tensor = tf.convert_to_tensor(tensor)
  input_shape_tensor = tf.shape(tensor)
  target_shape_tensor = tf.convert_to_tensor(shape)

  # Static checks.
  if target_shape_tensor.shape.rank != 1:
    raise ValueError(f"`shape` must have rank 1. Received: {shape}")

  # Support a target shape with less dimensions than input. In that case, the
  # target shape applies to the last dimensions of input.
  if not isinstance(shape, tf.Tensor):
    shape = [-1] * (tensor.shape.rank - len(shape)) + list(shape)
  target_shape_tensor = tf.concat([
      tf.tile([-1], [tf.rank(tensor) - tf.size(target_shape_tensor)]),
      target_shape_tensor], 0)

  # Dynamic checks.
  checks = [
      tf.debugging.assert_greater_equal(tf.rank(tensor), tf.size(shape)),
      tf.debugging.assert_less_equal(
          target_shape_tensor, tf.shape(tensor), message=(
              "Target shape cannot be greater than input shape."))
  ]
  with tf.control_dependencies(checks):
    tensor = tf.identity(tensor)

  # Crop the tensor.
  slice_begin = tf.where(
      target_shape_tensor >= 0,
      tf.math.maximum(input_shape_tensor - target_shape_tensor, 0) // 2,
      0)
  slice_size = tf.where(
      target_shape_tensor >= 0,
      tf.math.minimum(input_shape_tensor, target_shape_tensor),
      -1)
  tensor = tf.slice(tensor, slice_begin, slice_size)

  # Set static shape, if possible.
  static_shape = _compute_static_output_shape(tensor.shape, shape)
  if static_shape is not None:
    tensor = tf.ensure_shape(tensor, static_shape)

  return tensor


def resize_with_crop_or_pad(tensor, shape, padding_mode='constant'):
  """Crops and/or pads a tensor to a target shape.

  Pads symmetrically or crops centrally the input tensor as necessary to achieve
  the requested shape.

  Args:
    tensor: A `Tensor`.
    shape: A `Tensor`. The shape of the output tensor. The length of `shape`
      must be equal to or less than the rank of `tensor`. If the length of
      `shape` is less than the rank of tensor, the operation is applied along
      the last `len(shape)` dimensions of `tensor`. Any component of `shape` can
      be set to the special value -1 to leave the corresponding dimension
      unchanged.
    padding_mode: A `str`. Must be one of `'constant'`, `'reflect'` or
      `'symmetric'`.

  Returns:
    A `Tensor`. Has the same type as `tensor`. The symmetrically padded/cropped
    tensor.
  """
  tensor = tf.convert_to_tensor(tensor)
  input_shape = tensor.shape
  input_shape_tensor = tf.shape(tensor)
  target_shape = shape
  target_shape_tensor = tf.convert_to_tensor(shape)

  # Support a target shape with less dimensions than input. In that case, the
  # target shape applies to the last dimensions of input.
  if not isinstance(target_shape, tf.Tensor):
    target_shape = [-1] * (input_shape.rank - len(shape)) + list(shape)
  target_shape_tensor = tf.concat([
      tf.tile([-1], [tf.rank(tensor) - tf.size(shape)]),
      target_shape_tensor], 0)

  # Dynamic checks.
  checks = [
      tf.debugging.assert_greater_equal(tf.rank(tensor),
                                        tf.size(target_shape_tensor)),
  ]
  with tf.control_dependencies(checks):
    tensor = tf.identity(tensor)

  # Pad the tensor.
  pad_left = tf.where(
      target_shape_tensor >= 0,
      tf.math.maximum(target_shape_tensor - input_shape_tensor, 0) // 2,
      0)
  pad_right = tf.where(
      target_shape_tensor >= 0,
      (tf.math.maximum(target_shape_tensor - input_shape_tensor, 0) + 1) // 2,
      0)

  tensor = tf.pad(tensor, tf.transpose(tf.stack([pad_left, pad_right])), # pylint: disable=no-value-for-parameter,unexpected-keyword-arg
                  mode=padding_mode)

  # Crop the tensor.
  tensor = central_crop(tensor, target_shape)

  static_shape = _compute_static_output_shape(input_shape, target_shape)
  if static_shape is not None:
    tensor = tf.ensure_shape(tensor, static_shape)

  return tensor


def _compute_static_output_shape(input_shape, target_shape):
  """Compute the static output shape of a resize operation.

  Args:
    input_shape: The static shape of the input tensor.
    target_shape: The target shape.

  Returns:
    The static output shape.
  """
  output_shape = None

  if isinstance(target_shape, tf.Tensor):
    # If target shape is a tensor, we can't infer the output shape.
    return None

  # Get static tensor shape, after replacing -1 values by `None`.
  output_shape = tf.TensorShape(
      [s if s >= 0 else None for s in target_shape])

  # Complete any unspecified target dimensions with those of the
  # input tensor, if known.
  output_shape = tf.TensorShape(
      [s_target or s_input for (s_target, s_input) in zip(
          output_shape.as_list(), input_shape.as_list())])

  return output_shape


def update_tensor(tensor, slices, value):
  """Updates the values of a tensor at the specified slices.

  This operator performs slice assignment.

  .. note::
    Equivalent to `tensor[slices] = value`.

  .. warning::
    TensorFlow does not support slice assignment because tensors are immutable.
    This operator works around this limitation by creating a new tensor, which
    may have performance implications.

  Args:
    tensor: A `tf.Tensor`.
    slices: The indices or slices.
    value: A `tf.Tensor`.

  Returns:
    An updated `tf.Tensor` with the same shape and type as `tensor`.
  """
  # Using a private implementation in the TensorFlow NumPy API.
  # pylint: disable=protected-access
  return _with_index_update_helper(np_array_ops._UpdateMethod.UPDATE,
                                   tensor, slices, value)


def _with_index_update_helper(update_method, a, slice_spec, updates):  # pylint: disable=missing-param-doc
  """Implementation of ndarray._with_index_*."""
  # Adapted from tensorflow/python/ops/numpy_ops/np_array_ops.py.
  # pylint: disable=protected-access
  if (isinstance(slice_spec, bool) or (isinstance(slice_spec, tf.Tensor) and
                                       slice_spec.dtype == tf.dtypes.bool) or
      (isinstance(slice_spec, (np.ndarray, tnp.ndarray)) and
       slice_spec.dtype == np.bool_)):
    slice_spec = tnp.nonzero(slice_spec)

  if not isinstance(slice_spec, tuple):
    slice_spec = np_array_ops._as_spec_tuple(slice_spec)

  return np_array_ops._slice_helper(a, slice_spec, update_method, updates)


def map_fn(fn, elems, batch_dims=1, **kwargs):
  """Transforms `elems` by applying `fn` to each element.

  .. note::
    Similar to `tf.map_fn`, but it supports unstacking along multiple batch
    dimensions.

  For the parameters, see `tf.map_fn`. The only difference is that there is an
  additional `batch_dims` keyword argument which allows specifying the number
  of batch dimensions. The default is 1, in which case this function is equal
  to `tf.map_fn`.
  """
  # This function works by reshaping any number of batch dimensions into a
  # single batch dimension, calling the original `tf.map_fn`, and then
  # restoring the original batch dimensions.
  static_batch_dims = tf.get_static_value(batch_dims)

  # Get batch shapes.
  if static_batch_dims is None:
    # We don't know how many batch dimensions there are statically, so we can't
    # get the batch shape statically.
    static_batch_shapes = tf.nest.map_structure(
        lambda _: tf.TensorShape(None), elems)
  else:
    static_batch_shapes = tf.nest.map_structure(
        lambda x: x.shape[:static_batch_dims], elems)
  dynamic_batch_shapes = tf.nest.map_structure(
      lambda x: tf.shape(x)[:batch_dims], elems)

  # Flatten the batch dimensions.
  elems = tf.nest.map_structure(
      lambda x: tf.reshape(
          x, tf.concat([[-1], tf.shape(x)[batch_dims:]], 0)), elems)

  # Process each batch.
  output = tf.map_fn(fn, elems, **kwargs)

  # Unflatten the batch dimensions.
  output = tf.nest.map_structure(
      lambda x, dynamic_batch_shape: tf.reshape(
          x, tf.concat([dynamic_batch_shape, tf.shape(x)[1:]], 0)),
      output, dynamic_batch_shapes)

  # Set the static batch shapes on the output, if known.
  if static_batch_dims is not None:
    output = tf.nest.map_structure(
        lambda x, static_batch_shape: tf.ensure_shape(
            x, static_batch_shape.concatenate(x.shape[static_batch_dims:])),
        output, static_batch_shapes)

  return output


def slice_along_axis(tensor, axis, start, length):
  """Slices a tensor along the specified axis."""
  begin = tf.scatter_nd([[axis]], [start], [tensor.shape.rank])
  size = tf.tensor_scatter_nd_update(tf.shape(tensor), [[axis]], [length])
  return tf.slice(tensor, begin, size)

def get_nd_layer(name, rank):
  """Get an N-D layer object.

  Args:
    name: A `str`. The name of the requested layer.
    rank: An `int`. The rank of the requested layer.

  Returns:
    A `tf.keras.layers.Layer` object.

  Raises:
    ValueError: If the requested layer is unknown to TFMRI.
  """
  try:
    return _ND_LAYERS[(name, rank)]
  except KeyError as err:
    raise ValueError(
        f"Could not find a layer with name '{name}' and rank {rank}.") from err


_ND_LAYERS = {
    ('AveragePooling', 1): tf.keras.layers.AveragePooling1D,
    ('AveragePooling', 2): tf.keras.layers.AveragePooling2D,
    ('AveragePooling', 3): tf.keras.layers.AveragePooling3D,
    ('Conv', 1): tf.keras.layers.Conv1D,
    ('Conv', 2): tf.keras.layers.Conv2D,
    ('Conv', 3): tf.keras.layers.Conv3D,
    ('ConvLSTM', 1): tf.keras.layers.ConvLSTM1D,
    ('ConvLSTM', 2): tf.keras.layers.ConvLSTM2D,
    ('ConvLSTM', 3): tf.keras.layers.ConvLSTM3D,
    ('ConvTranspose', 1): tf.keras.layers.Conv1DTranspose,
    ('ConvTranspose', 2): tf.keras.layers.Conv2DTranspose,
    ('ConvTranspose', 3): tf.keras.layers.Conv3DTranspose,
    ('Cropping', 1): tf.keras.layers.Cropping1D,
    ('Cropping', 2): tf.keras.layers.Cropping2D,
    ('Cropping', 3): tf.keras.layers.Cropping3D,
    ('DepthwiseConv', 1): tf.keras.layers.DepthwiseConv1D,
    ('DepthwiseConv', 2): tf.keras.layers.DepthwiseConv2D,
#     ('DWT', 1): signal_layers.DWT1D,
#     ('DWT', 2): signal_layers.DWT2D,
#     ('DWT', 3): signal_layers.DWT3D,
    ('GlobalAveragePooling', 1): tf.keras.layers.GlobalAveragePooling1D,
    ('GlobalAveragePooling', 2): tf.keras.layers.GlobalAveragePooling2D,
    ('GlobalAveragePooling', 3): tf.keras.layers.GlobalAveragePooling3D,
    ('GlobalMaxPool', 1): tf.keras.layers.GlobalMaxPool1D,
    ('GlobalMaxPool', 2): tf.keras.layers.GlobalMaxPool2D,
    ('GlobalMaxPool', 3): tf.keras.layers.GlobalMaxPool3D,
#     ('IDWT', 1): signal_layers.IDWT1D,
#     ('IDWT', 2): signal_layers.IDWT2D,
#     ('IDWT', 3): signal_layers.IDWT3D,
    ('LocallyConnected', 1): tf.keras.layers.LocallyConnected1D,
    ('LocallyConnected', 2): tf.keras.layers.LocallyConnected2D,
    ('MaxPool', 1): tf.keras.layers.MaxPool1D,
    ('MaxPool', 2): tf.keras.layers.MaxPool2D,
    ('MaxPool', 3): tf.keras.layers.MaxPool3D,
    ('SeparableConv', 1): tf.keras.layers.SeparableConv1D,
    ('SeparableConv', 2): tf.keras.layers.SeparableConv2D,
    ('SpatialDropout', 1): tf.keras.layers.SpatialDropout1D,
    ('SpatialDropout', 2): tf.keras.layers.SpatialDropout2D,
    ('SpatialDropout', 3): tf.keras.layers.SpatialDropout3D,
    ('UpSampling', 1): tf.keras.layers.UpSampling1D,
    ('UpSampling', 2): tf.keras.layers.UpSampling2D,
    ('UpSampling', 3): tf.keras.layers.UpSampling3D,
    ('ZeroPadding', 1): tf.keras.layers.ZeroPadding1D,
    ('ZeroPadding', 2): tf.keras.layers.ZeroPadding2D,
    ('ZeroPadding', 3): tf.keras.layers.ZeroPadding3D
}


class ResizeAndConcatenate(tf.keras.layers.Layer):
  """Resizes and concatenates a list of inputs.

  Similar to `tf.keras.layers.Concatenate`, but if the inputs have different
  shapes, they are resized to match the shape of the first input.

  Args:
    axis: Axis along which to concatenate.
  """
  def __init__(self, axis=-1, **kwargs):
    super().__init__(**kwargs)
    self.axis = axis

  def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
        })
        return config

  def call(self, inputs):  # pylint: disable=missing-function-docstring,arguments-differ
    if not isinstance(inputs, (list, tuple)):
      raise ValueError(
          f"Layer {self.__class__.__name__} expects a list of inputs. "
          f"Received: {inputs}")

    rank = inputs[0].shape.rank
    if rank is None:
      raise ValueError(
          f"Layer {self.__class__.__name__} expects inputs with known rank. "
          f"Received: {inputs}")
    if self.axis >= rank or self.axis < -rank:
      raise ValueError(
          f"Layer {self.__class__.__name__} expects `axis` to be in the range "
          f"[-{rank}, {rank}) for an input of rank {rank}. "
          f"Received: {self.axis}")
    # Canonical axis (always positive).
    axis = self.axis % rank

    # Resize inputs.
    shape = tf.tensor_scatter_nd_update(tf.shape(inputs[0]), [[axis]], [-1])
    resized = [resize_with_crop_or_pad(tensor, shape)
               for tensor in inputs[1:]]

    # Set the static shape for each resized tensor.
    for i, tensor in enumerate(resized):
      static_shape = inputs[0].shape.as_list()
      static_shape[axis] = inputs[i + 1].shape.as_list()[axis]
      static_shape = tf.TensorShape(static_shape)
      resized[i] = tf.ensure_shape(tensor, static_shape)
    return tf.concat(inputs[:1] + resized, axis=self.axis)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter