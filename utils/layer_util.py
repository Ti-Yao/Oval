# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Layer utilities."""

import tensorflow as tf
import array_ops

# from tensorflow_mri.python.layers import convolutional
# from tensorflow_mri.python.layers import signal_layers


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
    resized = [array_ops.resize_with_crop_or_pad(tensor, shape)
               for tensor in inputs[1:]]

    # Set the static shape for each resized tensor.
    for i, tensor in enumerate(resized):
      static_shape = inputs[0].shape.as_list()
      static_shape[axis] = inputs[i + 1].shape.as_list()[axis]
      static_shape = tf.TensorShape(static_shape)
      resized[i] = tf.ensure_shape(tensor, static_shape)
    return tf.concat(inputs[:1] + resized, axis=self.axis)  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
