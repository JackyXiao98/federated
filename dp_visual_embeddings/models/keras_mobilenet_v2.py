# Copyright 2022, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MobileNet v2 model in tf.keras using Group Normalization.

For architectural details, see the original paper,
https://arxiv.org/abs/1801.04381.
"""

from typing import Optional

import tensorflow as tf
from tensorflow_addons import layers as tfa_layers

from dp_visual_embeddings.models import keras_utils


# This function is based on the original TF implementation of MobileNet.
# It ensures that all layers have a channel number that is divisible by 8.
# For more details (and the original implementation in TF) see
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
def _make_divisible(v, divisor, min_value=None):
  if min_value is None:
    min_value = divisor
  new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
  # Make sure that round down does not go down by more than 10%.
  if new_v < 0.9 * v:
    new_v += divisor
  return new_v


def _compute_pad(image_shape, kernel_size, enforce_odd=True):
  """Computes a padding length for a given image shape and kernel.

  Args:
    image_shape: A tuple or list of 2 integers.
    kernel_size: A positive integer.
    enforce_odd: A boolean indicating whether the padding should result in an
      image with odd width/height. If True, we remove one from the left/top of
      the padding (as necessary) to enforce this.

  Returns:
    A tuple of 2 tuples, each with 2 integers, indicating the left/right
      padding, and the top/down padding.
  """
  padding = (kernel_size//2, kernel_size//2)
  if enforce_odd:
    adjust = (1 - image_shape[0] % 2, 1 - image_shape[1] % 2)
  else:
    adjust = (0, 0)
  return ((padding[0] - adjust[0], padding[0]), (padding[1] - adjust[1],
                                                 padding[1]))


def _inverted_res_block(input_tensor,
                        expansion_factor,
                        stride,
                        filters,
                        alpha,
                        block_number,
                        num_groups=2,
                        dropout_prob=None,
                        expansion_layer=True,
                        trainable_conv=True):
  """Creates an inverted residual block.

  Args:
    input_tensor: A 4D input tensor, with shape (samples, channels, rows, cols)
      or (samples, rows, cols, channels).
    expansion_factor: A positive integer that governs (multiplicatively) how
      many channels are added in the initial expansion layer.
    stride: A positive integer giving the stride of the depthwise convolutional
      layer.
    filters: The base number of filters in the projection layer.
    alpha: A float multiplier for the number of filters in the projection layer.
      If set to 1.0, we use the number of filters is given by the`filters` arg.
    block_number: An integer specifying which inverted residual layer this is.
      Used only for naming purposes.
    num_groups: The number of groups to use in the GroupNorm layers.
    dropout_prob: The probability of setting a weight to zero in the dropout
      layer. If None, no dropout is used.
    expansion_layer: Whether to use an initial expansion layer.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.

  Returns:
    A 4D tensor with the same shape as the input tensor.
  """

  if tf.keras.backend.image_data_format() == 'channels_last':
    row_axis = 1
    col_axis = 2
    channel_axis = 3
  else:
    channel_axis = 1
    row_axis = 2
    col_axis = 3

  image_shape = (input_tensor.shape[row_axis], input_tensor.shape[col_axis])
  num_input_channels = input_tensor.shape[channel_axis]
  x = input_tensor
  prefix = 'block_{}_'.format(block_number)

  if expansion_layer:
    # We perform an initial pointwise convolution layer.
    x = tf.keras.layers.Conv2D(
        expansion_factor * num_input_channels,
        kernel_size=1,
        padding='same',
        use_bias=False,
        activation=None,
        name=prefix + 'expand_conv',
        trainable=trainable_conv)(
            x)
    x = tfa_layers.GroupNormalization(
        groups=num_groups, axis=channel_axis, name=prefix + 'expand_gn')(
            x)
    if dropout_prob:
      x = tf.keras.layers.Dropout(
          dropout_prob, name=prefix + 'expand_dropout')(
              x)
    x = tf.keras.layers.ReLU(6.0, name=prefix + 'expand_relu')(x)

  # We now use depthwise convolutions
  if stride%2 == 0:
    padding = _compute_pad(image_shape, 3, enforce_odd=True)
    x = tf.keras.layers.ZeroPadding2D(padding=padding, name=prefix + 'pad')(x)

  padding_type = 'same' if stride == 1 else 'valid'
  x = tf.keras.layers.DepthwiseConv2D(
      kernel_size=3,
      strides=stride,
      activation=None,
      use_bias=False,
      padding=padding_type,
      name=prefix + 'depthwise_conv',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=num_groups, axis=channel_axis, name=prefix + 'depthwise_gn')(
          x)
  if dropout_prob:
    x = tf.keras.layers.Dropout(
        dropout_prob, name=prefix + 'depthwise_dropout')(
            x)
  x = tf.keras.layers.ReLU(6.0, name=prefix + 'depthwise_relu')(x)

  # Projection phase, using pointwise convolutions
  num_projection_filters = _make_divisible(int(filters * alpha), 8)
  x = tf.keras.layers.Conv2D(
      num_projection_filters,
      kernel_size=1,
      padding='same',
      use_bias=False,
      activation=None,
      name=prefix + 'project_conv',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=num_groups, axis=channel_axis, name=prefix + 'project_gn')(
          x)
  if dropout_prob:
    x = tf.keras.layers.Dropout(
        dropout_prob, name=prefix + 'project_dropout')(
            x)
  if num_input_channels == num_projection_filters and stride == 1:
    x = tf.keras.layers.add([input_tensor, x])
  return x


def create_mobilenet_v2_for_embedding_prediction(
    input_shape: tuple[int, int, int],
    embedding_dim_size: int,
    alpha: float = 1.0,
    pooling: str = 'avg',
    num_groups: int = 2,
    dropout_prob: Optional[float] = None,
    always_normalize: bool = False,
    trainable_conv: bool = True) -> tf.keras.Model:
  """Instantiates MobileNetV2 model w/ GroupNorm, to convert image to embedding.

  The output embeddings will be normalized at inference time when
  `training=False` in the model `call`.

  Args:
    input_shape: A tuple of length 3 describing the number of rows, columns, and
      channels of an input. Can be in channel-first or channel-last format.
    embedding_dim_size: The size of the embedding vector to output.
    alpha: A float multiplier for the number of filters in the projection
      pointwise convolutional layers. If set to 1.0, we use the default number
      of filters from the original paper.
    pooling: String indicating the pooling mode for the final fully-connected
      layer. Can be one of 'avg' or 'max'.
    num_groups: The number of groups to use in the GroupNorm layers.
    dropout_prob: The probability of setting a weight to zero in the dropout
      layer. If None, no dropout is used.
    always_normalize: If True, the output embedding will be normalized for
      both training and inference; if False, only normalized for inference.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.

  Returns:
    A `tf.keras.Model`.
  """

  if len(input_shape) != 3:
    raise ValueError('Input shape should be a tuple of length 3.')

  if tf.keras.backend.image_data_format() == 'channels_last':
    row_axis, col_axis = (0, 1)
    channel_axis = 3
  else:
    row_axis, col_axis = (1, 2)
    channel_axis = 1

  image_shape = (input_shape[row_axis], input_shape[col_axis])

  img_input = tf.keras.layers.Input(shape=input_shape, name='images')

  initial_padding = _compute_pad(image_shape, 3, enforce_odd=True)
  x = tf.keras.layers.ZeroPadding2D(
      initial_padding, name='initial_pad')(img_input)

  num_filters_first_block = _make_divisible(32 * alpha, 8)
  x = tf.keras.layers.Conv2D(
      num_filters_first_block,
      kernel_size=3,
      strides=(2, 2),
      padding='valid',
      use_bias=False,
      name='initial_conv',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=num_groups, axis=channel_axis, name='initial_gn')(
          x)
  if dropout_prob:
    x = tf.keras.layers.Dropout(dropout_prob, name='initial_dropout')(x)
  x = tf.keras.layers.ReLU(6.0, name='initial_relu')(x)

  x = _inverted_res_block(
      x,
      expansion_factor=1,
      stride=1,
      filters=16,
      alpha=alpha,
      block_number=0,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      expansion_layer=False,
      trainable_conv=trainable_conv)

  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=2,
      filters=24,
      alpha=alpha,
      block_number=1,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=24,
      alpha=alpha,
      block_number=2,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)

  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=2,
      filters=32,
      alpha=alpha,
      block_number=3,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=32,
      alpha=alpha,
      block_number=4,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=32,
      alpha=alpha,
      block_number=5,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)

  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=2,
      filters=64,
      alpha=alpha,
      block_number=6,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=64,
      alpha=alpha,
      block_number=7,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=64,
      alpha=alpha,
      block_number=8,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=64,
      alpha=alpha,
      block_number=9,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)

  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=96,
      alpha=alpha,
      block_number=10,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=96,
      alpha=alpha,
      block_number=11,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=96,
      alpha=alpha,
      block_number=12,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)

  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=2,
      filters=160,
      alpha=alpha,
      block_number=13,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=160,
      alpha=alpha,
      block_number=14,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=160,
      alpha=alpha,
      block_number=15,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)

  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=320,
      alpha=alpha,
      block_number=16,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)

  # For the last layer, we do not use alpha < 1. This is to recreate the
  # non-usage of alpha in the last layer, as stated in the paper.
  if alpha > 1.0:
    last_block_filters = _make_divisible(1280 * alpha, 8)
  else:
    last_block_filters = 1280

  x = tf.keras.layers.Conv2D(
      last_block_filters,
      kernel_size=1,
      use_bias=False,
      name='last_conv',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=num_groups, axis=channel_axis, name='last_gn')(
          x)
  if dropout_prob:
    x = tf.keras.layers.Dropout(dropout_prob, name='last_dropout')(x)
  x = tf.keras.layers.ReLU(6.0, name='last_relu')(x)

  if pooling == 'avg':
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
  elif pooling == 'max':
    x = tf.keras.layers.GlobalMaxPooling2D()(x)

  predicted_embeddings = tf.keras.layers.Dense(
      embedding_dim_size, activation=None, use_bias=True,
      name='predicted_embeddings')(x)

  predicted_embeddings = keras_utils.EmbedNormLayer(
      always_normalize=always_normalize)(
          predicted_embeddings)
  return tf.keras.models.Model(inputs=img_input, outputs=predicted_embeddings)


def create_small_mobilenet_v2_for_embedding_prediction(
    input_shape: tuple[int, int, int],
    embedding_dim_size: int,
    alpha: float = 1.0,
    pooling: str = 'avg',
    num_groups: int = 2,
    dropout_prob: Optional[float] = None,
    always_normalize: bool = False,
    trainable_conv: bool = True) -> tf.keras.Model:
  """Instantiates small MobileNetV2 model w/ GroupNorm, to convert image->embedding.

  The output embeddings will be normalized at inference time when
  `training=False` in the model `call`.

  Args:
    input_shape: A tuple of length 3 describing the number of rows, columns, and
      channels of an input. Can be in channel-first or channel-last format.
    embedding_dim_size: The size of the embedding vector to output.
    alpha: A float multiplier for the number of filters in the projection
      pointwise convolutional layers. If set to 1.0, we use the default number
      of filters from the original paper.
    pooling: String indicating the pooling mode for the final
      fully-connected layer. Can be one of 'avg' or 'max'.
    num_groups: The number of groups to use in the GroupNorm layers.
    dropout_prob: The probability of setting a weight to zero in the dropout
      layer. If None, no dropout is used.
    always_normalize: If True, the output embedding will be normalized for
      both training and inference; if False, only normalized for inference.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.

  Returns:
    A `tf.keras.Model`.
  """

  if len(input_shape) != 3:
    raise ValueError('Input shape should be a tuple of length 3.')

  if tf.keras.backend.image_data_format() == 'channels_last':
    row_axis, col_axis = (0, 1)
    channel_axis = 3
  else:
    row_axis, col_axis = (1, 2)
    channel_axis = 1

  image_shape = (input_shape[row_axis], input_shape[col_axis])

  img_input = tf.keras.layers.Input(shape=input_shape, name='images')

  initial_padding = _compute_pad(image_shape, 3, enforce_odd=True)
  x = tf.keras.layers.ZeroPadding2D(
      initial_padding, name='initial_pad')(img_input)

  num_filters_first_block = _make_divisible(32 * alpha, 8)
  x = tf.keras.layers.Conv2D(
      num_filters_first_block,
      kernel_size=3,
      strides=(2, 2),
      padding='valid',
      use_bias=False,
      name='initial_conv',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=num_groups, axis=channel_axis, name='initial_gn')(
          x)
  if dropout_prob:
    x = tf.keras.layers.Dropout(dropout_prob, name='initial_dropout')(x)
  x = tf.keras.layers.ReLU(6.0, name='initial_relu')(x)

  x = _inverted_res_block(
      x,
      expansion_factor=1,
      stride=1,
      filters=16,
      alpha=alpha,
      block_number=0,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      expansion_layer=False,
      trainable_conv=trainable_conv)

  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=2,
      filters=24,
      alpha=alpha,
      block_number=1,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=24,
      alpha=alpha,
      block_number=2,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)

  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=2,
      filters=32,
      alpha=alpha,
      block_number=3,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=32,
      alpha=alpha,
      block_number=4,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)
  x = _inverted_res_block(
      x,
      expansion_factor=6,
      stride=1,
      filters=32,
      alpha=alpha,
      block_number=5,
      num_groups=num_groups,
      dropout_prob=dropout_prob,
      trainable_conv=trainable_conv)

  # For the last layer, we do not use alpha < 1. This is to recreate the
  # non-usage of alpha in the last layer, as stated in the paper.
  if alpha > 1.0:
    last_block_filters = _make_divisible(640 * alpha, 8)
  else:
    last_block_filters = 640

  x = tf.keras.layers.Conv2D(
      last_block_filters,
      kernel_size=1,
      use_bias=False,
      name='last_conv',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=num_groups, axis=channel_axis, name='last_gn')(
          x)
  if dropout_prob:
    x = tf.keras.layers.Dropout(dropout_prob, name='last_dropout')(x)
  x = tf.keras.layers.ReLU(6.0, name='last_relu')(x)

  if pooling == 'avg':
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
  elif pooling == 'max':
    x = tf.keras.layers.GlobalMaxPooling2D()(x)

  predicted_embeddings = tf.keras.layers.Dense(
      embedding_dim_size, activation=None, use_bias=True,
      name='predicted_embeddings')(x)

  predicted_embeddings = keras_utils.EmbedNormLayer(
      always_normalize=always_normalize)(
          predicted_embeddings)
  return tf.keras.models.Model(inputs=img_input, outputs=predicted_embeddings)


def create_mobilenet_v2_for_backbone_training(
    input_shape: tuple[int, int, int],
    num_identities: int,
    alpha: float = 1.0,
    pooling: str = 'avg',
    num_groups: int = 2,
    dropout_prob: Optional[float] = None,
    embedding_dim_size: int = 128,
    use_normalize: bool = False,
    trainable_conv: bool = True) -> keras_utils.EmbeddingModel:
  """Instantiates MobileNetV2 w/ GroupNorm with dense layer of embeddings at top.

  This model affords jointly training the MobileNet network with the reference
  embeddings.

  Args:
    input_shape: A tuple of length 3 describing the number of rows, columns, and
      channels of an input. Can be in channel-first or channel-last format.
    num_identities: The number of reference identities to learn embeddings.
    alpha: A float multiplier for the number of filters in the projection
      pointwise convolutional layers. If set to 1.0, we use the default number
      of filters from the original paper.
    pooling: String indicating the pooling mode for the final
      fully-connected layer. Can be one of 'avg' or 'max'.
    num_groups: The number of groups to use in the GroupNorm layers.
    dropout_prob: The probability of setting a weight to zero in the dropout
      layer. If None, no dropout is used.
    embedding_dim_size: The size of the embedding vector.
    use_normalize: If True, normalizes embeddings and head for training.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.

  Returns:
     A tuple of (`tf.keras.Model`, global_variables, client_variables);
     `global_variables` are feature extractors; `client_variables` are head
     (embedding -> identity) for training.
  """

  mobilenet_v2_base = create_mobilenet_v2_for_embedding_prediction(
      input_shape,
      embedding_dim_size,
      alpha,
      pooling,
      num_groups,
      dropout_prob,
      always_normalize=use_normalize,
      trainable_conv=trainable_conv)
  return keras_utils.add_embedding_head(
      mobilenet_v2_base, num_identities, use_normalize=use_normalize)


def create_small_mobilenet_v2_for_backbone_training(
    input_shape: tuple[int, int, int],
    num_identities: int,
    alpha: float = 1.0,
    pooling: str = 'avg',
    num_groups: int = 2,
    dropout_prob: Optional[float] = None,
    embedding_dim_size: int = 128,
    use_normalize: bool = False,
    trainable_conv: bool = True) -> keras_utils.EmbeddingModel:
  """Instantiates small MobileNetV2 w/ GroupNorm w/ dense layer of embeddings at top.

  This model affords jointly training the MobileNet network with the reference
  embeddings.

  Args:
    input_shape: A tuple of length 3 describing the number of rows, columns, and
      channels of an input. Can be in channel-first or channel-last format.
    num_identities: The number of reference identities to learn embeddings.
    alpha: A float multiplier for the number of filters in the projection
      pointwise convolutional layers. If set to 1.0, we use the default number
      of filters from the original paper.
    pooling: String indicating the pooling mode for the final
      fully-connected layer. Can be one of 'avg' or 'max'.
    num_groups: The number of groups to use in the GroupNorm layers.
    dropout_prob: The probability of setting a weight to zero in the dropout
      layer. If None, no dropout is used.
    embedding_dim_size: The size of the embedding vector.
    use_normalize: If True, normalizes embeddings and head for training.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.

  Returns:
     A tuple of (`tf.keras.Model`, global_variables, client_variables);
     `global_variables` are feature extractors; `client_variables` are head
     (embedding -> identity) for training.
  """
  small_mobilenet_v2_base = create_small_mobilenet_v2_for_embedding_prediction(
      input_shape,
      embedding_dim_size,
      alpha,
      pooling,
      num_groups,
      dropout_prob,
      always_normalize=use_normalize,
      trainable_conv=trainable_conv)
  return keras_utils.add_embedding_head(
      small_mobilenet_v2_base, num_identities, use_normalize=use_normalize)
