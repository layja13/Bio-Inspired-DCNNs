#ResNet implementation modified from:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/resnet.py
# v2.5.0
# Original copyright notice included below. 

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=invalid-name

# ALL-CNN implementation adapted from Alex Hernandez-Garcia

# import tensorflow as tf
# from tensorflow import keras
# import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, ConvLSTM2D, Reshape, BatchNormalization,
                                            GlobalAveragePooling2D, Activation, Dropout)
# from tensorflow.keras.layers import AveragePooling2D, Lambda, Flatten, Dense
# from tensorflow.keras.layers import ZeroPadding2D, MaxPooling2D
# from tensorflow.python.keras.layers import Activation, Dropout
# from tensorflow.keras.layers import add, Concatenate
from tensorflow.python.keras.regularizers import l2
# from tensorflow.keras.initializers import Constant, TruncatedNormal

from tensorflow.python.keras import backend
from tensorflow.keras.applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops.gen_nn_ops import conv2d
from tensorflow.python.util.tf_export import keras_export

# from tensorflow.python.keras.applications.resnet import stack1  # block1, block2, stack2, block3, stack3
from bionet.utils import GaborInitializer, DifferenceOfGaussiansInitializer, LowPassInitializer


layers = None


def BioResNet(stack_fn,
              preact,
              use_bias,
              model_name='resnet',
              include_top=True,
              weights='imagenet',
              kernels=None,
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000,
              classifier_activation='softmax',
              **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.
    Args:
        stack_fn: a function that returns output tensor for the
          stacked residual blocks.
        preact: whether to use pre-activation or not
          (True for ResNetV2, False for ResNet and ResNeXt).
        use_bias: whether to use biases for convolutional layers or not
          (True for ResNet and ResNetV2, False for ResNeXt).
        model_name: string, model name.
        include_top: whether to include the fully-connected
          layer at the top of the network.
        weights: one of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
          (i.e. output of `layers.Input()`)
          to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
          if `include_top` is False (otherwise the input shape
          has to be `(224, 224, 3)` (with `channels_last` data format)
          or `(3, 224, 224)` (with `channels_first` data format).
          It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
          when `include_top` is `False`.
          - `None` means that the output of the model will be
              the 4D tensor output of the
              last convolutional layer.
          - `avg` means that global average pooling
              will be applied to the output of the
              last convolutional layer, and thus
              the output of the model will be a 2D tensor.
          - `max` means that global max pooling will
              be applied.
        classes: optional number of classes to classify images
          into, only to be specified if `include_top` is True, and
          if no `weights` argument is specified.
        classifier_activation: A `str` or callable. The activation function to use
          on the "top" layer. Ignored unless `include_top=True`. Set
          `classifier_activation=None` to return the logits of the "top" layer.
          When loading pretrained weights, `classifier_activation` can only
          be `None` or `"softmax"`.
        **kwargs: For backwards compatibility only.
    Returns:
        A `keras.Model` instance.
    """
    global layers
    if 'layers' in kwargs:
        layers = kwargs.pop('layers')
    else:
        layers = VersionAwareLayers()

    if kwargs:
        raise ValueError('Unknown argument(s): %s' % (kwargs,))
    if not (weights in {'imagenet', None} or file_io.file_exists_v2(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
          input_shape,
          default_size=224,
          min_size=32,
          data_format=backend.image_data_format(),
          require_flatten=include_top,
          weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = layers.ZeroPadding2D(
          padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)

    if kernels is None:
        x = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)
    else:
        ####### New code for pre-specified kernels #######
#         if filter_type.capitalize().startswith('Combined'):
#             assert isinstance(params, dict)
#             assert len(params) > 1
#             configuration = params
#             for layer_type in configuration:
#                 assert isinstance(configuration[layer_type], dict)
#         else:
#             # Assume an unnested dictionary has been passed
#             configuration = {filter_type: params}
        for layer_type, params in kernels.items():
            if layer_type.lower() == 'gabor':
                # Parse parameters
                assert 'bs' in params
                if 'sigmas' not in params:
                    assert 'lambdas' in params
                    # params['sigmas'] = [utils.calc_sigma(lambd, b) for lambd in params['lambdas']
                    #                     for b in params['bs']]
                kernel_initializer = GaborInitializer(**params)
            elif layer_type.lower() == 'dog':
                kernel_initializer = DifferenceOfGaussiansInitializer(**params)
            elif layer_type.lower() == 'low-pass':
                kernel_initializer = LowPassInitializer(**params)
            n_kernels = kernel_initializer.n_kernels
            # When using this layer as the first layer in a model, provide the keyword argument 
            # input_shape (tuple of integers, does not include the batch axis), 
            # e.g. input_shape=(128, 128, 3) for 128x128 RGB pictures in data_format="channels_last".

            # Input shape: (batch, rows, cols, channels)
            # Output shape: (batch, new_rows, new_cols, filters)
            x = layers.Conv2D(n_kernels, params['ksize'], #padding='same',
                    activation='relu', use_bias=True,
                    #    activation=None, use_bias=use_bias, strides=2, padding='valid',
                    name=f"{layer_type.lower()}_conv",
                    kernel_initializer=kernel_initializer,
                    trainable=False)(x)
        ########################################################

    if not preact:
        x = layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact:
        x = layers.BatchNormalization(
                axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(classes, activation=classifier_activation,
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = training.Model(inputs, x, name=model_name)

    # Load weights.
    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = data_utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.
    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer.
    kernel_size: default 3, kernel size of the bottleneck layer.
    stride: default 1, stride of the first layer.
    conv_shortcut: default True, use convolution shortcut if True,
        otherwise identity shortcut.
    name: string, block label.
    Returns:
    Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.
    Args:
    x: input tensor.
    filters: integer, filters of the bottleneck layer in a block.
    blocks: integer, blocks in the stacked blocks.
    stride1: default 2, stride of the first layer in the first block.
    name: string, stack label.
    Returns:
    Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def BioResNet50(include_top=True,
                weights='imagenet',
                kernels=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 6, name='conv4')
        return stack1(x, 512, 3, name='conv5')

    return BioResNet(stack_fn, False, True, 'resnet50', include_top, weights,
                     kernels, input_tensor, input_shape, pooling, classes, **kwargs)


# model = BioResNet50(weights=weights, input_shape=image_shape, classes=n_classes)
# model.summary()

def allcnn(image_shape, n_classes, dropout=False, weight_decay=None,
           batch_norm=False, depth='orig', input_dropout=False):
    """
    Defines the All convolutional network (All-CNN), originally described in
    https://arxiv.org/abs/1412.6806. It is implemented in a modular way in
    order to allow for more flexibility, for instance modifying the depth of
    the network.

    Parameters
    ----------
    image_shape : int list
        Shape of the input images

    n_classes : int
        Number of classes in the training data set

    dropout : bool
        If True, Dropout regularization is included

    weight_decay : float
        L2 factor for the weight decay regularization or None

    batch_norm : bool
        If True, batch normalization is added after every convolutional layer

    depth : str
        Allows to specify a shallower or deeper architecture:
            - 'shallower': 1 convolutional block (A) + 1 output block (B)
            - 'deeper': 2 convolutional blocks (A) + 1 output block (B)

    Returns
    -------
    model : Model
        Keras model describing the architecture
    """

    def conv_layer(x, filters, kernel, stride, padding, n_layer):
        """
        Defines a convolutional block, formed by a convolutional layer, an
        optional batch normalization layer and a ReLU activation.

        Parameters
        ----------
        x : Tensor
            The input to the convolutional layer

        filters : int
            Number of filters (width) of the convolutional layer

        kernel : int
            Kernel size

        stride : int
            Stride of the convolution

        n_layer : int
            Layer number, with respect to the whole network
        
        Returns
        -------
        x : Tensor
            Output of the convolutional block
        """
        # Convolution
        x = Conv2D(filters=filters,
                   kernel_size=(kernel, kernel),
                   strides=(stride, stride),
                   padding=padding,
                   activation='linear',   
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=l2reg(weight_decay),   
                   bias_regularizer=l2reg(weight_decay),
                   input_shape=image_shape,
                   name='conv%d' % n_layer)(x)
        
        # Batch Normalization
        if batch_norm:
            x = BatchNormalization(axis=3, 
                                   epsilon=1.001e-5,
                                   gamma_regularizer=l2reg(weight_decay),
                                   beta_regularizer=l2reg(weight_decay),
                                   name='conv%dbn' % n_layer)(x)

        # ReLU Activation
        x = Activation('relu', name='conv%drelu' % n_layer)(x)    

        return x

    def block_a(x, filters, n_block):
        """
        Defines a block of 3 convolutional layers:
            1. 3x3 conv with stride 1
            2. 3x3 conv with stride 1
            3. 3x3 conv with stride 2
        """
        x = conv_layer(x, filters, kernel=3, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 3 + 1)
        x = conv_layer(x, filters, kernel=3, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 3 + 2)
        x = conv_layer(x, filters, kernel=3, stride=2, 
                            padding='same', n_layer=(n_block - 1) * 3 + 3)

        return x

    def block_b(x, filters, n_block):
        """
        Defines a block of 3 convolutional layers:
            1. 3x3 conv with stride 1
            2. 1x1 conv with stride 1 and valid padding
            3. 1x1 conv with stride 1 and valid padding
        """
        x = conv_layer(x, filters, kernel=3, stride=1, 
                            padding='same', n_layer=(n_block - 1) * 3 + 1)
        x = conv_layer(x, filters, kernel=1, stride=1, 
                            padding='valid', n_layer=(n_block - 1) * 3 + 2)
        x = conv_layer(x, n_classes, kernel=1, stride=1, 
                            padding='valid', n_layer=(n_block - 1) * 3 + 3)

        return x

    inputs = Input(shape=image_shape)

    # Dropout of 20 %
    if dropout & input_dropout:
        x = Dropout(rate=0.2, name='dropout0')(inputs)

        # Block 1: 96 filters
        x = block_a(x, filters=96, n_block=1)
    else:
        # Block 1: 96 filters
        x = block_a(inputs, filters=96, n_block=1)

    # Dropout of 50 %
    if dropout:
        x = Dropout(rate=0.5, name='dropout1')(x)

    # Block 2: 192 filters
    if depth == 'shallower':
        # Pre-logits block
        x = block_b(x, filters=192, n_block=2)
    else:
        x = block_a(x, filters=192, n_block=2)

        # Dropout of 50 %
        if dropout:
            x = Dropout(rate=0.5, name='dropout2')(x)

        # Block 3: 192 filters
        if depth == 'deeper':
            x = block_a(x, filters=192, n_block=3)
            # Pre-logits block
            x = block_b(x, filters=192, n_block=4)
        else:
            # Pre-logits block
            x = block_b(x, filters=192, n_block=3)

    # Global Average Pooling
    logits = GlobalAveragePooling2D(name='logits')(x)

    # Softmax
    predictions = Activation('softmax', name='softmax')(logits)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def allcnn_imagenet(image_shape, n_classes, dropout=False, weight_decay=None,
                    batch_norm=False, depth='orig', stride_conv1=4):
    """
    Defines the ImageNet version of the All convolutional network (All-CNN), 
    originally described in https://arxiv.org/abs/1412.6806. It is implemented 
    in a modular way in order to allow for more flexibility, for instance 
    modifying the depth of the network.

    Parameters
    ----------
    image_shape : int list
        Shape of the input images

    n_classes : int
        Number of classes in the training data set

    dropout : bool
        If True, Dropout regularization is included

    weight_decay : float
        L2 factor for the weight decay regularization or None

    batch_norm : bool
        If True, batch normalization is added after every convolutional layer

    depth : str
        Allows to specify a shallower or deeper architecture:
            - 'shallower': 1 convolutional block (A) + 1 output block (B)
            - 'deeper': 2 convolutional blocks (A) + 1 output block (B)

    Returns
    -------
    model : Model
        Keras model describing the architecture
    """

    def conv_layer(x, filters, kernel, stride, padding, n_layer):
        """
        Defines a convolutional block, formed by a convolutional layer, an
        optional batch normalization layer and a ReLU activation.

        Parameters
        ----------
        x : Tensor
            The input to the convolutional layer

        filters : int
            Number of filters (witdth) of the convolutional layer

        kernel : int
            Kernel size

        stride : int
            Stride of the convolution

        n_layer : int
            Layer number, with respect to the whole network
        
        Returns
        -------
        x : Tensor
            Output of the convolutional block
        """
        # Convolution
        x = Conv2D(filters=filters,
                   kernel_size=(kernel, kernel),
                   strides=(stride, stride),
                   padding=padding,
                   activation='linear',   
                   use_bias=True,
                   kernel_initializer='glorot_uniform',
                   bias_initializer='zeros',
                   kernel_regularizer=l2reg(weight_decay),   
                   bias_regularizer=l2reg(weight_decay),
                   input_shape=image_shape,
                   name='conv%d' % n_layer)(x)
        
        # Batch Normalization
        if batch_norm:
            x = BatchNormalization(axis=3, 
                                   epsilon=1.001e-5,
                                   gamma_regularizer=l2reg(weight_decay),
                                   beta_regularizer=l2reg(weight_decay),
                                   name='conv%dbn' % n_layer)(x)

        # ReLU Activation
        x = Activation('relu', name='conv%drelu' % n_layer)(x)    

        return x

    def block_a(x, filters, k1, s1, n_block):
        """
        Defines a block of 3 convolutional layers:
            1. k1xk1 conv with stride s1
            2. 1x1 conv with stride 1
            3. 3x3 conv with stride 2
        """
        x = conv_layer(x, filters, kernel=k1, stride=s1, padding='same', 
                       n_layer=(n_block - 1) * 3 + 1)
        x = conv_layer(x, filters, kernel=1, stride=1, padding='same', 
                       n_layer=(n_block - 1) * 3 + 2)
        x = conv_layer(x, filters, kernel=3, stride=2, padding='same', 
                       n_layer=(n_block - 1) * 3 + 3)

        return x

    def block_b(x, filters, n_block):
        """
        Defines a block of 3 convolutional layers:
            1. 3x3 conv with stride 1
            2. 1x1 conv with stride 1 and valid padding
            3. 1x1 conv with stride 1 and valid padding
        """
        x = conv_layer(x, filters, kernel=3, stride=1, padding='same', 
                       n_layer=(n_block - 1) * 3 + 1)
        x = conv_layer(x, filters, kernel=1, stride=1, padding='valid', 
                       n_layer=(n_block - 1) * 3 + 2)
        x = conv_layer(x, n_classes, kernel=1, stride=1, padding='valid', 
                       n_layer=(n_block - 1) * 3 + 3)

        return x

    inputs = Input(shape=image_shape)

    # Block 1: 96 filters
    x = block_a(inputs, filters=96, k1=11, s1=stride_conv1, n_block=1)

    # Block 2: 256 filters
    x = block_a(x, filters=256, k1=5, s1=1, n_block=2)

    # Block 3: 384 filters
    x = block_a(x, filters=384, k1=3, s1=1, n_block=3)

    if dropout:
        x = Dropout(rate=0.5, name='dropout1')(x)

    # Block 4: 1024 filters
    x = block_b(x, filters=1024, n_block=4)

    # Global Average Pooling
    logits = GlobalAveragePooling2D(name='logits')(x)

    # Softmax
    predictions = Activation('softmax', name='softmax')(logits)

    model = Model(inputs=inputs, outputs=predictions)

    return model


def retinal_bottleneck(model, N):
    """
        Defines a bottleneck with three convolutional 
        layers placed at tbe beginning of the current model.

        Parameters
        ----------
        model : the current model in which the bottleneck will be implemented
        N : Number of neurons in the second and third layer of the bottleneck
        
        Returns
        -------
        model_with_bottleneck : Model object of tensorflow.keras
            The model introduced as argument with the bottleneck integrated 
            at the beggining of it.
      """

    # reusing the original model input
    input_shape = model.input_shape[1:]
    x_bottleneck = Input(input_shape, name="input_1")

    # retina-net (bottleneck) is the architecture of below 
    retina_net_input_layer = Conv2D(32, (3, 3), padding='same', activation='relu')(x_bottleneck)  # N=64,32,24
    retina_net_hidden_layer = Conv2D(N, (3, 3), padding='same', activation='relu')(retina_net_input_layer)  # N=1,2,4,8,16,24
    retina_net_output_layer = Conv2D(N, (3, 3), padding='same', activation='relu')(retina_net_hidden_layer)  

    # reshaping input shape of first conv layer of original model
    gabor_layer = model.layers[1]
    new_gabor_layer = Conv2D(
        filters=gabor_layer.filters,
        kernel_size=gabor_layer.kernel_size,
        kernel_initializer=gabor_layer.kernel_initializer,
        kernel_regularizer=gabor_layer.kernel_regularizer,
        strides=gabor_layer.strides,
        padding=gabor_layer.padding,
        activation=gabor_layer.activation,
        name=gabor_layer.name,
        use_bias=gabor_layer.use_bias,
        bias_initializer=gabor_layer.bias_initializer,
        bias_regularizer=gabor_layer.bias_regularizer,
        input_shape=retina_net_output_layer.shape
    )

    # connecting retina-net as the input of original model
    x = new_gabor_layer(retina_net_output_layer)
    for layer in model.layers[2:]:
        x = layer(x)

    bottleneck_model = Model(inputs=x_bottleneck, outputs=x, name='Model_with_Retinal_Bottleneck')
    
    print("Retinal Bottleneck added to the model")
    bottleneck_model.summary()

    return bottleneck_model


   
def recurrent_connections_LSTM_back(model):
    print("---------------\nRecurrent Connections \n-----------------")

    feedback = [None for _ in range(len(model.layers))]  # Initialize feedback input vector from higher layers
    layer_number = 0
    last_recurrent_layer = 7
    cont_max_pooling_layers = 0
    input_shape = model.input_shape[1:]

    inputs = Input(shape=input_shape)
    x = inputs

    if model.name == "Model_with_Retinal_Bottleneck":
        text = "Model with Recurrent Connections and Retinal Bottleneck"
        n = 4
    else:
        text = "Model with Recurrent Connections"
        n = 1

    # Add bottleneck layers without recurrent connections
    for layer in model.layers[1:n]:
        x = layer(x)


    # Add recurrent connections after bottleneck
    for layer in model.layers[n:n+last_recurrent_layer]:
        if isinstance(layer, Conv2D):
            # Add time dimension for LSTM
            conv_output_shape = x.shape[1:] 
            feedback_output_shape = feedback[layer_number+1].shape[1:] if feedback[layer_number+1] is not None else (0, 0, 0)

            k_size_dims = conv_output_shape[:2]
            total_channels = conv_output_shape[2] + feedback_output_shape[2]
            
            # Reshape to add time dimension
            x = Reshape((1,) + conv_output_shape)(x)

            # Adding the output feedback of next layer as part of the input of current layer
            if feedback[layer_number+1] is not None:
                feedback_reshaped = Reshape((1,) + feedback_output_shape)(feedback[layer_number+1])
                # Ensure dimensions match for concatenation
                feedback_reshaped = tf.image.resize(feedback_reshaped, size=k_size_dims)  # Align spatial dimensions
                x = tf.concat([x, feedback_reshaped], axis=-1)
            
            # Convert from Conv2D to ConvLSTM2D
            x = ConvLSTM2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                padding=layer.padding,
                strides=layer.strides,
                activation=layer.activation,
                return_sequences=True,
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate,
                recurrent_activation='sigmoid',
                use_bias=layer.use_bias,
                recurrent_initializer='orthogonal',
                bias_initializer=layer.bias_initializer,
                unit_forget_bias=True,
                kernel_regularizer=layer.kernel_regularizer,
                recurrent_regularizer=None,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=None,
                kernel_constraint=layer.kernel_constraint,
                recurrent_constraint=None,
                bias_constraint=layer.bias_constraint,
                dropout=0.0,
                recurrent_dropout=0.0,
                return_state=False,
                go_backwards=False,
                stateful=False
            )(x)

            # Reshape to delete the time dimension
            conv_output_shape = x.shape[-3:] 
            x = Reshape(conv_output_shape)(x)

            feedback[layer_number] = x
            layer_number += 1

            # Reset layer_number if it reaches the length of the model layers
            if layer_number == last_recurrent_layer - cont_max_pooling_layers :
                layer_number = 0
        else:
            x = layer(x)
            cont_max_pooling_layers+= 1


    for layer in model.layers[n+last_recurrent_layer+4::]:
        x = layer(x)

    # Creating the model with all the ConvLSTM2D layers
    model_with_recurrent = Model(inputs=inputs, outputs=x)

    # Defining a function of the summary of the model without the reshape layers
    def custom_summary(model):
        print("Model Summary:")
        for layer in model.layers:
            if not isinstance(layer, Reshape):
                layer_type = type(layer).__name__
                output_shape = layer.output_shape
                num_params = layer.count_params()
                print(f"{layer_type:20} | Output shape: {output_shape} | Params: {num_params}")

    print(text)
    custom_summary(model_with_recurrent)
    # model_with_recurrent.summary()

    return model_with_recurrent



def recurrent_connections_LSTM(model):
    print("---------------\nRecurrent Connections \n-----------------")

    last_recurrent_layer = 1
    input_shape = model.input_shape[1:]

    inputs = Input(shape=input_shape)
    x = inputs

    if model.name == "Model_with_Retinal_Bottleneck":
        text = "Model with Recurrent Connections and Retinal Bottleneck"
        n = 4
    else:
        text = "Model with Recurrent Connections"
        n = 1

    # Add bottlene4ck layers without recurrent connections
    for layer in model.layers[1:n]:
        x = layer(x)


    # Add recurrent connections after bottleneck
    for layer in model.layers[n:n+last_recurrent_layer]:
        if isinstance(layer, Conv2D):
            # Add time dimension for LSTM
            conv_output_shape = x.shape[1:] 
            
            # Reshape to add time dimension
            x = Reshape((1,) + conv_output_shape)(x)
            
            # Convert from Conv2D to ConvLSTM2D
            x = ConvLSTM2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                padding=layer.padding,
                strides=layer.strides,
                activation='tanh',
                return_sequences=False,
                data_format=layer.data_format,
                dilation_rate=layer.dilation_rate,
                recurrent_activation='sigmoid',
                use_bias=layer.use_bias,
                recurrent_initializer='orthogonal',
                bias_initializer=layer.bias_initializer,
                unit_forget_bias=True,
                kernel_regularizer=layer.kernel_regularizer,
                recurrent_regularizer=None,
                bias_regularizer=layer.bias_regularizer,
                activity_regularizer=None,
                kernel_constraint=layer.kernel_constraint,
                recurrent_constraint=None,
                bias_constraint=layer.bias_constraint,
                dropout=0.0,
                recurrent_dropout=0.0,
                return_state=False,
                go_backwards=False,
                stateful=False
            )(x)


            # Pooling layers are processed 
        else:
            x = layer(x)

    for layer in model.layers[n+last_recurrent_layer::]:
        x = layer(x)
    

    # Creating the model with all the ConvLSTM2D layers
    model_with_recurrent = Model(inputs=inputs, outputs=x)

    model_with_recurrent.summary()

    return model_with_recurrent



def l2reg(wd):
    """
    Defines the regularizer for the kernel and bias parameters. It can be
    either None or L2 (weight decay, if the coefficient is divided by 2. [see
    https://bbabenko.github.io/weight-decay/]).

    Parameters
    ----------
    wd : float
        L2 regularization factor.

    Returns
    -------
    l2 : function
        Regularization function or None
    """
    if wd is None:
        return None
    else:
        return l2(wd / 2.)