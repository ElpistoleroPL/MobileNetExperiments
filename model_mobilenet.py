from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import warnings

from keras.models import Model
from keras.layers import Input, Activation, Dropout, Reshape, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Conv2D
from keras.layers.core import Lambda
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.callbacks import EarlyStopping, CSVLogger, Callback

from keras import initializers, regularizers, constraints
from keras.utils import conv_utils, get_file
from keras.engine.topology import get_source_inputs, InputSpec
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K
from keras import layers

def relu6(x):
    return K.relu(x, max_value=6)


def preprocess_input(x):
    return imagenet_utils.preprocess_input(x, mode='tf')


class DepthwiseConv2D(Conv2D):
    """Depthwise separable 2D convolution.

    Depthwise Separable convolutions consists in performing
    just the first step in a depthwise spatial convolution
    (which acts on each input channel separately).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.

    # Arguments
        kernel_size: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: one of `'valid'` or `'same'` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filters_in * depth_multiplier`.
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be 'channels_last'.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. 'linear' activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its 'activation').
            (see [regularizer](../regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).

    # Input shape
        4D tensor with shape:
        `[batch, channels, rows, cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, rows, cols, channels]` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `[batch, filters, new_rows, new_cols]` if data_format='channels_first'
        or 4D tensor with shape:
        `[batch, new_rows, new_cols, filters]` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv2D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) < 4:
            raise ValueError('Inputs to `DepthwiseConv2D` should have rank 4. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = 3
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv2D` '
                             'should be defined. Found `None`.')
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        outputs = K.depthwise_conv2d(
            inputs,
            self.depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == 'channels_last':
            rows = input_shape[1]
            cols = input_shape[2]
            out_filters = input_shape[3] * self.depth_multiplier

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, rows, cols)
        elif self.data_format == 'channels_last':
            return (input_shape[0], rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv2D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config


def MobileNet(input_shape=None,
              alpha=1.0,
              depth_multiplier=1,
              dropout=1e-3,
              include_top=True,
              weights='imagenet',
              input_tensor=None,
              pooling=None,
              classes=1000,
              mode=0):

    if K.backend() != 'tensorflow':
        raise RuntimeError('Only TensorFlow backend is currently supported, '
                           'as other backends do not support '
                           'depthwise convolution.')

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if K.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if K.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if weights == 'imagenet':
        if depth_multiplier != 1:
            raise ValueError('If imagenet weights are being loaded, '
                             'depth multiplier must be 1')

        if alpha not in [0.25, 0.50, 0.75, 1.0]:
            raise ValueError('If imagenet weights are being loaded, '
                             'alpha can be one of'
                             '`0.25`, `0.50`, `0.75` or `1.0` only.')

        if rows != cols or rows not in [128, 160, 192, 224]:
            raise ValueError('If imagenet weights are being loaded, '
                             'input must have a static square shape (one of '
                             '(128,128), (160,160), (192,192), or (224, 224)).'
                             ' Input shape provided = %s' % (input_shape,))

    if K.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    print("Mode: %d" % mode)
    all_layers = []
            
    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1, mode=mode, all_layers=all_layers)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=2, mode=mode, all_layers=all_layers)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3, mode=mode, all_layers=all_layers)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier,
                              strides=(2, 2), block_id=4, mode=mode, all_layers=all_layers)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5, mode=mode, all_layers=all_layers)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6, mode=mode, all_layers=all_layers)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7, mode=mode, all_layers=all_layers)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8, mode=mode, all_layers=all_layers)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9, mode=mode, all_layers=all_layers)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10, mode=mode, all_layers=all_layers)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11, mode=mode, all_layers=all_layers)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12, mode=mode, all_layers=all_layers)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13, mode=mode, all_layers=all_layers)

    if include_top:
        if K.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Conv2D(classes, (1, 1),
                   padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((classes,), name='reshape_2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise ValueError('Weights for "channels_last" format '
                             'are not available.')
        if alpha == 1.0:
            alpha_text = '1_0'
        elif alpha == 0.75:
            alpha_text = '7_5'
        elif alpha == 0.50:
            alpha_text = '5_0'
        else:
            alpha_text = '2_5'

        if include_top:
            model_name = 'mobilenet_%s_%d_tf.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        else:
            model_name = 'mobilenet_%s_%d_tf_no_top.h5' % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides,
               name='conv1')(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)

def pad_depth(x, desired_channels, block_id):
    print("===")
    print(x.get_shape().as_list())
    y = K.zeros_like(x, name='conv_iden_zeros_%d' % block_id)
    print(y.get_shape().as_list())
    new_channels = desired_channels - x.shape.as_list()[-1]
    y = y[:,:,:,:new_channels]
    print(y.get_shape().as_list())
    return layers.concatenate([x,y], name='conv_iden_conc_%d' % block_id)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1, mode=0, all_layers=[]):
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConv2D((3, 3),
                        padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    if mode!=7 and mode!=8:
        x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    if mode==7:
        mode=2
    elif mode==8:
        mode=5

    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)
    
    #orginal
    if mode == 0:
        return x
    #only simple identity when all dimmensions are the same
    if mode == 1:
        if strides == (1, 1) and inputs.get_shape().as_list()[3] == pointwise_conv_filters:
            x = layers.add([x, inputs])
            return x

    #simple identity when all dimmensions are the same + conv when depth is diffrent
    if mode == 2:
        if strides == (1, 1):
            if inputs.get_shape().as_list()[3] != pointwise_conv_filters:
                x_bn = Conv2D(pointwise_conv_filters, (1, 1),
                       padding='same',
                       use_bias=False,
                       strides=(1, 1),
                       name='conv_iden_pw_%d' % block_id)(inputs)
                inputs = BatchNormalization(axis=channel_axis, name='conv_iden_%d_bn' % block_id)(x_bn)
            return layers.add([x, inputs])
    
    #simple identity when all dimmensions are the same + conv with stride if different
    if mode == 3:
        if inputs.get_shape().as_list()[3] != pointwise_conv_filters:
            x_bn = Conv2D(pointwise_conv_filters, (1, 1),
                   padding='same',
                   use_bias=False,
                   strides=strides,
                   name='conv_iden_pw_%d' % block_id)(inputs)
            inputs = BatchNormalization(axis=channel_axis, name='conv_iden_%d_bn' % block_id)(x_bn)
        return layers.add([x, inputs])
        
    #simple dense
    if mode == 4:
        r = x
        i = 0
        for l in all_layers:
            if l.get_shape().as_list() == x.get_shape().as_list():
                r = layers.add([r, l])
                i += 1
        print("Merged %d out of %d layers" %(i, len(all_layers)))
        x = r
        
    #zero pad when depth is not the same
    if mode == 5:
        if strides == (1, 1):
            if inputs.get_shape().as_list()[3] != pointwise_conv_filters:
                desired_channels = x.shape.as_list()[-1]
                s_skip = Lambda(pad_depth,
                                name='conv_iden_lambda_%d' % block_id,
                                arguments={'desired_channels':desired_channels, 'block_id':block_id})(inputs)

                return layers.add([s_skip, x], name='some_add')
            return layers.add([x, inputs])
    if mode == 6:
        if strides == (1, 1):
            if inputs.get_shape().as_list()[3] != pointwise_conv_filters:
                x_bn = Conv2D(pointwise_conv_filters, (1, 1),
                       padding='same',
                       use_bias=False,
                       strides=(1, 1),
                       name='conv_iden_pw_%d' % block_id)(inputs)
                inputs = BatchNormalization(axis=channel_axis, name='conv_iden_%d_bn' % block_id)(x_bn)
            added=layers.add([x, inputs])    
            return Activation(relu6, name='conv_identity_%d_relu' % block_id)(added) 
    
    all_layers.append(x)
    #print(x.get_shape().as_list())
    return x
