"""
Dilated Resnet implementation adapted from the TF 2.0 Resnet implementation at
https://github.com/YunYang1994/TensorFlow2.0-Examples/blob/master/3-Neural_Network_Architecture/resnet.py
to match the official Dilated Resnet pytorch implementation at
https://github.com/fyu/drn/blob/master/drn.py
"""
import tensorflow as tf
from model_lib.layers.wrap_common_layers_sn import spectralnorm_wrapper


# __all__ = ['DRN', 'drn26', 'drn42', 'drn58']


class BasicBlock(tf.keras.Model):
    expansion = 1

    def __init__(self, out_channels, strides=1, downsample=None,
                 dilation=(1, 1), l2_reg=1e-4, residual=True, dropout=0.0,
                 **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            out_channels, kernel_size=3,
            strides=strides, padding="same",
            dilation_rate=dilation[0],
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

        self.conv2 = tf.keras.layers.Conv2D(
            out_channels, kernel_size=3,
            strides=1, padding="same",
            dilation_rate=dilation[1],
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.downsample = downsample
        self.residual = residual

        self.conv1 = spectralnorm_wrapper(self.conv1, **kwargs)
        self.bn1 = spectralnorm_wrapper(self.bn1, **kwargs)
        self.conv2 = spectralnorm_wrapper(self.conv2, **kwargs)
        self.bn2 = spectralnorm_wrapper(self.bn2, **kwargs)

    def call(self, x, training=False):
        residual = x
        out = self.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)

        if self.downsample is not None:
            residual = self.downsample(x, training)
        if self.residual:
            out += residual

        return self.relu(out)


class Bottleneck(tf.keras.Model):
    expansion = 4

    def __init__(self, out_channels, strides=1, downsample=None,
                 dilation=(1, 1), l2_reg=1e-4, residual=True, dropout=0.0,
                 **kwargs):
        super(Bottleneck, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            out_channels, 1, 1, use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            out_channels, 3, strides, padding="same",
            dilation_rate=dilation[1], use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(
            out_channels * self.expansion, 1, 1,
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.relu = tf.keras.layers.ReLU()
        self.downsample = downsample

        self.conv1 = spectralnorm_wrapper(self.conv1, **kwargs)
        self.bn1 = spectralnorm_wrapper(self.bn1, **kwargs)
        self.conv2 = spectralnorm_wrapper(self.conv2, **kwargs)
        self.bn2 = spectralnorm_wrapper(self.bn2, **kwargs)
        self.conv3 = spectralnorm_wrapper(self.conv3, **kwargs)
        self.bn3 = spectralnorm_wrapper(self.bn3, **kwargs)

    def call(self, x, training=False):
        residual = x

        out = self.relu(self.bn1(self.conv1(x), training))
        out = self.relu(self.bn2(self.conv2(out), training))
        out = self.bn3(self.conv3(out), training)

        if self.downsample is not None:
            residual = self.downsample(x, training)

        out += residual
        return self.relu(out)


class DRN(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10,
                 channels=(16, 32, 64, 128, 256, 512, 512, 512),
                 out_map=False,
                 out_middle=False,
                 pool_size=28,
                 arch='D',
                 l2_reg=1e-4,
                 dropout: float = 0.0,
                 **kwargs
                 ):
        super(DRN, self).__init__()
        self.in_channels = channels[0]
        self.out_map = out_map
        self.out_dim = channels[-1]
        self.out_middle = out_middle
        self.arch = arch

        if arch == 'C':
            self.conv1 = tf.keras.layers.Conv2D(
                channels[0], 7, 1, padding="same", use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            self.conv1 = spectralnorm_wrapper(self.conv1, **kwargs)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn1 = spectralnorm_wrapper(self.bn1, **kwargs)
            self.relu = tf.keras.layers.ReLU()

            self.layer1 = self._make_layer(
                block, channels[0], num_blocks[0], l2_reg=l2_reg, stride=1)
            self.layer2 = self._make_layer(
                block, channels[1], num_blocks[1], l2_reg=l2_reg, stride=2)
        elif arch == 'D':
            self.conv1 = tf.keras.layers.Conv2D(
                channels[0], 7, 1, padding="same", use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            self.conv1 = spectralnorm_wrapper(self.conv1, **kwargs)
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.bn1 = spectralnorm_wrapper(self.bn1, **kwargs)
            self.relu = tf.keras.layers.ReLU()

            self.layer1 = self._make_conv_layer(
                channels[0], num_blocks[0], l2_reg=l2_reg, stride=1, **kwargs)
            self.layer2 = self._make_conv_layer(
                channels[1], num_blocks[1], l2_reg=l2_reg, stride=2, **kwargs)

        self.layer3 = self._make_layer(
            block, channels[2], num_blocks[2], l2_reg=l2_reg, stride=2,
            **kwargs)
        self.layer4 = self._make_layer(
            block, channels[3], num_blocks[3], l2_reg=l2_reg, stride=2,
            **kwargs)
        self.layer5 = self._make_layer(
            block, channels[4], num_blocks[4],
            l2_reg=l2_reg, dilation=2, new_level=False, **kwargs)
        self.layer6 = None if num_blocks[5] == 0 else self._make_layer(
            block, channels[5], num_blocks[5],
            l2_reg=l2_reg, dilation=4, new_level=False, **kwargs)

        if arch == 'C':
            self.layer7 = None if num_blocks[6] == 0 else self._make_layer(
                BasicBlock, channels[6], num_blocks[6], dilation=2,
                l2_reg=l2_reg, new_level=False, residual=False, **kwargs)
            self.layer8 = None if num_blocks[7] == 0 else self._make_layer(
                BasicBlock, channels[7], num_blocks[7], dilation=1,
                l2_reg=l2_reg, new_level=False, residual=False, **kwargs)
        elif arch == 'D':
            self.layer7 = None if num_blocks[6] == 0 else self._make_conv_layer(
                channels[6], num_blocks[6], dilation=2, l2_reg=l2_reg, **kwargs)
            self.layer8 = None if num_blocks[7] == 0 else self._make_conv_layer(
                channels[7], num_blocks[7], dilation=1, l2_reg=l2_reg, **kwargs)

        # TODO(mattiasegu): place dropout layers for MCDO
        self.dropout = tf.keras.layers.Dropout(rate=dropout)

        self.avgpool = tf.keras.layers.AveragePooling2D(pool_size)
        self.fc = tf.keras.layers.Conv2D(
            num_classes, 1, 1,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.fc = spectralnorm_wrapper(self.fc, **kwargs)

        # TODO(mattiasegu): check that output size is the same as in paper

    def _make_layer(self, block, out_channels, num_blocks, stride=1, dilation=1,
                    l2_reg=1e-4, new_level=True, residual=True, **kwargs):
        assert dilation == 1 or dilation % 2 == 0
        # Adds a shortcut between input and residual block
        downsample = None
        if stride != 1 or self.in_channels != block.expansion * out_channels:
            conv = tf.keras.layers.Conv2D(
                block.expansion * out_channels,
                kernel_size=1, strides=stride,
                use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            bn = tf.keras.layers.BatchNormalization()

            downsample = tf.keras.Sequential([
                spectralnorm_wrapper(conv, **kwargs),
                spectralnorm_wrapper(bn, **kwargs)]
            )

        layers = []
        layers.append(block(
            out_channels, stride, downsample,
            dilation=(1, 1) if dilation == 1 else (
                dilation // 2 if new_level else dilation, dilation),
            residual=residual, l2_reg=l2_reg, **kwargs))
        self.in_channels = out_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(out_channels,
                                dilation=(dilation, dilation),
                                residual=residual, l2_reg=l2_reg, **kwargs))
        return tf.keras.Sequential(layers)

    def _make_conv_layer(self, channels, num_convs,
                         stride=1, dilation=1, l2_reg=1e-4, **kwargs):
        modules = []
        for i in range(num_convs):
            conv = tf.keras.layers.Conv2D(
                channels, 3, strides=stride if i == 0 else 1,
                padding="same", dilation_rate=dilation, use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            bn = tf.keras.layers.BatchNormalization()
            modules.extend([
                spectralnorm_wrapper(conv, **kwargs),
                spectralnorm_wrapper(bn, **kwargs),
                tf.keras.layers.ReLU()])
            self.in_channels = channels
        return tf.keras.Sequential(modules)

    def call(self, x, training=False):
        y = list()

        if self.arch == 'C' or self.arch == 'D':
            x = self.conv1(x)
            x = self.bn1(x, training)
            x = self.relu(x)

        x = self.layer1(x, training)
        x = self.dropout(x)
        y.append(x)
        x = self.layer2(x, training)
        x = self.dropout(x)
        y.append(x)

        x = self.layer3(x, training)
        x = self.dropout(x)
        y.append(x)

        x = self.layer4(x, training)
        x = self.dropout(x)
        y.append(x)

        x = self.layer5(x, training)
        y.append(x)

        if self.layer6 is not None:
            x = self.dropout(x)
            x = self.layer6(x, training)
            y.append(x)

        if self.layer7 is not None:
            x = self.dropout(x)
            x = self.layer7(x, training)
            y.append(x)

        if self.layer8 is not None:
            x = self.dropout(x)
            x = self.layer8(x, training)
            y.append(x)

        if self.out_map:
            x = self.fc(x)
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = tf.reshape(x, (x.shape[0], -1))

        if self.out_middle:
            return x, y
        else:
            return x


class DRN_A(tf.keras.Model):
    def __init__(self, block, num_blocks, num_classes=10, l2_reg=1e-4,
                 dropout=0.0, **kwargs):
        super(DRN_A, self).__init__()
        self.in_channels = 64
        self.out_dim = 512 * block.expansion

        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.conv1 = tf.keras.layers.Conv2D(
            64, 7, 2, padding="same", use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.conv1 = spectralnorm_wrapper(self.conv1, **kwargs)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn1 = spectralnorm_wrapper(self.bn1, **kwargs)
        self.relu = tf.keras.layers.ReLU()
        self.maxpool = tf.keras.layers.MaxPool2D(3, 2, padding="same")
        self.layer1 = self._make_layer(block, 64, num_blocks[0], l2_reg=l2_reg, **kwargs)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       l2_reg=l2_reg, **kwargs)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=1,
                                       dilation=2, l2_reg=l2_reg, **kwargs)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1,
                                       dilation=4, l2_reg=l2_reg, **kwargs)
        # TODO(mattiasegu): place dropout layers for MCDO

        self.avgpool = tf.keras.layers.AveragePooling2D(28, strides=1)
        self.fc = tf.keras.layers.Dense(
            num_classes,
            kernel_initializer='he_normal',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
        self.fc = spectralnorm_wrapper(self.fc, **kwargs)

    def _make_layer(self, block, out_channels, num_blocks,
                    stride=1, dilation=1, l2_reg=1e-4, **kwargs):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            conv = tf.keras.layers.Conv2D(
                    out_channels * block.expansion, 1, stride,
                    use_bias=False,
                    kernel_initializer='he_normal',
                    kernel_regularizer=tf.keras.regularizers.l2(l2_reg))
            bn = tf.keras.layers.BatchNormalization()

            downsample = tf.keras.Sequential([
                spectralnorm_wrapper(conv, **kwargs),
                spectralnorm_wrapper(bn, **kwargs)
            ])

        layers = []
        layers.append(block(out_channels, stride, downsample, l2_reg=l2_reg,
                            **kwargs))
        self.in_channels = out_channels * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(out_channels, l2_reg=l2_reg,
                                dilation=(dilation, dilation), **kwargs))

        return tf.keras.Sequential(layers)

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x, training)
        x = self.dropout(x, training)
        x = self.layer2(x, training)
        x = self.dropout(x, training)
        x = self.layer3(x, training)
        x = self.dropout(x, training)
        x = self.layer4(x, training)

        x = self.avgpool(x)
        x = tf.reshape(x, (x.shape[0], -1))
        x = self.fc(x)

        return x


def drn_a_50(**kwargs):
    model = DRN_A(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def drn_c_26(**kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='C', **kwargs)
    return model


def drn_c_42(**kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    return model


def drn_c_58(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='C', **kwargs)
    return model


def drn_d_22(**kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 1, 1], arch='D', **kwargs)
    return model


def drn_d_24(**kwargs):
    model = DRN(BasicBlock, [1, 1, 2, 2, 2, 2, 2, 2], arch='D', **kwargs)
    return model


def drn_d_38(**kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    return model


def drn_d_40(**kwargs):
    model = DRN(BasicBlock, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    return model


def drn_d_54(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 1, 1], arch='D', **kwargs)
    return model


def drn_d_56(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 6, 3, 2, 2], arch='D', **kwargs)
    return model


def drn_d_105(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 1, 1], arch='D', **kwargs)
    return model


def drn_d_107(**kwargs):
    model = DRN(Bottleneck, [1, 1, 3, 4, 23, 3, 2, 2], arch='D', **kwargs)
    return model
