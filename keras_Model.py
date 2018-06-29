import numpy as np
import cv2
import tensorflow as tf
from keras.layers import Dense, Input, Dropout, Conv2D, LSTM, MaxPool2D, Lambda
from keras.layers import concatenate, add
from keras.models import Model, load_model, save_model
from keras.utils import np_utils
from keras import optimizers
from keras import losses
from keras import metrics
from keras import objectives
from keras import backend as K
from keras.applications import vgg16
from keras_custom_layer import CustomLayer

# 测试keras自带的vgg16
# a = vgg16.VGG16()
# print('vgg16.VGG16', type(a))
# 测试卷积网络的输入输出
# x_input = Input(shape=(256, 256, 3))
# y_output = Conv2D(128, (3, 3), padding='SAME')(x_input)
# print('y_output', y_output)

# 测试Lambda层
x_input = Input(shape=(10, 3))
# 测试自定义网络的输入输出
# x_input = Input(shape=(10, 3))
# a = CustomLayer(output_dim=5)
# print('aaa')
# y_output = a(x_input)
# print('y_output', y_output)
# 残差网络
x = Input(shape=(256, 256, 3))
y = Conv2D(3, (3, 3), padding='SAME')(x)
z = add([x, y])
print('z', z)

# Mnist
f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)
print('x_train', x_train.shape)
print('y_train', y_train.shape, y_train[0, :])

f.close()
# 简单的全连接网络
inputs = Input(shape=(784,))
print('inputs', inputs, type(inputs))
b = Dense(64, activation='relu')
print('b', b, type(b))
x = b(inputs)
x = Dense(784, activation='relu')(x)
print('x', x, type(x))
auxiliary_input = Input(shape=(784,), name='aux_input')

# 共享层
share = Dense(256, activation='relu', name='share')
print('share name', type(share), share.name)
s1 = share(x)
print('s1', s1)
s2 = share(auxiliary_input)
print('s2', s2)
print('share', share.get_input_at(0))
print('share', share.get_output_at(1))
concat = concatenate([s1, s2], axis=-1)
print('concat', type(concat), concat)

predictions = Dense(10, activation='softmax')(x)
aux_output = Dense(10, activation='sigmoid')(concat)
# 单输入单输出模型
# model = Model(inputs=inputs, outputs=predictions)
# print('model', model, type(model))
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
# history = model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=2, validation_split=0.1)
# score = model.evaluate(x_test, y_test, verbose=1)

# 多输入多输出模型
model = Model(inputs=[inputs, auxiliary_input], outputs=[predictions, aux_output])


# 自定义loss 方法一、定义一个loss函数
def custom_loss(y_true, y_pred):
    print('custom_loss y_true:', y_true)
    a = K.mean((y_pred - y_true), axis=-1)
    print('a', a)
    return a


def custom_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    print('target', target)
    print('output', output)
    output_dimensions = list(range(len(output.get_shape())))
    print('output_dimensions', output_dimensions)
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        print('reduce_sum', output)
        # manual computation of crossentropy
        _epsilon = 1e-7
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        print('clip_by_value', output)
        return - tf.reduce_sum(target * tf.log(output), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)


# model.compile(optimizer=optimizers.Adadelta(lr=1e-3), loss=custom_categorical_crossentropy, metrics=['acc'],
#               loss_weights=[1, 0.2])
model.compile(optimizer=optimizers.Adadelta(lr=1e-3),
              loss=losses.categorical_crossentropy,
              metrics=[metrics.mean_squared_error, metrics.binary_crossentropy],
              loss_weights=[1, 0.2])
print('summary', model.summary())
for i in model.layers:
    print(i.name, ' ', end='')
print('layers', model.layers)
model.save('model_save_before_fit.h5')
save_model(model, 'model_save_model_function_before_fit.h5')
history = model.fit([x_train, x_train], [y_train, y_train], batch_size=32, epochs=1, verbose=2, validation_split=0.1)
score = model.evaluate([x_test, x_test], [y_test, y_test], verbose=1)
# 共享视觉模型
digit_input = Input(shape=(28, 28, 1))
x = Conv2D(64, (3, 3), padding='SAME')(digit_input)
x = Conv2D(64, (3, 3), padding='SAME')(x)
print('x', x)
x = MaxPool2D((2, 2), strides=2)(x)
print('x', x)
print('history', type(history), list(history.history.keys()), history.history)
model.save_weights('model_weights.h5')

print('metrics_names', model.metrics_names)
print('score:', score, type(score))
print('Test score:', score[0])
print('Test accuracy:', score[1])

x_1 = cv2.imread('x_train_0.jpg', flags=cv2.IMREAD_GRAYSCALE)
print(x_1.shape)
x_1 = np.expand_dims(x_1.flatten(), axis=0)
print('x_1', x_1.shape)

# pred = model.predict(x_1, batch_size=1, verbose=1)
pred = model.predict([x_1, x_1], batch_size=1, verbose=1)
print('pred', pred, type(pred))
