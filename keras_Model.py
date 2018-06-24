import numpy as np
import cv2
from keras.layers import Dense, Input, Dropout, Conv2D, LSTM, MaxPool2D
from keras.layers import concatenate, add
from keras.models import Model
from keras.utils import np_utils
from keras import optimizers

# x_input = Input(shape=(256, 256, 3))
# y_output = Conv2D(128, (3, 3), padding='SAME')(x_input)
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
model.compile(optimizer=optimizers.Adadelta(lr=1e-3), loss='categorical_crossentropy', metrics=['acc'],
              loss_weights=[1, 0.2])
print('summary', model.summary())
history = model.fit([x_train, x_train], [y_train, y_train], batch_size=32, epochs=1, verbose=1, validation_split=0.1)
score = model.evaluate([x_test, x_test], [y_test, y_test], verbose=1)
# 共享视觉模型
digit_input = Input(shape=(28, 28, 1))
x = Conv2D(64, (3, 3), padding='SAME')(digit_input)
x = Conv2D(64, (3, 3), padding='SAME')(x)
print('x', x)
x = MaxPool2D((2, 2), strides=2)(x)
print('x', x)

print('history', type(history), list(history.history.keys()), history.history)
model.save_weights('model.h5')

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
