import numpy as np
import cv2

np.random.seed(1)
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

batch_size = 128
nb_classes = 10
epoches = 1
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
f = np.load('mnist.npz')
x_train, y_train = f['x_train'], f['y_train']
x_test, y_test = f['x_test'], f['y_test']
f.close()
print('x_train', x_test.shape)
# 根据不同的backend定下不同的格式
if K.image_dim_ordering() == 'th':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# print('x_train', y_train[0, :, :, 0], y_train.dtype)
cv2.imwrite('x_train_0.jpg', x_train[0, :, :, :])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape, type(x_train))
print('y_train shape:', y_train.shape, type(y_train))
print('x_test shape:', x_test.shape, type(x_test))
print('y_test shape:', y_test.shape, type(y_test))

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# print('x_train', y_train.shape, y_train.dtype)


# 转换为one_hot类型
y_train = np_utils.to_categorical(y_train, nb_classes)
print('y_train to categorical', y_train.shape, type(y_train))
print(y_train[0])
y_test = np_utils.to_categorical(y_test, nb_classes)

# 构建模型
model = Sequential()
conv1_1 = Conv2D(filters=nb_filters, kernel_size=kernel_size, strides=1, padding='SAME', input_shape=input_shape,
                 name='conv1_1')
model.add(conv1_1)
conv1_2 = Conv2D(filters=60, kernel_size=kernel_size, strides=2, padding='VALID', name='conv1_2')
model.add(conv1_2)
model.add(Activation('relu'))
# model.add()
model.add((MaxPooling2D(pool_size=(2, 2))))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
print('model.layers', model.layers)
# 编译模型
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
print('conv1_1', conv1_1.input, conv1_1.output, conv1_1.name)
print('conv1_2', conv1_2.input, conv1_2.output, conv1_2.name)
# 训练模型
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epoches, verbose=2, validation_data=(x_test, y_test))
# model.fit_generator()
# model.save_weights('weight.h5')
# 评估模型
model.load_weights('weight.h5')
score = model.evaluate(x_test, y_test, verbose=1)
print('metrics_names', model.metrics_names)
print('score:', score, type(score))
print('Test score:', score[0])
print('Test accuracy:', score[1])

x_1 = cv2.imread('x_train_0.jpg', flags=cv2.IMREAD_GRAYSCALE)
print(x_1.shape)
x_1 = x_1[np.newaxis, :, :, np.newaxis]
print('x_1', x_1.shape)
model1 = Sequential()

pred = model.predict(x_1, batch_size=1, verbose=1)
print('pred', pred, type(pred))
