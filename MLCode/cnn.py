import numpy as np

path = "project/"
trainCSV = path + "train.csv"# The dataset can be changed by loading different dataset
testCSV = path +"test.csv"

train_csv = np.genfromtxt(trainCSV, delimiter=',')
test_csv = np.genfromtxt(testCSV, delimiter=',')

train_data = train_csv[0:30000,9:438]
train_label = train_csv[0:30000,1]
test_data = test_csv[:,9:438]

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

batch_size = 100
nb_classes = 98
nb_epoch = 20

img_rows, img_cols = 33, 13
pool_size = (2, 2)

if K.image_dim_ordering() == 'th':
    X_train = train_data.reshape(train_data.shape[0], 1, img_rows, img_cols)
    X_test = test_data.reshape(test_data.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)
    X_test = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols,1)

Y_train = np_utils.to_categorical(train_label, nb_classes)

model = Sequential()

model.add(ZeroPadding2D((1,1), input_shape=input_shape))
model.add(Convolution2D(32, 3, 3, activation='relu', name='conv1',init='he_normal'))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(32, 3, 3, activation='relu', name='conv2',init='he_normal'))
model.add(MaxPooling2D((3, 3), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv3',init='he_normal'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv4',init='he_normal'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128*8, init='he_normal', activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(128*8,init='he_normal',activation='relu'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(98))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,verbose=1, validation_split=0.2)

train_data_1 = train_csv[30001:50282,9:438]
train_label_1 = train_csv[30001:50282,1]

if K.image_dim_ordering() == 'th':
    X_train1 = train_data_1.reshape(train_data_1.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train1 = train_data_1.reshape(train_data_1.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
Y_train1 = np_utils.to_categorical(train_label_1, nb_classes)

score = model.evaluate(X_train1, Y_train1, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

predicted = model.predict_classes(test_data)

test_index = test_csv[:,0]
result = np.vstack([test_index,predicted])
result = result.T

np.savetxt(path+"cnn_result.csv",result,fmt="%d",delimiter=",",header="Id,Character")
