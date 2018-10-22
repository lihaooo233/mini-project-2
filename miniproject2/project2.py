import tensorflow as tf
from tensorflow import keras
from flask import request
import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

class_ = ['roses', 'sunflowers']
class_idx = {'roses': 0, 'sunflowers': 1}

HEIGHT = 200
WIDTH = 200
DATA_PATH = './data'

def preprocess(images):
    pass


def get_data():
    # label the photos
    image_filenames = glob.glob('./sunflower\\*.jpg')
    train_labels = list(map(lambda x: class_idx[x.split('\\')[-1].split('.')[0]], image_filenames))

    x_train, x_val, x_test = list(), list(), list() # 6:2:2
    y_train, y_val, y_test = [], [], []

    category_count = [len(train_labels) - sum(train_labels), sum(train_labels)]
    category_split_count = [[0, 0, 0], [0, 0, 0]]  # train_count, val_count, test_count

    for i in range(len(train_labels)):
        if category_split_count[train_labels[i]][0] < category_count[train_labels[i]] * 0.6:
            x_train.append(image_filenames[i])
            y_train.append(train_labels[i])
            category_split_count[train_labels[i]][0] += 1
        elif category_split_count[train_labels[i]][1] < category_count[train_labels[i]] * 0.2:
            x_val.append(image_filenames[i])
            y_val.append(train_labels[i])
            category_split_count[train_labels[i]][1] += 1
        else:
            x_test.append(image_filenames[i])
            y_test.append(train_labels[i])
            category_split_count[train_labels[i]][2] += 1



    def get_image(filenames, labels):
        X = []
        for name in filenames:
            im = Image.open(name)
            im = im.resize((HEIGHT, WIDTH))
            im_array = np.array(im.convert('L'))
            im_array = im_array.reshape((HEIGHT, WIDTH, 1))
            X.append(im_array)
        X = np.array(X)
        labels = np.array(labels)
        permutation = np.random.permutation(labels.shape[0])
        return X[permutation, :, :], labels[permutation]

    return (get_image(x_train, y_train)), (get_image(x_val, y_val)), (get_image(x_test, y_test))


def setup_network2():
    model = keras.models.Sequential([
        # keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(16, activation=tf.nn.relu),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(1, activation=tf.nn.sigmoid)
        keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding="same",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                            activation=tf.nn.relu,
                            trainable=True,
                            #  (height, width, channels)
                            input_shape=(HEIGHT, WIDTH, 1),
                            # 如果 data_format='channels_last'， 输入 4D 张量，尺寸为 (batchsize, rows, cols, channels)。
                            data_format="channels_last"),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
                            padding='same', activation=tf.nn.relu,
                            trainable=True),
        keras.layers.MaxPool2D(pool_size=(3, 3), padding='same'),

        keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
                            padding='same', activation=tf.nn.relu,
                            trainable=True),
        keras.layers.MaxPool2D(pool_size=(3, 3), padding='same'),

        keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
                            padding='same', activation=tf.nn.relu,
                            trainable=True),
        keras.layers.MaxPool2D(pool_size=(3, 3), padding='same'),

        keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
                            padding='same', activation=tf.nn.relu,
                            trainable=True),
        keras.layers.MaxPool2D(pool_size=(3, 3), padding='same'),

        keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                            padding='same', activation=tf.nn.relu,
                            trainable=True),
        keras.layers.MaxPool2D(pool_size=(3, 3), padding='same'),


        keras.layers.Flatten(input_shape=(HEIGHT, WIDTH, 1)),
        #
        # keras.layers.Dense(1024, activation=tf.nn.relu),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(192, activation=tf.nn.relu),

        # keras.layers.Dense(2, activation=tf.nn.softmax)
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', "binary_crossentropy"]
                  # loss="categorical_crossentropy",
                  #  metrics=['accuracy', 'categorical_crossentropy']
                  )
    print(model.summary())
    return model


def setup_network1():
    model = keras.models.Sequential([
        # keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(16, activation=tf.nn.relu),
        # keras.layers.Dropout(0.5),
        # keras.layers.Dense(1, activation=tf.nn.sigmoid)
        keras.layers.Conv2D(filters=32, kernel_size=(5, 5), padding="same",
                            kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                            activation=tf.nn.relu,
                            trainable=True,
                            #  (height, width, channels)
                            input_shape=(HEIGHT, WIDTH, 1),
                            # 如果 data_format='channels_last'， 输入 4D 张量，尺寸为 (batchsize, rows, cols, channels)。
                            data_format="channels_last"),
        keras.layers.MaxPool2D(pool_size=(2, 2), padding='same'),

        keras.layers.Conv2D(filters=32, kernel_size=(5, 5),
                            padding='same', activation=tf.nn.relu,
                            trainable=True),
        keras.layers.MaxPool2D(pool_size=(3, 3), padding='same'),

        keras.layers.Conv2D(filters=64, kernel_size=(5, 5),
                            padding='same', activation=tf.nn.relu,
                            trainable=True),
        keras.layers.MaxPool2D(pool_size=(3, 3), padding='same'),


        keras.layers.Flatten(input_shape=(HEIGHT, WIDTH, 1)),
        #
        # keras.layers.Dense(1024, activation=tf.nn.relu),
        # keras.layers.Dropout(0.5),
        keras.layers.Dense(192, activation=tf.nn.relu),
        keras.layers.Dropout(0.5),
        # keras.layers.Dense(2, activation=tf.nn.softmax)
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', "binary_crossentropy"]
                  # loss="categorical_crossentropy",
                  #  metrics=['accuracy', 'categorical_crossentropy']
                  )
    print(model.summary())
    return model










def predict(model, test_images):
    predictions = model.predict(test_images)  # [ [0.3 0.7] , [0.4, 0.6], [0.9, 0.1] ]
    labels = np.argmax(predictions, 1)  # [1, 1, 0]

    return labels


def get_test_data():
    pass


def model_exist():

    if os.path.exists('model1.h5')==1 or os.path.exists('model2.h5')==1:
        return True
    else:
        return False


def load_model1():
    return keras.models.load_model('model1.h5')
def load_model2():
    return keras.models.load_model('model2.h5')


def save_model1():
    model.save("model1.h5")
def save_model2():
    model.save("model2.h5")

if __name__ == "__main__":
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = get_data()
    print(x_train.shape)
    #print(os.path.exists('my_model1.h5'))
    if model_exist():
        print('which mode you want to use? 1 or 2 (other input will be assumed 2)')
        n=input()
        if n=='1':
            model = load_model1()
        else :
            model = load_model2()
    else:
        model = setup_network1()  # train model2 use setup_network2()
        model.fit(x_train, y_train,
                  epochs=20, batch_size=65,
                  validation_data=(x_val, y_val),
                  verbose=2)
        save_model2()
    test_loss, test_acc, _ = model.evaluate(x_test, y_test)
    print(test_loss, test_acc, _)
    # other_images = get_test_data()
    # labels = predict(model, other_images)
    # print(labels)
    # print(class_[i] for i in labels)