import tensorflow as tf
import keras
import keras_tuner as kt
(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
print(len(img_train))