import tensorflow as tf
import keras
import keras_tuner as kt
from matplotlib import pyplot as plt

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

img_train = img_train.astype('float32')/255
img_test = img_test.astype('float32')/255

def model_builder(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    hp_units = hp.Int('units',min_value=32,max_value=512,step=32)
    model.add(keras.layers.Dense(units=hp_units,activation='relu'))
    hp_learning_rate = hp.Choice('learning_rate',values=[1e-2, 1e-2, 1e-3])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='kt_data',
                     project_name="intro_to_kt")

stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)

tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
