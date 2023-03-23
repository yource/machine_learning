from tensorflow import keras
from keras import layers
import keras_tuner
import numpy as np

def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Flatten())
    # Int 设置layer层数
    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            layers.Dense(
                # Int 设置units数量
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                # Choice 设置激活函数
                activation=hp.Choice("activation", ["relu", "tanh"]),
            )
        )
    # Boolean 设置是否增加dropout层
    if hp.Boolean("dropout"): 
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(10, activation="softmax"))
    # Float 设置学习率
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# 测试build_model是否成功
# build_model(keras_tuner.HyperParameters())

tuner = keras_tuner.RandomSearch( #RandomSearch, BayesianOptimization, Hyperband
    hypermodel=build_model,
    objective="val_accuracy", #优化指标
    max_trials=3, #运行的试验总数
    executions_per_trial=2, #每次试验的执行次数
    overwrite=True, #覆盖同一目录中以前的结果 还是继续以前的搜索
    directory="kt_data", #存储搜索结果的目录的路径
    project_name="helloworld", #子目录名
)
# 打印搜索摘要
tuner.search_space_summary()


(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]
x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

# # 检索最佳模型
# models = tuner.get_best_models(num_models=2)
# best_model = models[0]
# # Build the model.
# # Needed for `Sequential` without specified `input_shape`.
# best_model.build(input_shape=(None, 28, 28))
# best_model.summary()

# 重新训练模型
# Get the top 2 hyperparameters.
best_hps = tuner.get_best_hyperparameters(5)
# Build the model with the best hp.
model = build_model(best_hps[0])
# Fit with the entire dataset.
x_all = np.concatenate((x_train, x_val))
y_all = np.concatenate((y_train, y_val))
model.fit(x=x_all, y=y_all, epochs=1)