from tensorflow import keras
from keras import layers
import keras_tuner as kt
import numpy as np

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
# build_model(kt.HyperParameters())

##### RandomSearch #####
# tuner = kt.RandomSearch(
#     hypermodel=build_model,
#     objective="val_accuracy", #优化指标
#     max_trials=3, #运行的试验总数
#     executions_per_trial=2, #每次试验的执行次数
#     overwrite=True, #覆盖同一目录中以前的结果 还是继续以前的搜索
#     directory="kt_data", #存储搜索结果的目录的路径
#     project_name="helloworld", #子目录名
# )
# tuner.search(x_train, y_train, epochs=2, validation_data=(x_val, y_val))

##### Hyperband #####
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='kt_data',
    project_name='helloworld'
)
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

# 使用从搜索中获得的超参数找到训练模型的最佳周期数
best_hp=tuner.get_best_hyperparameters(num_trials=1)[0]
model = build_model(best_hp)
x_all = np.concatenate((x_train, x_val))
y_all = np.concatenate((y_train, y_val))
history = model.fit(x=x_all, y=y_all, epochs=1, epochs=50, validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# 重新实例化超模型并使用上面的最佳周期数对其进行训练
hypermodel = build_model(best_hp)
hypermodel.fit(x_all, y_all, epochs=best_epoch, validation_split=0.2)
# 保存模型
hypermodel.save("./model/mnist")
# 重新加载模型
newModel = keras.models.load_model("./model/mnist")
print(newModel.summary())
