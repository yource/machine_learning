'''
PLAN_03
1. 只用 open/high/low/close 数据
2. 归一化 移动平均数 
3. 分析历史样本涨跌幅度 制定对应的标签
------
甲醇_MA VC_v 聚丙烯_pp 豆粕_m 菜粕_RM 塑料_l
锰硅_SM 鸡蛋_jd 乙二醇_eg 螺纹钢_rb 热卷_hc 燃油_fu 玻璃_FG

不去除首条数据 去除首个窗口 实战时倒退删除首个窗口
增加交易量数据 增加日期周期数据
平滑数据 rolling_mean
各标签对应的样本数量一致
test结果细致分析
预测结果结合百分数决定是否下手
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
import keras.backend as K
import keras_tuner as kt

project_name="plan_07_001"
code = "rb"; min="15"
file = "data/main/"+code+"_"+min+"min.csv"
df = pd.read_csv(file)
cols = df.columns
df = df[[cols[2],cols[3],cols[4],cols[5]]]
df = df[1:]
data = np.array(df, dtype=np.float32)
percent_train = 0.76
percent_val = 0.88
len_data = len(data)
len_train = int(len_data*percent_train)

'''
根据过去的lookback期数据 预测之后的第delay期的数据
隔step取一个有效值
隔offset个数据生成一个窗口 0代表各window不交叉
'''
print("##### 生成 样本-目标")
lookback = 12; delay = 2; step = 1; offset = 3
window_size = lookback+delay  # 每个窗口用到的数据量
if offset == 0:
    offset = window_size
window_count = (len_data-window_size)//offset  # 可以切分出多少个窗口
if(step == 0):
    step = 1
samples = np.zeros((window_count, lookback//step, data.shape[-1]))
target_percents = np.zeros(window_count)
for i in range(0, window_count):
    window = data[offset*i:offset*i+window_size]
    window_samples = window[0:lookback]
    window_mean = window_samples.mean(axis=0)
    window_samples -= window_mean
    window_std = window_samples.std(axis=0)
    window_samples /= window_std
    window_sample_value = window[lookback-1][3]
    window_target = window[lookback+delay-1]
    window_target_high = (window_target[0]+window_target[1]+window_target[3])/3
    window_target_low = (window_target[0]+window_target[2]+window_target[3])/3
    window_target_value = (((window_target[0]+window_target[1]+window_target[2])/3)+window_target[3]*2)/3
    window_target_percent = int(((window_target_value-window_sample_value)/window_sample_value)*10000)
    samples[i] = window_samples
    target_percents[i] = window_target_percent
    
n = window_count
n_train = int(percent_train * n)
n_val = int(percent_val * n)

target_percent_train = target_percents[:n_train]
target_percent_low  = np.percentile(target_percent_train, 33)
target_percent_high = np.percentile(target_percent_train, 67)
print("target_percent_low",target_percent_low)
print("target_percent_high",target_percent_high)
target_labels = np.zeros((window_count,3))

for i,y in enumerate(target_percents):
    if(y<=target_percent_low):
        target_labels[i] = [1,0,0]
    elif(y>=target_percent_high):
        target_labels[i] = [0,0,1]
    else:
        target_labels[i] = [0,1,0]

print("##### 拆分数据 训练集、验证集、测试集")
all_x = samples #约1911天的数据
all_y = target_labels
train_x = all_x[0:n_train] 
train_y = all_y[0:n_train]
val_x   = all_x[n_train:n_val]
val_y   = all_y[n_train:n_val]
test_x  = all_x[n_val:] 
test_y  = all_y[n_val:] 
print("train_x",train_x.shape)
print("val_x",val_x.shape)
print("test_x",test_x.shape)


def build_model(hp):
    model = Sequential()
    model.add(layers.Flatten())

    hp_units1 = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units1))

    if hp.Boolean("normal1"): 
        model.add(layers.BatchNormalization())

    if hp.Boolean("active1"): 
        model.add(layers.Activation('relu'))

    if hp.Boolean("dropout1"): 
        model.add(layers.Dropout(rate=hp.Choice('rate1', values=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])))

    hp_units2 = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units2))

    if hp.Boolean("normal2"): 
        model.add(layers.BatchNormalization())

    if hp.Boolean("active2"): 
        model.add(layers.Activation('relu'))

    if hp.Boolean("dropout2"): 
        model.add(layers.Dropout(rate=hp.Choice('rate2', values=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])))

    model.add(layers.Dense(3, activation='softmax'))
    
    # Float 设置学习率
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# model = Sequential([
#     layers.Flatten(),
#     layers.Dense(128),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dropout(0.25),
#     layers.Dense(64),
#     layers.BatchNormalization(),
#     layers.Activation('relu'),
#     layers.Dropout(0.25),
#     layers.Dense(3, activation='softmax')
# ])
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=5000,
    factor=3,
    directory='kt_data',
    project_name=project_name,
)
print(tuner.search_space_summary())
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(train_x,train_y, epochs=5000, validation_data=(val_x,val_y), callbacks=[stop_early])
print("########## tuner.results_summary ##########")
print(tuner.results_summary())

# 使用从搜索中获得的超参数找到训练模型的最佳周期数
best_hp=tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = build_model(best_hp)
best_model.build()
print("########## best_model.summary ##########")
print(best_model.summary())
x_all = np.concatenate((train_x, val_x))
y_all = np.concatenate((train_y, val_y))
history = best_model.fit(x=x_all, y=y_all, epochs=500, validation_split=0.2)
val_acc_per_epoch = history.history['val_accuracy']
best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
print('Best epoch: %d' % (best_epoch,))

# 重新实例化超模型并使用上面的最佳周期数对其进行训练
hypermodel = build_model(best_hp)
hypermodel.fit(x_all, y_all, epochs=best_epoch)
# 保存模型
hypermodel.save("model/"+project_name)
# 重新加载模型
# newModel = keras.models.load_model("./model/mnist")
# print(newModel.summary())

########## tuner.results_summary ##########
'''
Results summary
Results in kt_data/plan_07_001
Showing 10 best trials
<keras_tuner.engine.objective.Objective object at 0x7fc569b6a220>
Trial summary
Hyperparameters:
units: 256
normal1: True
active1: False
dropout1: False
normal2: False
active2: False
dropout2: False
lr: 0.00034929128241395816
rate1: 0.3
rate2: 0.4
tuner/epochs: 13
tuner/initial_epoch: 5
tuner/bracket: 6
tuner/round: 2
tuner/trial_id: 2992
Score: 0.998196005821228
--------------------------------
Trial summary
Hyperparameters:
units: 416
normal1: True
active1: False
dropout1: True
normal2: False
active2: False
dropout2: False
lr: 0.0002693387023501433
rate1: 0.45
rate2: 0.4
tuner/epochs: 38
tuner/initial_epoch: 13
tuner/bracket: 6
tuner/round: 3
tuner/trial_id: 1145
Score: 0.9957907199859619
--------------------------------
Trial summary
Hyperparameters:
units: 448
normal1: True
active1: False
dropout1: False
normal2: False
active2: True
dropout2: False
lr: 0.00011940882987867163
rate1: 0.45
rate2: 0.15
tuner/epochs: 38
tuner/initial_epoch: 13
tuner/bracket: 5
tuner/round: 2
tuner/trial_id: 1590
Score: 0.9951894283294678
--------------------------------
Trial summary
Hyperparameters:
units: 224
normal1: True
active1: False
dropout1: False
normal2: False
active2: False
dropout2: False
lr: 0.0010145077995017421
rate1: 0.4
rate2: 0.25
tuner/epochs: 38
tuner/initial_epoch: 13
tuner/bracket: 6
tuner/round: 3
tuner/trial_id: 1134
Score: 0.9933854341506958
--------------------------------
Trial summary
Hyperparameters:
units: 256
normal1: True
active1: False
dropout1: False
normal2: False
active2: False
dropout2: False
lr: 0.0012916500815288833
rate1: 0.25
rate2: 0.5
tuner/epochs: 5
tuner/initial_epoch: 0
tuner/bracket: 5
tuner/round: 0
Score: 0.9933854341506958
--------------------------------
Trial summary
Hyperparameters:
units: 288
normal1: True
active1: False
dropout1: False
normal2: True
active2: False
dropout2: True
lr: 0.001382827869491224
rate1: 0.5
rate2: 0.35
tuner/epochs: 5
tuner/initial_epoch: 0
tuner/bracket: 5
tuner/round: 0
Score: 0.9927841424942017
--------------------------------
Trial summary
Hyperparameters:
units: 352
normal1: True
active1: False
dropout1: False
normal2: False
active2: False
dropout2: False
lr: 0.00018723379392602342
rate1: 0.15
rate2: 0.15
tuner/epochs: 13
tuner/initial_epoch: 5
tuner/bracket: 5
tuner/round: 1
tuner/trial_id: 1376
Score: 0.9927841424942017
--------------------------------
Trial summary
Hyperparameters:
units: 448
normal1: True
active1: False
dropout1: True
normal2: True
active2: False
dropout2: False
lr: 0.005322635881784047
rate1: 0.1
rate2: 0.1
tuner/epochs: 38
tuner/initial_epoch: 0
tuner/bracket: 3
tuner/round: 0
Score: 0.9921827912330627
--------------------------------
Trial summary
Hyperparameters:
units: 96
normal1: True
active1: False
dropout1: True
normal2: True
active2: False
dropout2: False
lr: 0.008845571743601299
rate1: 0.15
rate2: 0.3
tuner/epochs: 5
tuner/initial_epoch: 0
tuner/bracket: 5
tuner/round: 0
Score: 0.9915814995765686
--------------------------------
Trial summary
Hyperparameters:
units: 160
normal1: True
active1: False
dropout1: False
normal2: False
active2: True
dropout2: True
lr: 0.0007315903709982806
rate1: 0.25
rate2: 0.1
tuner/epochs: 13
tuner/initial_epoch: 5
tuner/bracket: 5
tuner/round: 1
tuner/trial_id: 5596
Score: 0.9915814995765686
'''