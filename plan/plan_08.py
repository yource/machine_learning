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
通过历史回测获取历史数据
3年循环训练 验证模型
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
import keras.backend as K
import keras_tuner as kt

project_name="plan_08_001"
code = "rb"; min="15"
file = "data/main/"+code+"_"+min+"min.csv"
df = pd.read_csv(file)
timestamp_s = df['datetime_nano']/1000000000
day = 24*60*60
year = (365.2425)*day
df['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))
cols = df.columns  # cols 2:open 3:high 4:low 5:close 6:volume 7:open_oi 8:close_oi 9:year_sin 10:year_cos
df = df[[cols[2],cols[3],cols[4],cols[5],cols[6]]]
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
    window_target_percent = int(((window_target_value-window_sample_value)/window_sample_value)*100000)
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

model = Sequential([
    layers.Flatten(),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.35),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history = model.fit(train_x,train_y, 
                    epochs=100, batch_size=1024,
                    validation_data=(val_x,val_y))
predictions = model.predict(test_x)
testLen = len(test_y)
test = 0
test_y_num = np.zeros(testLen)
pred_y_num = np.zeros(testLen)
pred = 0
pred_right = 0
pred_wrong = 0
# 0负 1平 2盈
for i,y in enumerate(test_y):
    test_y_num[i] = np.argmax(y)
for i,y in enumerate(predictions):
    if(np.argmax(y)==0 and y[0]>0.4 and y[2]<0.3):
        pred_y_num[i] = 0
    elif(np.argmax(y)==2 and y[2]>0.4 and y[0]<0.3):
        pred_y_num[i] = 2
    else:
        pred_y_num[i] = 1

for i in range(0,testLen):
    if(test_y_num[i] != 1):
        test += 1
    if(pred_y_num[i] != 1):
        pred += 1
        if(test_y_num[i] == pred_y_num[i]):
            pred_right += 1
        else:
            pred_wrong += 1

print(test,pred,pred_right,pred_wrong)

# def build_model(hp):
#     model = Sequential()
#     model.add(layers.Flatten())

#     model.add(keras.layers.Dense(units=hp.Int('units1', min_value=32, max_value=512, step=32)))

#     if hp.Boolean("normal1"): 
#         model.add(layers.BatchNormalization())

#     if hp.Boolean("active1"): 
#         model.add(layers.Activation('relu'))

#     model.add(layers.Dropout(rate=hp.Choice('rate1', values=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])))

#     model.add(keras.layers.Dense(units=hp.Int('units2', min_value=32, max_value=512, step=32)))

#     if hp.Boolean("normal2"): 
#         model.add(layers.BatchNormalization())

#     if hp.Boolean("active2"): 
#         model.add(layers.Activation('relu'))

#     model.add(layers.Dropout(rate=hp.Choice('rate2', values=[0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5])))

#     model.add(layers.Dense(3, activation='softmax'))
    
#     # Float 设置学习率
#     learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
#     # learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#     model.compile(
#         optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
#         loss="categorical_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model

# tuner = kt.Hyperband(
#     build_model,
#     objective='val_accuracy',
#     max_epochs=5000,
#     factor=3,
#     directory='kt_data',
#     project_name=project_name,
# )
# print(tuner.search_space_summary())
# stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# tuner.search(train_x,train_y, epochs=5000, validation_data=(val_x,val_y), callbacks=[stop_early])
# print("########## tuner.results_summary ##########")
# print(tuner.results_summary())

# # 使用从搜索中获得的超参数找到训练模型的最佳周期数
# best_hp=tuner.get_best_hyperparameters(num_trials=1)[0]
# best_model = build_model(best_hp)
# x_all = np.concatenate((train_x, val_x))
# y_all = np.concatenate((train_y, val_y))
# history = best_model.fit(x=x_all, y=y_all, epochs=500, validation_split=0.2)
# print("########## best_model.summary ##########")
# print(best_model.summary())
# val_acc_per_epoch = history.history['val_accuracy']
# best_epoch = val_acc_per_epoch.index(max(val_acc_per_epoch)) + 1
# print('Best epoch: %d' % (best_epoch,))

# # 重新实例化超模型并使用上面的最佳周期数对其进行训练
# hypermodel = build_model(best_hp)
# hypermodel.fit(x_all, y_all, epochs=best_epoch)
# # 保存模型
# hypermodel.save("model/"+project_name+".h5")
# # 重新加载模型
# # newModel = keras.models.load_model("./model/mnist")
# # print(newModel.summary())
