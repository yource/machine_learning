'''
PLAN_03
1. 只用 open/high/low/close 数据
2. 归一化 移动平均数 
3. 分析历史样本涨跌幅度 制定对应的标签
------
甲醇_MA VC_v 聚丙烯_pp 豆粕_m 菜粕_RM 塑料_l
锰硅_SM 鸡蛋_jd 乙二醇_eg 螺纹钢_rb 热卷_hc 燃油_fu 玻璃_FG

不去除首条数据 去除首个窗口
增加交易量
各标签数量一致
test结果细致分析
预测结果结合百分数
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import layers
from keras.models import Sequential

code = "rb"; min="15"
modelName = "model/plan02_01_"+code+min+".h5"
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
lookback = 18; delay = 2; step = 1; offset = 3
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
target_percent_low  = np.percentile(target_percent_train, 40)
target_percent_high = np.percentile(target_percent_train, 75)
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
    layers.Dropout(0.3),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.25),
    layers.Dense(3, activation='softmax')
])


# model = Sequential([
#     layers.LSTM(32, input_shape=(None,train_x.shape[-1]) ,return_sequences=True),
#     layers.LSTM(32, input_shape=(None,train_x.shape[-1])),
#     layers.Dense(3, activation='softmax')
# ])
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
# metrics=[keras.metrics.MeanAbsoluteError()]
# callbacks=[early_stopping]
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history = model.fit(train_x,train_y, 
                    epochs=60, batch_size=1024,
                    validation_data=(val_x,val_y))

predictions = model.predict(test_x)
testLen = len(test_y)
test = 0
test_y_num = np.zeros(testLen)
pred_y_num = np.zeros(testLen)
pred = 0
pred_right = 0
pred_wrong = 0
for i,y in enumerate(test_y):
    test_y_num[i] = np.argmax(y)
for i,y in enumerate(predictions):
    pred_y_num[i] = np.argmax(y)
for i in range(0,testLen):
    if(test_y_num[i] != 1):
        test += 1
        # print("test y", test_y[i])
        # print("pred", predictions[i])
    if(pred_y_num[i] != 1):
        pred += 1
        if(test_y_num[i] == pred_y_num[i]):
            pred_right += 1
        else:
            pred_wrong += 1

print(test,pred,pred_right,pred_wrong)


# MA 956 1709 773 936
# rb 944 1664 936 728