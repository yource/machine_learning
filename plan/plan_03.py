'''
PLAN_03
1. 只用 open/high/low/close 数据
2. 不做归一化 
3. 分析历史样本涨跌幅度 制定对应的标签
------
甲醇_MA VC_v 聚丙烯_pp 豆粕_m 菜粕_RM 塑料_l
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import layers
from keras.models import Sequential

code = "MA"; min="15"
modelName = "model/plan02_01_"+code+min+".h5"
file = "data/main/"+code+"_"+min+"min.csv"
df = pd.read_csv(file)
cols = df.columns
df = df[[cols[2],cols[3],cols[4],cols[5]]]
df = df[1:]
df = np.array(df, dtype=np.float32)
percent_train = 0.76
percent_val = 0.88

'''
根据过去的lookback期数据 预测之后的第delay期的数据
隔step取一个有效值
隔offset个数据生成一个窗口 0代表各window不交叉
'''
print("##### 生成 样本-目标")
lookback = 6; delay = 2; step = 1; offset = 3
data_len = len(df)
window_size = lookback+delay  # 每个窗口用到的数据量
if offset == 0:
    offset = window_size
window_count = (data_len-window_size)//offset  # 可以切分出多少个窗口
if(step == 0):
    step = 1
samples = np.zeros((window_count, lookback//step, df.shape[-1]))
target_percents = np.zeros(window_count)
for i in range(0, window_count):
    window = df[offset*i:offset*i+window_size]
    window_samples = window[0:lookback]
    window_sample_value = window[lookback-1][3]
    window_target = window[lookback+delay-1]
    window_target_high = (window_target[0]+window_target[1]+window_target[3])/3
    window_target_low = (window_target[0]+window_target[2]+window_target[3])/3
    window_target_value = (((window_target[0]+window_target[1]+window_target[2])/3)+window_target[3]*2)/3
    window_target_percent = int(((window_target_value-window_sample_value)/window_sample_value)*10000)
    samples[i] = window[0:lookback:step]
    target_percents[i] = window_target_percent

target_percent_low  = np.percentile(target_percents, 25)
target_percent_high = np.percentile(target_percents, 75)
target_labels = np.zeros((len(samples),3))

for i,y in enumerate(target_percents):
    if(y<=target_percent_low):
        target_labels[i] = [1,0,0]
    elif(y>=target_percent_high):
        target_labels[i] = [0,0,1]
    else:
        target_labels[i] = [0,1,0]

all_x = samples
all_y = target_labels
print("##### 拆分数据 训练集、验证集、测试集")
n = len(all_x)
n_train = int(percent_train * n)
n_val = int(percent_val * n)
train_x = all_x[0:n_train]  #约1911天的数据
train_y = all_y[0:n_train]
val_x   = all_x[n_train:n_val]
val_y   = all_y[n_train:n_val]
test_x  = all_x[n_val:] 
test_y  = all_y[n_val:] 
print("train_x",train_x.shape)
print("val_x",val_x.shape)
print("test_x",test_x.shape)

model = Sequential([
    layers.LSTM(32, input_shape=(None,train_x.shape[-1]) ,return_sequences=True),
    layers.LSTM(32, input_shape=(None,train_x.shape[-1])),
    layers.Dense(3, activation='softmax')
])
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
# metrics=[keras.metrics.MeanAbsoluteError()]
# callbacks=[early_stopping]
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history = model.fit(train_x,train_y, 
                    epochs=50, batch_size=128,
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
        print("test y", test_y[i])
        print("pred", predictions[i])
    if(pred_y_num[i] != 1):
        pred += 1
        if(test_y_num[i] == pred_y_num[i]):
            pred_right += 1
        else:
            pred_wrong += 1

print(test,pred,pred_right,pred_wrong)