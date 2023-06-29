'''
PLAN_01
1. 只用 open/high/low/close 数据
2. 不做归一化 
3. 回归 预测值
------
甲醇_MA VC_v 聚丙烯_pp 豆粕_m 菜粕_RM 塑料_l
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
import util

file = "data/main/MA_15min.csv"
df = pd.read_csv(file)
cols = df.columns
df = df[[cols[2],cols[3],cols[4],cols[5]]]
df = df[1:]
df = np.array(df, dtype=np.float32)
n = len(df)

print("##### 拆分数据 训练集、验证集、测试集")
data_train = df[0: int(0.7 * n)]  #约1911天的数据
data_val = df[int(0.7 * n):int(0.88 * n)]
data_test = df[int(0.88 * n):]

print("##### 生成 样本-目标")
lookback = 6; delay = 2; step = 1; offset = 2
train_x, train_y = util.generator(data_train,lookback=lookback,delay=delay,step=step,offset=offset,normalize=False,targetLabel=False)
val_x, val_y = util.generator(data_val,lookback=lookback,delay=delay,step=step,offset=offset,normalize=False,targetLabel=False)
test_x, test_y = util.generator(data_test,lookback=lookback,delay=delay,step=step,offset=offset,normalize=False,targetLabel=False)
# len1 = len(np.where(train_y<0)[0])
# len2 = len(np.where(train_y>0)[0])
# print(str(len1)+"+"+str(len2)+"="+str(len1+len2))

model = Sequential([
    layers.LSTM(64, input_shape=(None,train_x.shape[-1]) ,return_sequences=True),
    layers.LSTM(64, input_shape=(None,train_x.shape[-1])),
    layers.Dense(1)
])
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
# metrics=[keras.metrics.MeanAbsoluteError()]
# callbacks=[early_stopping]
model.compile(loss='mse',optimizer=keras.optimizers.RMSprop(),metrics=['MAE'])
history = model.fit(train_x,train_y, 
                    epochs=150, batch_size=256,
                    validation_data=(val_x,val_y))