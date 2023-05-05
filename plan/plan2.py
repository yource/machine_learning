'''
5分钟 
简单平均数,
无时间,
少量指标,
一次性拆分、生成
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import keras
from keras import layers
from keras.models import Sequential
# 设置浮点数精度
# pd.set_option("display.float_format", "{:.6f}".format) 

# 数据类型
dataDict = "main5min/"

# 文件路径
files = {
    "c" : './data/'+dataDict+'c_main_5min.csv',  # 玉米
    "cs": './data/'+dataDict+'cs_main_5min.csv', # 淀粉
    "i": './data/'+dataDict+'i_main_5min.csv', # 淀粉
    "j": './data/'+dataDict+'j_main_5min.csv', # 淀粉
}

contract = 'i'
print("##### 读取文件 选取数据")
df = pd.read_csv(files[contract])
cols = df.columns # cols 2:open 3:high 4:low 5:close 6:volume 7:open_oi 8:close_oi
features = df[[cols[2],cols[3],cols[4],cols[5]]]
features.index = df["datetime"]
features = features[1:]
df = features
print("##### 检查数据错误 最大值最小值") 
print(df.head)
print(df.describe().transpose())

# 整体作图
def makeFig():
    plot_features = df[[cols[2],cols[5]]]
    plot_date = pd.to_datetime(df.pop('datetime'), format='%Y-%m-%d')
    plot_features.index = plot_date
    plot = plot_features.plot()
    plt.show()
    fig = plot.get_figure()
    fig.savefig('./pics/'+dataDict+contract+'.png')
# makeFig()

print("##### 数据归一化 减去平均值、除以标准差")
n = int(df.shape[0])
df_train = df[:int(0.75 * n)]
train_mean = df_train.mean(axis=0)
train_std = df_train.std(axis=0)
df = (df-train_mean)/train_std
print("##### 拆分数据 训练集、验证集、测试集")
all_data = np.array(df, dtype=np.float32)
print(all_data)
train_data = all_data[0 : int(0.75 * n) - 1]
val_data = all_data[int(0.75 * n):int(0.9 * n) - 1]
test_data = all_data[int(0.9 * n):]
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)
'''
根据过去的lookback期数据 预测delay期之后的数据
隔step取一个有效值
隔offset个数据生成一个窗口 0代表各window不交叉
'''
def generator(data,lookback,delay,step=6,offset=0):
    data_len = len(data)
    window_size = lookback+delay  # 每个窗口用到的数据量
    if offset==0:
        offset = window_size
    window_count= (data_len-window_size)//offset # 可以切分出多少个窗口
    if(step==0):
        step=1
    samples = np.zeros((window_count, lookback//step, data.shape[-1]))
    targets = np.zeros((window_count,))
    for i in range(0,window_count):
        window = data[offset*i:offset*i+window_size]
        samples[i] = window[0:lookback:step]
        targets[i] = window[lookback+delay-1][0]
    return samples,targets
print("##### 生成窗口数据")
lookback = 24
delay=6
step=1
offset=2
train_x,train_y = generator(train_data,lookback,delay,step,offset)
val_x,val_y = generator(val_data,lookback,delay,step,offset)
print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)

model = Sequential([
    # layers.LSTM(32, input_shape=(None,train_x.shape[-1]) ,return_sequences=True),
    layers.LSTM(32, input_shape=(None,train_x.shape[-1])),
    layers.Dense(1)
])
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=2,mode='min')
# metrics=[keras.metrics.MeanAbsoluteError()]
# callbacks=[early_stopping]
model.compile(loss='mae',optimizer=keras.optimizers.RMSprop())
history = model.fit(train_x,train_y, 
                    epochs=20, batch_size=128,
                    validation_data=(val_x,val_y))