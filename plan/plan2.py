'''
5分钟 
标准化：简单平均数
拆分数据：一次性拆分、生成
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# 设置浮点数精度
# pd.set_option("display.float_format", "{:.6f}".format) 

# 数据类型
dataDict = "main5min/"

# 文件路径
files = {
    "c" : './data/'+dataDict+'c_main_5min.csv',  # 玉米
    "cs": './data/'+dataDict+'cs_main_5min.csv', # 淀粉
}

contract = 'c'
print("##### 读取文件 选取数据")
df = pd.read_csv(files[contract])
cols = df.columns # cols 2:open 3:high 4:low 5:close 6:volume 7:open_oi 8:close_oi
features = df[[cols[2],cols[3],cols[4],cols[5],cols[6],cols[7],cols[8]]]
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



num_features = df.shape[1]
step = 6
past = 6*8
future = 6*2
learning_rate = 0.001
batch_size = 256
epochs = 10
start = past+future
