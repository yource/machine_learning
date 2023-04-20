'''
5分钟 
标准化：简单平均数
拆分数据：一次性拆分、生成
'''
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# 设置浮点数精度
pd.set_option("display.float_format", "{:.6f}".format) 

# 数据类型
dataDict = "main5min/"

# 文件路径
files = {
    "c" : './data/'+dataDict+'c_main_5min.csv',  # 玉米
    "cs": './data/'+dataDict+'cs_main_5min.csv', # 淀粉
}

contract = 'c'
# 读取文件 选取数据
df = pd.read_csv(files[contract])
cols = df.columns # cols 2:open 3:high 4:low 5:close 6:volume 7:open_oi 8:close_oi
df = df[[cols[0],cols[2],cols[3],cols[4],cols[5],cols[6],cols[7],cols[8]]]
df = df[1:]
# 检查数据错误 最大值最小值 
# print(df.head)
# print(df.describe().transpose())

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

# 拆分数据 训练集、验证集、测试集
n = len(df)
df_train = df[0:int(n*0.7)]
df_val = df[int(n*0.7):int(n*0.9)]
df_test = df[int(n*0.9):]
num_features = df.shape[1]
print(n,len(df_train),len(df_val),len(df_test))
print(num_features)

# 数据归一化 减去平均值、除以标准差
# train_mean = df_train.mean()
# train_std = df_train.std()
# df_train = (df_train-train_mean)/train_std
