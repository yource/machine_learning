'''
主连日K
根据前5天 预测后一天
不包含时间
'''
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# 设置浮点数精度
pd.set_option("display.float_format", "{:.6f}".format) 

# 文件路径
files = {
    "rr": './data/mainDaily/rr_main_daily.csv', # 粳米
    "jd": './data/mainDaily/jd_main_daily.csv', # 鸡蛋
    "c" : './data/mainDaily/c_main_daily.csv',  # 玉米
    "cs": './data/mainDaily/cs_main_daily.csv', # 淀粉
    "SM": './data/mainDaily/SM_main_daily.csv', # 锰硅
    "SF": './data/mainDaily/SF_main_daily.csv', # 硅铁
    "hc": './data/mainDaily/hc_main_daily.csv', # 热卷
    "rb": './data/mainDaily/rb_main_daily.csv' # 螺纹钢
}

contract = 'c'
# 读取文件 选取数据
df = pd.read_csv(files[contract])
cols = df.columns # cols 2:open 3:high 4:low 5:close 6:volume 7:open_oi 8:close_oi
df = df[[cols[2],cols[3],cols[4],cols[5],cols[6],cols[7],cols[8]]]
df = df[1:]
# 检查数据错误 最大值最小值 
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
    fig.savefig('./pics/mainDaily/'+contract+'.png')

# 拆分数据 训练集、验证集、测试集
len_all = len(df)
len_train = int(len_all*0.7)
len_val = int(len_all*0.15)
len_test = len_all - len_train - len_val
df_train = df[0:len_train]
df_val = df[len_train:len_train+len_val]
df_test = df[len_train+len_val:]

# 数据归一化 减去平均值、除以标准差
train_mean = df_train.mean()
train_std = df_train.std()
df_train = (df_train-train_mean)/train_std

print(df_train.head)