import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# 读取文件
file_c = './data/futuresDaily/C9999.XDCE.csv' # 玉米
file_cs = './data/futuresDaily/CS9999.XDCE.csv' # 淀粉
file_sr = './data/futuresDaily/SR9999.XZCE.csv' # 白糖
# 设置浮点数精度
pd.set_option("display.float_format", "{:.6f}".format) 
# 清理空值
dataframe = pd.read_csv(file_sr,na_values = ["n/a", "na", "--",""])
df = dataframe.dropna()

# 整体作图
# plot_cols = ['open','close']
# plot_features = df[plot_cols]
# date = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
# plot_features.index = date
# print(plot_features.shape)
# print(plot_features.head)
# plot_features.plot()
# plt.show()

# 选取数据
df = df[['open','close','high','low','volume','open_interest','money']]
n = len(df)

# 特征工程 将日期转为一个周期中的某一天/一年中的某一天
# symbols = df.pop('symbol')
# cur_symbol = ""
# cur_symbol_len = 0
# feature_date = np.zeros(n)
# symbol_len_list = []
# for i,sy in enumerate(symbols):
#     if(sy!=cur_symbol): 
#         cur_symbol = sy
#         if(cur_symbol_len>0):
#             symbol_len_list.append(cur_symbol_len)
#         cur_symbol_len = 1
#     else: 
#         cur_symbol_len += 1
# lenlist = np.array(symbol_len_list)

# 拆分数据 训练集、验证集、测试集
train_len = int(n*0.7)
train_df = df[0:train_len]
val_len = int(n*0.9)
val_df = df[train_len:val_len]
test_len = n-train_len-val_len
test_df = df[val_len:]

# 归一化数据 减去平均值、除以标准差
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

print(type(test_df))
print(test_df.head)