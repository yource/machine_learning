'''
主连日K
根据前5天 预测后一天
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

current = 'jd'
# 读取文件数据
df = pd.read_csv(files[current])
# 检查数据错误 最大值最小值 
print(df.describe().transpose())

# cols 2:open 3:high 4:low 5:close 6:volume 7:open_oi 8:close_oi
cols = df.columns
# 整体作图
plot_features = df[[cols[2],cols[5]]]
date = pd.to_datetime(df.pop('datetime'), format='%Y-%m-%d')
plot_features.index = date
dfplt = plot_features.plot()
plt.show()
fig = dfplt.get_figure()
fig.savefig('./pics/mainDaily/'+current+'.png')