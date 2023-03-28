import pandas as pd

# 读取日K线
file_c = './data/futuresDaily/C9999.XDCE.csv' # 玉米
file_cs = './data/futuresDaily/CS9999.XDCE.csv' # 淀粉

df = pd.read_csv(file_c)
# target = df.pop('target')
date = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
plot_cols = ['open','close']
plot_features = df[plot_cols]
plot_features.index = date
_ = plot_features.plot(subplots=True)
plot_features = df[plot_cols][:480]
plot_features.index = date[:480]