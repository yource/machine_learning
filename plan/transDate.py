import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

code = "rb"; min="15"
file = "data/main/"+code+"_"+min+"min.csv"
df = pd.read_csv(file)
# date_time = pd.to_datetime(df.pop('datetime'))
# timestamp_s = date_time.map(pd.Timestamp.timestamp)
timestamp_s = df.pop('datetime_nano')/1000000000
day = 24*60*60
year = (365.2425)*day
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
print(df['Year sin'])
plt.plot(np.array(df['Year sin']))
# plt.plot(np.array(df['Year cos']))
plt.xlabel('Time [y]')
plt.title('Time of year')
plt.show()