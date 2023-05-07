import numpy as np
import pandas as pd
a = np.array([0.1,1.1,2.1,3.1,4.1])
for i in  range(0,4):
    if a[i]==0.1:
        a[i] = 1
print(a)
