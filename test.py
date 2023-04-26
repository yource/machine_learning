import numpy as np
import pandas as pd

a = np.zeros((100,6))
for i in range(0,100):
    a[i] = [i+0.1,i+0.2,i+0.3,i+0.4,9+0.5,i+0.6]

print(a)
