# import numpy as np

# a = np.zeros((2,3,4))
# print(a)
# print(a.shape)
# print(a.shape[-1])

import os
from tensorflow.python.client import device_lib
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "99"
 
if __name__ == "__main__":
    print(device_lib.list_local_devices())
