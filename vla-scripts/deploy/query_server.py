import requests
import json_numpy
json_numpy.patch()
import numpy as np

import time

t0 = time.time()

N = 20

for i in range(N):

    t1 = time.time()
    action = requests.post(
        "http://0.0.0.0:8000/act",
        json={
            "full_image": np.zeros((480, 640, 3), dtype=np.uint8), 
            "left_wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "right_wrist_image": np.zeros((480, 640, 3), dtype=np.uint8),
            "state": np.zeros((14,), dtype=np.float32),
            "instruction": "do something"}
    ).json()

    action = np.array(action)
    print("time cost", time.time() - t1, "|action.shape", action.shape)

    #time.sleep(0.05)

avg_t = (time.time() - t0) / N

print(N, avg_t)
