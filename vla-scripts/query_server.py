import requests
import json_numpy
json_numpy.patch()
import numpy as np

import time

t0 = time.time()

N = 200

for i in range(N):

    action = requests.post(
        "http://0.0.0.0:8000/act",
        json={"image": np.zeros((480, 640, 3), dtype=np.uint8), "instruction": "do something"}
    ).json()

    #time.sleep(0.05)

avg_t = (time.time() - t0) / N

print(N, avg_t)
print(action)
