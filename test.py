import pandas as pd
from matplotlib import pyplot as plt

import numpy as np

x = [0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1.0]
y = [1,1,0.67,0.6,0.5,0.42,0.375,0.35,0]
fig, ax = plt.subplots()

plt.plot(x, y)
for i in range(len(x)):
    plt.text(x[i], y[i], f'({x[i]}, {y[i]})')

plt.title("precision-recall-curve")
plt.xlabel('recall')
plt.ylabel('precision')

plt.show()