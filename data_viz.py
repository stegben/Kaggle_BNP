import os
import csv
import gc
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Reading data ...")
df = pd.read_csv("train.csv")

num_col = []
for feat in df:
    if df[feat].dtype == "float":
        num_col.append(feat)

os.chdir("image/")
for idx in range(len(num_col)):
    col1 = num_col[idx]
    for idx2 in range(idx+1, len(num_col)):
        col2 = num_col[idx2]
        fname = col1 + "_" + col2 + ".png"
        print(fname)

        true_x = df[df["target"]==1][col1].fillna(-1.1)
        true_y = df[df["target"]==1][col2].fillna(-0.9)

        false_x = df[df["target"]==0][col1].fillna(-0.9)
        false_y = df[df["target"]==0][col2].fillna(-1.1)

        plt.scatter(true_x, true_y, c="b", alpha=0.1)
        plt.scatter(false_x, false_y, c="r", alpha=0.05)

        plt.savefig(fname)
        plt.close()
