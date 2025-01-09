import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

import scipy

fileExt = ["A","B","C","D","E"]
s = 0
N = 1000
for i in range(N):
    resnet_results = []
    trans_results = []
    for i in fileExt:
        df = pd.read_csv(f"results{i}.csv")
        if i=="B":
            row = df[df["epoch"]==10]
        else:
            row = df[df["epoch"]==10]
        row = row[row["split"]=="val"]
        resnet_results.append(row["r2"].iloc[-1])

        data = np.loadtxt(f"val_results{i}.txt")
        trans_results.append(data[5,0])

    resnet_results = np.random.choice(resnet_results,8)
    trans_results = np.random.choice(trans_results,8)

    wil = scipy.stats.wilcoxon(resnet_results,trans_results)
    s += wil.pvalue

print(s/N)

fileExt = ["A","B","C","D","E"]
s = 0
N = 1000
for i in range(N):
    resnet_results = []
    trans_results = []
    for i in fileExt:
        df = pd.read_csv(f"results{i}.csv")
        if i=="B":
            row = df[df["epoch"]==10]
        else:
            row = df[df["epoch"]==10]
        row = row[row["split"]=="val"]
        resnet_results.append(row["mse"].iloc[-1]**0.5)

        data = np.loadtxt(f"val_results{i}.txt")
        trans_results.append(data[5,2]**0.5)

    resnet_results = np.random.choice(resnet_results,8)
    trans_results = np.random.choice(trans_results,8)

    wil = scipy.stats.wilcoxon(resnet_results,trans_results)
    s += wil.pvalue

print(s/N)