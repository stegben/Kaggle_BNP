import csv

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


print("Reading data ...")
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

print("Drop categorical features...")
categorical_columns = df_train.columns[df_train.dtypes == 'object']
df_train = df_train.drop(categorical_columns, axis=1)
df_test = df_test.drop(categorical_columns, axis=1)

print("Clean data...")
df_train = df_train.fillna(-1)
df_test = df_test.fillna(-1)

print("Extract values")
y = df_train["target"].values
x = df_train.drop(["ID", "target"], axis=1).values
test_id = df_test["ID"] # for submission file
x_test = df_test.drop(["ID"], axis=1).values

print("Build model and train...")
clf = RandomForestClassifier()
clf.fit(x, y)

print("Predict...")
pred = clf.predict_proba(x_test)[:, 1]
with open("sub.csv", "w") as fw:
    writer = csv.writer(fw)
    writer.writerow(["ID", "PredictedProb"])
    writer.writerows(zip(test_id, pred))
