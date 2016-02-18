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
clf = RandomForestClassifier(n_estimators=100,
                             max_depth=5,
                             max_features=0.05,
                             verbose=2)
clf.fit(x, y)

print("Predict...")
true_idx = clf.classes_.tolist().index(1)
pred = clf.predict_proba(x_test)[:, true_idx]
with open("sub1.csv", "w") as fw:
    writer = csv.writer(fw)
    writer.writerow(["ID", "PredictedProb"])
    writer.writerows(zip(test_id, pred))
