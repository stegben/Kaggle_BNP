import csv
import gc
from pprint import pprint

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from utils import write_ans
from pre import feature_engineering


MAX_CAT = 200
TUNED_PARAMS = [
                {'n_estimators': [500], 
                 'max_depth': [None], 
                 'max_features': ['sqrt']}
               ]


print("Reading data ...")
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

y = df_train["target"].values
test_id = df_test["ID"] # for submission file

df_train = df_train.drop(["ID", "target"], axis=1)
df_test = df_test.drop(["ID"], axis=1)

df_train["isTest"] = False
df_test["isTest"] = True
df = pd.concat([df_train, df_test], axis=0)

del df_train
del df_test

df = feature_engineering(df, ignore_col="isTest")
gc.collect()

print("Extract values")
x = df[~df["isTest"]].drop("isTest", axis=1).values
x_test = df[df["isTest"]].drop("isTest", axis=1).values
col_name = df.columns
del df

print("Build model and train...")
clf = GridSearchCV(RandomForestClassifier(n_jobs=-1),
                   param_grid=TUNED_PARAMS,
                   scoring='log_loss',
                   n_jobs=1, 
                   verbose=5,
                   cv=3,
                   refit=True
                  )
clf.fit(x, y)

pprint(clf.grid_scores_)

imp = clf.best_estimator_.feature_importances_
plt.scatter(np.arange(len(imp)), imp)
plt.savefig("imp.png")

for ind in np.argsort(imp)[-50:-1]:
    print("{0}: {1}".format(col_name[ind], imp[ind]))

print("Predict...")
true_idx = clf.best_estimator_.classes_.tolist().index(1)
pred = clf.predict_proba(x_test)[:, true_idx]
write_ans(fname="sub3.csv", 
          header=["ID", "PredictedProb"], 
          sample_id=test_id,
          pred=pred)
"""
with open("sub3.csv", "w") as fw:
    writer = csv.writer(fw)
    writer.writerow(["ID", "PredictedProb"])
    writer.writerows(zip(test_id, pred))
"""