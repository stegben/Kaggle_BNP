import csv
import gc

import numpy as np
import pandas as pd

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


MAX_CAT = 200
TUNED_PARAMS = [
                {'n_estimators': [100], 
                 'max_depth': [10, None], 
                 'max_features': [0.3, 'sqrt']}
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

gc.collect()


print("One hot encoding...")
for feat in df:
    if feat == "isTest":
        continue
    elif df[feat].dtype == "object":
        if len(df[feat].unique()) < MAX_CAT:
            dummy = pd.get_dummies(df[feat], prefix=feat, dummy_na=True)
            df = pd.concat([df, dummy], axis=1)
        else:
            print("we drop it: {}".format(feat))
        df.drop(feat, axis=1, inplace=True)
    elif df[feat].dtype == "float" or df[feat].dtype == "int":
        fill = df[feat].min() - 1
        # fill = df[feat].mean()
        # fill = df[feat].max() + 1
        df[feat+"_na"] = pd.isnull(df[feat]).astype(int)
        df[feat] = df[feat].fillna(fill)
    else:
        print("Wrong type: {}".format(df[feat].dtype))


print("Extract values")
x = df[~df["isTest"]].drop("isTest", axis=1).values
x_test = df[df["isTest"]].drop("isTest", axis=1).values
del df
gc.collect()


print("Build model and train...")
clf = GridSearchCV(RandomForestClassifier(n_jobs=2),
                   param_grid=TUNED_PARAMS,
                   scoring='log_loss',
                   n_jobs=2, 
                   verbose=5,
                   cv=3
                  )
clf.fit(x, y)

print("Predict...")
true_idx = clf.best_estimator_.classes_.tolist().index(1)
pred = clf.predict_proba(x_test)[:, true_idx]
with open("sub3.csv", "w") as fw:
    writer = csv.writer(fw)
    writer.writerow(["ID", "PredictedProb"])
    writer.writerows(zip(test_id, pred))
