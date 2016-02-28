import sys
import pickle as pkl
from pprint import pprint

import numpy as np
import xgboost as xgb
# from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold

from utils import write_ans, show_feature_importances

XGB_PARAM = {
   "objective": "binary:logistic",
   "booster": "gbtree",
   "eval_metric": "logloss",
   "eta": 0.025, # 0.06,
   # "min_child_weight": 1,
   "subsample": 0.9,
   "colsample_bytree": 0.4,
   "max_depth": 11,
   "nthread": 30,
   "verbose": 0
}

NUM_ROUND = 1000

FOLDS = 5


if __name__ == "__main__":
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    with open(data_fname, "rb") as fpkl:
        data = pkl.load(fpkl)

    x = data["train"]["x"]
    y = data["train"]["y"]
    x_test = data["test"]["x"]
    test_id = data["test"]["ID"]
    # feat_name = data["feature_name"]
    print(x.shape)

    skf = StratifiedKFold(y, n_folds=FOLDS, random_state=1234)
    models = []
    error = []
    for tr_idx, val_idx in skf:
        x_train, x_val = x[tr_idx, :], x[val_idx, :]
        y_train, y_val = y[tr_idx], y[val_idx]

        xgtrain = xgb.DMatrix(x_train, y_train)
        xgval = xgb.DMatrix(x_val, y_val)

        watchlist = [(xgtrain, 'train'), (xgval, 'eval')]
        model = xgb.train(XGB_PARAM,
                          xgtrain,
                          NUM_ROUND,
                          watchlist,
                          early_stopping_rounds=20)
        models.append(model)
        error.append(model.best_score)
    print(error)
    print(sum(error)/FOLDS)
    xgtest = xgb.DMatrix(x_test)
    # true_idx = clf.classes_.tolist().index(1)
    pred = np.zeros(x_test.shape[0])
    for model in models:
        pred += model.predict(xgtest)
    pred = pred/FOLDS
    write_ans(fname=sub_fname,
              header=["ID", "PredictedProb"],
              sample_id=test_id,
              pred=pred)


