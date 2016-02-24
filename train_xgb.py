import sys
import pickle as pkl
from pprint import pprint

import xgboost as xgb
from sklearn.cross_validation import StratifiedShuffleSplit

from utils import write_ans, show_feature_importances

XGB_PARAM = {
   "objective": "binary:logistic",
   "booster": "gbtree",
   "n_estimators": 1000,
   "eval_metric": "logloss",
   "eta": 0.01, # 0.06,
   # "min_child_weight": 20,
   "subsample": 0.75,
   "colsample_bytree": 0.8,
   "max_depth": 7,
   "nthread": 3
}

NUM_ROUND = 1000


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

    sss = StratifiedShuffleSplit(y, n_iter=3, test_size=0.1)
    for tr_idx, val_idx in sss:
        x_train, x_val = x[tr_idx, :], x[val_idx, :]
        y_train, y_val = y[tr_idx], y[val_idx]

        xgtrain = xgb.DMatrix(x_train, y_train)
        xgval = xgb.DMatrix(x_val, y_val)

        watchlist = [(xgval, 'eval'), (xgtrain, 'train')]
        model = xgb.train(XGB_PARAM,
                          xgtrain,
                          NUM_ROUND,
                          watchlist,
                          early_stopping_rounds=10)
    xgtest = xgb.DMatrix(x_test)
    # true_idx = clf.classes_.tolist().index(1)
    pred = model.predict(xgtest)
    write_ans(fname=sub_fname,
              header=["ID", "PredictedProb"],
              sample_id=test_id,
              pred=pred)


