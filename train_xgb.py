import sys
import pickle as pkl
from pprint import pprint

import xgboost as xgb
# from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.cross_validation import StratifiedKFold

from utils import write_ans, show_feature_importances

XGB_PARAM = {
   "objective": "binary:logistic",
   "booster": "gbtree",
   "n_estimators": 1000,
   "eval_metric": "logloss",
   "eta": 0.025, # 0.06,
   # "min_child_weight": 20,
   "subsample": 0.5,
   "colsample_bytree": 0.7,
   "max_depth": 8,
   "nthread": 24
}

NUM_ROUND = 1000

FOLDS = 3


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

    skf = StratifiedKFold(y, n_folds=FOLDS)
    models = []
    for tr_idx, val_idx in skf:
        x_train, x_val = x[tr_idx, :], x[val_idx, :]
        y_train, y_val = y[tr_idx], y[val_idx]

        xgtrain = xgb.DMatrix(x_train, y_train)
        xgval = xgb.DMatrix(x_val, y_val)

        watchlist = [(xgval, 'eval'), (xgtrain, 'train')]
        model = xgb.train(XGB_PARAM,
                          xgtrain,
                          NUM_ROUND,
                          watchlist,
                          early_stopping_rounds=15)
        models.append(model)

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


