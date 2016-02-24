import sys
import pickle as pkl
from pprint import pprint

from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

from utils import write_ans, show_feature_importances

TUNED_PARAMS = [
                {'n_estimators': [10000],
                 'learning_rate': [0.05],
                 'loss': ["deviance"],
                 'max_depth': [5, 10],
                 'max_features': [1.0, 0.8],
                 'subsample': [0.7, 1.0]}
               ]


if __name__ == "__main__":
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    with open(data_fname, "rb") as fpkl:
        data = pkl.load(fpkl)

    x_train = data["train"]["x"]
    y_train = data["train"]["y"]
    x_test = data["test"]["x"]
    test_id = data["test"]["ID"]
    feat_name = data["feature_name"]
    print(x_train.shape)

    clf_search = GridSearchCV(GradientBoostingClassifier(verbose=2),
                   param_grid=TUNED_PARAMS,
                   scoring='log_loss',
                   n_jobs=1,
                   verbose=5,
                   cv=3,
                   refit=True
                  )
    clf_search.fit(x_train, y_train)
    pprint(clf_search.grid_scores_)
    clf = clf_search.best_estimator_

    imp = clf.feature_importances_
    show_feature_importances(imp, feat_name)

    true_idx = clf.classes_.tolist().index(1)
    pred = clf.predict_proba(x_test)[:, true_idx]
    write_ans(fname=sub_fname,
              header=["ID", "PredictedProb"],
              sample_id=test_id,
              pred=pred)


