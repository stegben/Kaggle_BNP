import sys
import pickle as pkl
from pprint import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression

from utils import write_ans, show_feature_importances

TUNED_PARAMS = [
                {'penalty': ['l2'],
                 'C': [0.001],
                 'fit_intercept': [True],
                 'class_weight': [{1:1, 0:1.5},
                                  {1:1, 0:1}]}
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

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    clf_search = GridSearchCV(
                   LogisticRegression(tol=0.0001 ,
                                      solver='sag',
                                      n_jobs=1,
                                      verbose=1),
                   param_grid=TUNED_PARAMS,
                   scoring='log_loss',
                   n_jobs=20,
                   verbose=5,
                   cv=3,
                   refit=True
                  )
    clf_search.fit(x_train, y_train)
    pprint(clf_search.grid_scores_)
    clf = clf_search.best_estimator_

    weight = clf.coef_[0]
    show_feature_importances(weight, feat_name)

    pred = clf.predict_proba(x_test)[:, 1]
    write_ans(fname=sub_fname,
              header=["ID", "PredictedProb"],
              sample_id=test_id,
              pred=pred)


