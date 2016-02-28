import sys
import pickle as pkl
from pprint import pprint

from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

from utils import write_ans, show_feature_importances

TUNED_PARAMS = [
                {'kernel': ["rbf"],
                 'C': [0.01, 0.1, 1.0, 10],
                 'gamma': ['auto', 0.01, 0.1, 1.0],
                 'probability': [True]}
               ]


if __name__ == "__main__":
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    with open(data_fname, "rb") as fpkl:
        data = pkl.load(fpkl)

    tmpx = data["train"]["x"]
    y = data["train"]["y"]
    x_test = data["test"]["x"]
    test_id = data["test"]["ID"]
    feat_name = data["feature_name"]
    na_count_idx = feat_name.tolist().index("na_count")
    v12idx = feat_name.tolist().index("v12")
    v89idx = feat_name.tolist().index("v89")
    x_train = tmpx[tmpx[:, na_count_idx]<10, ]
    x_train = x_train[:, [v12idx, v89idx]]
    y_train = y[tmpx[:, na_count_idx]<10]
    print(x_train.shape)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)

    clf_search = GridSearchCV(SVC(tol=0.0001, cache_size=512),
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

    true_idx = clf.classes_.tolist().index(1)
    pred = clf.predict_proba(x_test)[:, true_idx]
    write_ans(fname=sub_fname,
              header=["ID", "PredictedProb"],
              sample_id=test_id,
              pred=pred)


