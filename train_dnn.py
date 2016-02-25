import sys
import pickle as pkl
from pprint import pprint

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, Adadelta
from keras.layers.advanced_activations import PReLU

from utils import write_ans, show_feature_importances


if __name__ == "__main__":
    data_fname = sys.argv[1]
    sub_fname = sys.argv[2]

    with open(data_fname, "rb") as fpkl:
        data = pkl.load(fpkl)

    x = data["train"]["x"]
    y = data["train"]["y"]
    x_test = data["test"]["x"]
    test_id = data["test"]["ID"]
    feat_name = data["feature_name"]
    print(x.shape)

    scaler = StandardScaler()

    x = scaler.fit_transform(x)
    x_test = scaler.transform(x_test)

    skf = StratifiedKFold(y, n_folds=3, shuffle=False, random_state=1234)
    for tr_idx, val_idx in skf:
        x_train, x_val = x[tr_idx, :], x[val_idx, :]
        y_train, y_val = y[tr_idx], y[val_idx]

        model = Sequential()

        model.add(Dense(1024, input_dim=x_train.shape[1], activation='linear'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='linear'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='linear'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        trainer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # trainer = Adadelta(lr=0.1, tho=0.98, epsilon=1e-7)
        model.compile(loss='binary_crossentropy', optimizer=trainer)

        model.fit(x_train, y_train, nb_epoch=30, batch_size=32, verbose=1, validation_data=(x_val, y_val))

    pred = model.predict_proba(x_test)
    write_ans(fname=sub_fname,
              header=["ID", "PredictedProb"],
              sample_id=test_id,
              pred=pred)


