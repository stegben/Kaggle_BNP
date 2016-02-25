import sys
import pickle as pkl
from pprint import pprint

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD, Adadelta
from keras.layers.advanced_activations import PReLU, ELU

from utils import write_ans, show_feature_importances


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

    model = Sequential()

    model.add(Dense(100, input_dim=Xtr.shape[1], activation='tanh'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(80, activation='linear'))
    model.add(ELU(alpha=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(60, activation='tanh'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(40, activation='linear'))
    model.add(ELU(alpha=0.1))
    model.add(Dropout(0.2))
    model.add(Dense(15, activation='linear'))
    model.add(PReLU())
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    # trainer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    trainer = Adadelta(lr=0.1, tho=0.98, epsilon=1e-7)
    model.compile(loss='binary_crossentropy', optimizer=trainer)

    model.fit(x_train, y_train, nb_epoch=30, batch_size=32, verbose=1, validation_data=(Xval, Yval))

    pred = model.predict_proba(x_test)
    write_ans(fname=sub_fname,
              header=["ID", "PredictedProb"],
              sample_id=test_id,
              pred=pred)


