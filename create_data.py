"""
Create data for model, store in pkl

X, Y, Xtest are np array

{"train": (X, Y),
 "test": (Xtest, ID)
 "feature_name": }
"""
import sys
import pickle as pkl

import pandas as pd

from pre import feature_engineering


if __name__ == "__main__":
    output_fname = sys.argv[1]

    print("Read data...")
    df_train = pd.read_csv("train.csv", index_col=["ID"])
    df_test = pd.read_csv("test.csv", index_col=["ID"])

    y = df_train["target"].values
    test_id = df_test.index # for submission file

    df_train = df_train.drop("target", axis=1)

    df_train["isTest"] = False
    df_test["isTest"] = True
    df = pd.concat([df_train, df_test], axis=0)

    df = feature_engineering(df, ignore_col="isTest")

    print("Extract values")
    x = df[~df["isTest"]].drop("isTest", axis=1).values
    x_test = df[df["isTest"]].drop("isTest", axis=1).values

    feature_name = df.drop("isTest", axis=1).columns

    print(test_id)
    print(feature_name)

    with open(output_fname, "wb") as fpkl:
        output = {}
        output["train"] = {"x": x, "y": y}
        output["test"] = {"x": x_test, "ID": test_id}
        output["feature_name"] = feature_name
        pkl.dump(output, fpkl)
