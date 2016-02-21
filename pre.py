import sys

import h5py
import numpy as np
import pandas as pd


# TODO
def data_cleaning(df, ignore_col=None):
    """
    # don't replace NaN in categorical column
    # for numeric, replace NaN with some specific

    ##### maybe, use NMF to fill in data 
    """
    pass


def feature_engineering(df, ignore_col=None):
    """
    # one hot encoding (on unique values < 200)
    # count NaNs
    # binarize v50 (pd.qcut)
    # NaN pattern
    """
    if ignore_col is None:
        proc_df = df
    else:
        proc_df = df.drop(ignore_col, axis=1)
    
    feature_dfs = []
    feature_dfs.append(df[ignore_col])

    # create features
    feature_dfs.append(process_numerical(proc_df))
    feature_dfs.append(one_hot(proc_df, max_cat=200))
    feature_dfs.append(discretize(proc_df, target_col="v50", bins=20))
    feature_dfs.append(count_nan(proc_df, pattern=True))
    return pd.concat(feature_dfs, axis=1)


def process_numerical(df):
    new_df = pd.DataFrame()
    for feat in df:
        if df[feat].dtype == "float" or df[feat].dtype == "int":
            # fill = df[feat].min() - 1
            fill = df[feat].mean()
            # fill = df[feat].max() + 1
            new_df[feat+"_na"] = pd.isnull(df[feat]).astype(int)
            new_df[feat] = df[feat].fillna(fill)
    return new_df


def one_hot(df, max_cat=200):
    new_df = pd.DataFrame()
    for feat in df:
        if df[feat].dtype == "object":
            cat_num = len(df[feat].unique())
            if cat_num < max_cat:
                dummy = pd.get_dummies(df[feat], prefix=feat, dummy_na=True)
                new_df = pd.concat([new_df, dummy], axis=1)
            else:
                print("""we drop column {0}, 
                         since it contains {1} unique values,
                         which is more than {2} """
                         .format(feat, cat_num, max_cat))
    return new_df


def discretize(df, target_col, bins):
    discretized_col = pd.qcut(df[target_col], q=bins)
    return pd.get_dummies(discretized_col, 
                          prefix=target_col, 
                          dummy_na=True)


def count_nan(df, pattern=False):
    new_df = pd.DataFrame()
    new_df["na_count"] = df.isnull().sum(axis=1)
    if pattern:
        na_pattern = df.isnull().apply(
            lambda row: ",".join(df.columns[row].tolist()), 
            axis=1,
            raw=True
        )
        na_pattern_dummy = pd.get_dummies(
            na_pattern, 
            prefix="na_pattern_", 
            dummy_na=True
        )
        new_df = pd.concat([new_df, na_pattern_dummy], axis=1)
    return new_df



def main():
    return

if __name__ == "__main__":
    main()
