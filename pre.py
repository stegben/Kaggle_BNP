import sys
from time import time

# import h5py
import numpy as np
import pandas as pd

from sklearn.decomposition import KernelPCA


CAT_EXCLUDE_COL = ["v107"]
NUM_EXCLUDE_COL = ["v69", "v76", "v64", "v60"]

# TODO
def data_cleaning(df):
    """
    # don't replace NaN in categorical column
    # for numeric, replace NaN with some specific

    ##### maybe, use NMF to fill in data
    """
    new_df = df
    for feat in df:
        if df[feat].isnull().sum() < 100:
            target_sr = df[feat]
            new_df[feat] = target_sr.fillna(target_sr.value_counts().index[0])
    return new_df


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
    # proc_df = data_cleaning(proc_df)
    feature_dfs = []
    feature_dfs.append(df[ignore_col])

    # create features
    feature_dfs.append(process_numerical(proc_df))
    feature_dfs.append(one_hot(proc_df, max_cat=200))
    feature_dfs.append(discretize(proc_df, target_col="v10", bins=20))
    feature_dfs.append(discretize(proc_df, target_col="v82", bins=20))
    feature_dfs.append(discretize(proc_df, target_col="v46", bins=20))
    feature_dfs.append(discretize(proc_df, target_col="v89", bins=20))
    feature_dfs.append(discretize(proc_df, target_col="v105", bins=20))
    feature_dfs.append(discretize(proc_df, target_col="v124", bins=20))
    feature_dfs.append(discretize(proc_df, target_col="v25", bins=20))
    feature_dfs.append(discretize(proc_df, target_col="v16", bins=20))
    feature_dfs.append(discretize(proc_df, target_col="v63", bins=20))
    feature_dfs.append(discretize(proc_df, target_col="v69", bins=20))
    feature_dfs.append(count_nan(proc_df, pattern=True))
    # feature_dfs.append(divide_numerical(proc_df))
    feature_dfs.append(az2int(proc_df, target_col="v22"))
    #feature_dfs.append(multiply_col(proc_df, col1="v50", col2="v21"))
    #feature_dfs.append(multiply_col(proc_df, col1="v50", col2="v10"))
    #feature_dfs.append(multiply_col(proc_df, col1="v50", col2="v19"))
    #feature_dfs.append(multiply_col(proc_df, col1="v50", col2="v14"))
    #feature_dfs.append(multiply_col(proc_df, col1="v50", col2="v40"))
    #feature_dfs.append(multiply_col(proc_df, col1="v50", col2="v12"))
    feature_dfs.append(special_50(proc_df, "v21"))
    new_df = pd.concat(feature_dfs, axis=1)

    print("PCA...")
    t1 = time()
    pca = KernelPCA(n_components=10, kernel="rbf", eigen_solver='arpack')
    pca_x = pca.fit_transform(new_df.values)
    pca_df = pd.DataFrame(data=pca_x,
                          columns=["pca_"+str(i) for i in range(pca_x.shape[1])])
    t2 = time()
    print(t2-t1)

    new_df = pd.concat([new_df, pca_df], axis=1)
    return new_df


def process_numerical(df):
    df = df.drop(NUM_EXCLUDE_COL, axis=1)
    new_df = pd.DataFrame()
    for feat in df:
        if df[feat].dtype == "float" or df[feat].dtype == "int":
            # fill = df[feat].min() - 1
            fill = df[feat].mean()
            # fill = df[feat].max() + 1
            new_df[feat+"_missing"] = pd.isnull(df[feat]).astype(int)
            new_df[feat] = df[feat].fillna(fill)
    return new_df


def divide_numerical(df):
    num_col = []
    for feat in df:
        if df[feat].dtype == "float":
            num_col.append(feat)

    new_col = []
    for idx in range(len(num_col)):
        col1 = num_col[idx]
        for idx2 in range(idx+1, len(num_col)):
            col2 = num_col[idx2]
            # new_col.append(col1+"_d_"+col2)
            new_col.append(col1+"_m_"+col2)
            # new_col.append(col1+"_-_"+col2)

    new_df = pd.DataFrame(data=np.ones((len(df.index), len(new_col))), index=df.index, columns=new_col, dtype=np.float64)
    for idx in range(len(num_col)):
        col1 = num_col[idx]
        for idx2 in range(idx+1, len(num_col)):
            col2 = num_col[idx2]
            fill1 = -1
            fill2 = -1

            sr1 = df[col1].fillna(fill1)
            sr2 = df[col2].fillna(fill2)

            # new_df[col1+"_d_"+col2] = sr1 / sr2
            new_df[col1+"_m_"+col2] = sr1 * sr2
            # new_df[col1+"_-_"+col2] = sr1 - sr2

    return new_df


def one_hot(df, max_cat=200):
    new_df = pd.DataFrame()
    df = df.drop(CAT_EXCLUDE_COL, axis=1)

    def az_to_int(az):
        if az==az:  #catch NaN
            hv = 0
            for i in range(len(az)):
                hv += (ord(az[i].lower())-ord('a')+1)*26**(len(az)-1-i)
            return hv
        else:
            return -1

    for feat in df:
        cat_num = len(df[feat].unique())
        if cat_num < max_cat:
            dummy = pd.get_dummies(df[feat], prefix=feat, dummy_na=True)
            new_df = pd.concat([new_df, dummy], axis=1)
            if df[feat].dtype == "object":
                new_df[feat+"_fact"], _ = pd.factorize(df[feat], na_sentinel=0)
                new_df[feat+"_2int"] = df[feat].apply(az_to_int)
        else:
            if df[feat].dtype == "object":
                print("""we drop column {0},
                     since it contains {1} unique values,
                     which is more than {2} """
                     .format(feat, cat_num, max_cat))
    return new_df


def discretize(df, target_col, bins):
    discretized_col = pd.qcut(df[target_col], bins)
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


def az2int(df, target_col="v22"):
    new_df = pd.DataFrame()
    def az_to_int(az):
        if az==az:  #catch NaN
            hv = 0
            for i in range(len(az)):
                hv += (ord(az[i].lower())-ord('a')+1)*26**(len(az)-1-i)
            return hv
        else:
            return -1
    new_df[target_col + "_2int"] =  df[target_col].apply(az_to_int)
    new_df[target_col + "_fact"], _ = pd.factorize(df[target_col], na_sentinel=0)
    return new_df


def divide_col(df, col1, col2, fill=-1):
    new_df = pd.DataFrame()
    sr1 = df[col1].fillna(fill)
    sr2 = df[col2].fillna(fill)

    new_df[col1+"_d_"+col2] = sr1 / sr2
    return new_df


def multiply_col(df, col1, col2, fill=-1):
    new_df = pd.DataFrame()
    sr1 = df[col1].fillna(fill)
    sr2 = df[col2].fillna(fill)

    new_df[col1+"_m_"+col2] = sr1 * sr2
    return new_df


def diff_col(df, col1, col2, fill=0):
    new_df = pd.DataFrame()
    sr1 = df[col1].fillna(fill)
    sr2 = df[col2].fillna(fill)

    new_df[col1+"_-_"+col2] = sr1 - sr2
    return new_df


def special_50(df, col):
    new_df = pd.DataFrame()
    norm_target = df[col]-df[col].mean()
    col_name = "spe_50_" + col
    new_sr = df["v50"]*df["v50"] + norm_target*norm_target
    new_df[col_name] = new_sr.fillna(new_sr.max() + 1)
    return new_df


