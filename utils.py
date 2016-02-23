import csv

import numpy as np
import matplotlib.pyplot as plt


def write_ans(fname, header, sample_id, pred):
    with open(fname, "w") as fw:
        writer = csv.writer(fw)
        writer.writerow(header)
        writer.writerows(zip(sample_id, pred))


def show_feature_importances(imp, feat_name):
    plt.scatter(np.arange(len(imp)), imp)
    plt.savefig("imp.png")

    with open("imp.txt", "w") as fw:
        for ind in np.argsort(imp):
            fw.write("{0}: {1}".format(feat_name[ind], imp[ind]))
            fw.write("\n")
