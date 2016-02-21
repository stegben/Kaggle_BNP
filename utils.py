import csv


def write_ans(fname, header, sample_id, pred):
    with open(fname, "w") as fw:
        writer = csv.writer(fw)
        writer.writerow(header)
        writer.writerows(zip(sample_id, pred))
