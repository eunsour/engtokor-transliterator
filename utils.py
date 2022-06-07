import random

from config import log


def load_data(path):
    data = []
    with open(path, "r", encoding="utf-8-sig") as rf:
        for line in rf:
            line = line.strip()
            data.append(line)
    return data


def split_data(data_list, ratio):
    """ ratio: 0 ~ 1
    """
    random.seed(42)
    random.shuffle(data_list)
    pivot = int(len(data_list) * ratio)
    ldata = data_list[:pivot]
    rdata = data_list[pivot:]
    return ldata, rdata
