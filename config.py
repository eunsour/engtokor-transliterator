import torch
import datetime

CUDA = torch.cuda.is_available()
DEBUG_MODE = True

YEARMONTHDAY = datetime.datetime.now().strftime("%Y%m%d")


def log(*s):  # multiple args
    if DEBUG_MODE:
        print(s)


params = {
    "TRAIN_RATIO": 0.9,
    "VALID_RATIO": 0.1,
    "BATCH_SIZE": 1,
    "EPOCHS": 20,
    "MAX_SEQUENCE_LENGTH": 30,
    "MAX_LENGTH": 96,
    "LENGTH_PANALTY": 1,
    "NUM_BEAMS": 1,
}

SAVE_NAME_list = [
    YEARMONTHDAY,
    str(params["TRAIN_RATIO"]),
    str(params["VALID_RATIO"]),
    str(params["BATCH_SIZE"]),
]

SAVE_NAME = "__".join(SAVE_NAME_list)
SAVE_NAME = SAVE_NAME.replace(".", "_")
