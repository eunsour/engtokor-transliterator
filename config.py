import datetime

TRAIN_RATIO = 0.9
VALID_RATIO = 0.1
YEARMONTHDAY = datetime.datetime.now().strftime("%Y%m%d")

DEBUG_MODE = True

def log(*s): # multiple args
    if DEBUG_MODE:
        print(s)

params = {
    'TRAIN_RATIO' : 0.9,
    'VALID_RATIO' : 0.1,
    'BATCH_SIZE' : 1, 
    'EPOCHS' : 20, 
    'MAX_SEQUENCE_LENGTH' : 30
}

SAVE_NAME_list = [
    YEARMONTHDAY,
    str(params['TRAIN_RATIO']),
    str(params['VALID_RATIO']),
    str(params['BATCH_SIZE']),
]

SAVE_NAME = '__'.join(SAVE_NAME_list)
SAVE_NAME = SAVE_NAME.replace('.', '_')
