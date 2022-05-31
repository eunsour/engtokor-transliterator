import datetime

TRAIN_RATIO = 0.9
VALID_RATIO = 0.1
YEARMONTHDAY = datetime.datetime.now().strftime("%Y%m%d")

params = {
    'TRAIN_RATIO' : 0.95,
    'VALID_RATIO' : 0.05,
    'BATCH_SIZE' : 16, 
    'EPOCHS' : 300, 
    'MAX_SEQUENCE_LENGTH' : 50
}

SAVE_NAME_list = [
    YEARMONTHDAY,
    str(params['TRAIN_RATIO']),
    str(params['VALID_RATIO']),
    str(params['BATCH_SIZE']),
]

SAVE_NAME = '__'.join(SAVE_NAME_list)
SAVE_NAME = SAVE_NAME.replace('.', '_')
