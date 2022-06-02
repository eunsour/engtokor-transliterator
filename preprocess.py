import pandas as pd

from config import log

def preprocessing(data):
    log('> Preprocessing')

    data_lst = []
    
    for i, _ in enumerate(data):
        source_eng = data[i].split('\t')[0]
        target_kor = data[i].split('\t')[-1]

        data_lst.append(['eng to kor', source_eng, target_kor])
        data_lst.append(["kor to eng", target_kor, source_eng])

    df = pd.DataFrame(
        data_lst, columns=["prefix", "input_text", "target_text"]
    )

    return df