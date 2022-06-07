import pandas as pd

from config import log


def preprocessing(data):
    log("> Preprocessing")

    """
    < Input Formatting >
    ... ... ...
    eng to kor  cable car  케이블 카
    kor to eng  케이블 카   cable car
    eng to kor  Toronto    토론토
    kor to eng  토론토     Toronto
    ... ... ...
    """

    preprocess_inputs = []

    for i, _ in enumerate(data):
        source_eng = data[i].split("\t")[0]
        target_kor = data[i].split("\t")[-1]

        preprocess_inputs.append(["eng to kor", source_eng, target_kor])
        preprocess_inputs.append(["kor to eng", target_kor, source_eng])

    df = pd.DataFrame(
        preprocess_inputs, columns=["prefix", "input_text", "target_text"]
    )

    return df
