import logging
import argparse

import warnings
warnings.filterwarnings("ignore")

from simpletransformers.t5 import T5Model

from utils import *
from config import *
from preprocess import *
from model import mT5

parser = argparse.ArgumentParser(description="")
parser.add_argument("--train", action="store_true", help="Train Mode")
parser.add_argument("--test", action="store_true", help="Test Mode")
parser.add_argument("--decode", action="store_true", help="Decode Mode")
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


data_path = "./dataset/data.txt"
pretrained_model_path = "./outputs/best_model"


class Transliterator(object):
    def __init__(self):
        self._load_data()
        self._process_data()

    def _load_data(self):
        raw = load_data(data_path)
        train, eval = split_data(raw, ratio=params["TRAIN_RATIO"])

        log(">> total number of data:", len(train) + len(eval))
        log(">> number of train data:", len(train))
        log(">> number of test data:", len(eval))

        self.train = train
        self.eval = eval

    def _process_data(self):
        self.train_df = preprocessing(self.train)
        self.eval_df = preprocessing(self.eval)

        self.mt5 = mT5(params)
        self.mt5.build_model()

    def run_train(self):
        log("> Train Model Start...")

        self.train_df["prefix"], self.eval_df["prefix"] = "", ""

        self.model = T5Model(
            "mt5", "google/mt5-base", use_cuda=CUDA, args=self.mt5.train_params
        )
        self.model.train_model(self.train_df, eval_data=self.eval_df)

    def use_pretrained_model(self):
        self.model = T5Model(
            "mt5", pretrained_model_path, use_cuda=CUDA, args=self.mt5.pred_params
        )


def prediction(model, text):
    log("> Pretrained Model Start...")
    result = model.model.predict(text)
    return result


if __name__ == "__main__":

    model = Transliterator()

    if args.train:
        model.run_train()

    elif args.test:
        model.use_pretrained_model()
        test_list = [
            "attention",
            "tokenizer",
            "transliterator",
            "suddenly",
            "mecab",
            "adidas",
            "nike",
        ]

        [print(i) for i in prediction(model, test_list)]

    elif args.decode:
        model.use_pretrained_model()
        input_text = str(input(">> "))
        print("".join(prediction(model, input_text)))
