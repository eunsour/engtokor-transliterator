import logging
import argparse
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from simpletransformers.t5 import T5Model, T5Args

from utils import *
from config import *
from preprocess import *
from model import Seq2seqAtt

parser = argparse.ArgumentParser(description="")
parser.add_argument("--train", action="store_true", help="Train Mode")
parser.add_argument("--test", action="store_true", help="Test Mode")
args = parser.parse_args()


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


data_path = "./dataset/data.txt"
pretrained_model_path = "./outputs/checkpoint-4168-epoch-4"


class Transliterator(object):
    def __init__(self):
        self._load_data()
        self._process_data()


    def _load_data(self):
        raw = load_data(data_path)
        train, eval = split_data(raw, ratio=params['TRAIN_RATIO'])

        log(">> total number of data:", len(train) + len(eval))
        log(">> number of train data:", len(train))
        log(">> number of test data:", len(eval))

        self.train = train
        self.eval = eval
    
    def _process_data(self):
        self.train_df = preprocessing(self.train)
        self.eval_df = preprocessing(self.eval)

        self.seq2seq_att = Seq2seqAtt(params)
        self.seq2seq_att.build_model()
        
        
    def run_train(self):
        log("> Train Model Start...")

        self.train_df["prefix"], self.eval_df["prefix"] = "", ""

        self.model =  self.seq2seq_att.model
        self.model.train_model(self.train_df, eval_data=self.eval_df)


    def use_pretrained_model(self):
        model_args = T5Args()
        model_args.max_seq_length = params["MAX_SEQUENCE_LENGTH"]

        self.model = T5Model(
            "mt5", pretrained_model_path, use_cuda=False, args=model_args
        )


def prediction(model, text):
    log("> Pretrained Model Start...")
    return print(model.model.predict(text))


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

        prediction(model, test_list)
