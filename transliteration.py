import logging
import argparse
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from simpletransformers.t5 import T5Model, T5Args

from config import *

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

    def _load_data(self):
        train_lst, eval_lst = [], []
        header = ["prefix", "input_text", "target_text"]

        data = pd.read_csv(data_path, sep="\t", encoding="utf-8-sig")
        data["prefix"] = "eng to kor"

        train, eval = train_test_split(data, test_size=params['VALID_RATIO'], random_state=42)

        train_to_lst = train[header].values.tolist()
        eval_to_lst = eval[header].values.tolist()

        for data in train_to_lst:
            train_lst.append([data[0], data[1], data[2]])
            train_lst.append(["kor to eng", data[2], data[1]])

        for data in eval_to_lst:
            eval_lst.append([data[0], data[1], data[2]])
            eval_lst.append(["kor to eng", data[2], data[1]])

        train_df = pd.DataFrame(
            train_lst, columns=["prefix", "input_text", "target_text"]
        )
        eval_df = pd.DataFrame(
            eval_lst, columns=["prefix", "input_text", "target_text"]
        )

        self.train_df = train_df
        self.eval_df = eval_df
        
        
    def run_train(self):
        log("> Train Model Start...")

        self.train_df["prefix"], self.eval_df["prefix"] = "", ""

        model_args = T5Args()
        model_args.max_seq_length = params["MAX_SEQUENCE_LENGTH"]
        model_args.train_batch_size = params["BATCH_SIZE"]
        model_args.eval_batch_size = params["BATCH_SIZE"]
        model_args.num_train_epochs = params["EPOCHS"]
        model_args.evaluate_during_training = True
        model_args.evaluate_during_training_steps = 30000
        model_args.use_multiprocessing = False
        model_args.fp16 = False
        model_args.save_steps = -1
        model_args.save_eval_checkpoints = False
        model_args.no_cache = True
        model_args.reprocess_input_data = True
        model_args.overwrite_output_dir = True
        model_args.preprocess_inputs = False
        model_args.num_return_sequences = 1
        model_args.best_model_dir = f"outputs-{SAVE_NAME}-/best_model"
        model_args.output_dir = f"outputs-{SAVE_NAME}-/"
        model_args.wandb_project = "eng2kor transliterator using mT5-base"

        self.model = T5Model("mt5", "google/mt5-base", use_cuda=True, args=model_args)
        self.model.train_model(self.train_df, eval_data=self.eval_df)


    def use_pretrained_model(self):
        model_args = T5Args()
        model_args.max_seq_length = params["MAX_SEQUENCE_LENGTH"]

        self.model = T5Model(
            "mt5", pretrained_model_path, use_cuda=True, args=model_args
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
