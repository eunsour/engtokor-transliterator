import logging
import argparse
import gradio as gr

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
parser.add_argument("--gradio", action="store_true", help="gradio Mode")
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


def prediction(input_text):
    log("> Pretrained Model Start...")
    model.use_pretrained_model()
    return "".join(model.model.predict(input_text))


if __name__ == "__main__":

    model = Transliterator()

    if args.train:
        model.run_train()


    elif args.test:
        log("> Pretrained Model Start...")
        model.use_pretrained_model()

        test_list = [
            "machinelearning",
            "deeplearning",
            "transformer",
            "attention",
        ]

        [print(f'{i}\t:\t{j}') for i, j in zip(test_list, model.model.predict(test_list))]


    elif args.decode:
        print("종료는 'q' 입니다.")

        while True:
            input_text = str(input(">> "))

            if input_text == "q":
                break

            print(prediction(input_text))


    elif args.gradio:
        iface = gr.Interface(
            fn=prediction,
            inputs=gr.inputs.Textbox(type="str", label="Input Text"),
            outputs=gr.outputs.Textbox(),
            title="English to Korean Transliteration",
            description="Model for Transliterating English to Korean using a Google mT-5</a>",
            article='Author: <a href="https://huggingface.co/eunsour">Eunsoo Kang</a> . Using training and inference script from <a href="https://github.com/eunsour/engtokor-transliterator.git">eunsour/engtokor-transliterator</a><p><center><img src="https://visitor-badge.glitch.me/badge?page_id=eunsour/en-ko-transliterator" alt="visitor badge"></center></p>',
            examples=[["transformer"], ["attention"]],
        )

        iface.launch(enable_queue=True)
