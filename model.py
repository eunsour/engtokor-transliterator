from config import *


class mT5(object):
    def __init__(self, args):
        self.args = args
        pass

    def build_model(self):

        self.train_params = {
            "max_seq_length": self.args["MAX_SEQUENCE_LENGTH"],
            "train_batch_size": self.args["BATCH_SIZE"],
            "eval_batch_size": self.args["BATCH_SIZE"],
            "num_train_epochs": self.args["EPOCHS"],
            "evaluate_during_training": True,
            "evaluate_during_training_steps": 30000,
            "use_multiprocessing": False,
            "fp16": False,
            "save_steps": -1,
            "save_eval_checkpoints": False,
            "no_cache": True,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "preprocess_inputs": False,
            "num_return_sequences": 1,
            "best_model_dir": f"outputs-{SAVE_NAME}/best_model",
            "output_dir": f"outputs-{SAVE_NAME}/",
            "overwrite_output_dir": True,
            "wandb_project": "eng2kor transliterator using mT5-base",
        }

        self.pred_params = {
            "max_length": self.args["MAX_LENGTH"],
            "max_seq_length": self.args["MAX_SEQUENCE_LENGTH"],
            "length_panalty": self.args["LENGTH_PANALTY"],
            "num_beams": self.args["NUM_BEAMS"],
        }
