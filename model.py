from simpletransformers.t5 import T5Model, T5Args

class Seq2seqAtt(object):
    def __init__(self, args):
        self.args = args
        pass

    def build_model(self):
        model_args = T5Args()
        model_args.max_seq_length = self.args["MAX_SEQUENCE_LENGTH"]
        model_args.train_batch_size = self.args["BATCH_SIZE"]
        model_args.eval_batch_size = self.args["BATCH_SIZE"]
        model_args.num_train_epochs = self.args["EPOCHS"]
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
        model_args.best_model_dir = f"outputs/best_model"
        model_args.output_dir = f"outputs/"
        model_args.wandb_project = "eng2kor transliterator using mT5-base"

        self.model = T5Model("mt5", "google/mt5-base", use_cuda=False, args=model_args)