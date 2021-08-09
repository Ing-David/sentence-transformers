import logging
import math
from pathlib import Path

import torch.multiprocessing as mp
import os
from datetime import datetime

import nltk
import pandas as pd
import transformers
from torch import nn
import torch.distributed
from fairscale.utils.testing import dist_init
from torch._C._distributed_c10d import HashStore
from torch.utils.data import DataLoader
from tqdm import tqdm

from examples.training.ms_marco.eval_agritrop import create_evaluator
from sentence_transformers import InputExampleDocument
from sentence_transformers import LoggingHandler
from sentence_transformers.cross_encoder import DocumentBiEncoder


def fit_model(i, model, train_dataloader,
              evaluator,
              epochs,
              warmup_steps,
              output_path, use_amp):
    open("ddp_temp", 'a').close()
    open("ffp_temp_rpc", 'a').close()
    dist_init(i, 2, "ddp_temp", "ffp_temp_rpc")
    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator,
              epochs=epochs,
              warmup_steps=warmup_steps,
              output_path=output_path, use_amp=use_amp)


# torch.distributed.init_process_group(backend="nccl",store=HashStore(), world_size=8, rank=0)

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

os.putenv("TOKENIZERS_PARALLELISM", "true")

logger = logging.getLogger(__name__)
#### /print debug information to stdout
if __name__ == '__main__':
    # dataset's path
    agritrop_dataset_path = 'datasets/corpus_agritrop_training_transformers.tsv'

    # Define our Cross-Encoder
    train_batch_size = 1
    num_epochs = 100
    model_save_path = 'output/training_agritrop_transformer-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Read Agritrop's dataset
    logger.info("Read Agritrop's train dataset")
    df_transformer = pd.read_csv(agritrop_dataset_path, sep='\t')

    # list sample
    train_samples = []
    dev_samples = []
    test_samples = []

    df_document_groups = df_transformer.groupby("doc_ids")

    for group in tqdm(df_document_groups):

        split_document_sentences = nltk.tokenize.sent_tokenize(group[1]['sentence1'].iloc[0])
        concept_labels = []
        labels = []
        for index, row in group[1].iterrows():
            split_concept_labels = list(row['sentence2'].split(","))
            concept_labels.append(split_concept_labels)
            labels.append(int(row['score']))
        input_example = InputExampleDocument(document_sentences=split_document_sentences, concept_labels=concept_labels,
                                             labels=labels)
        split = group[1]['split'].iloc[0]
        if split == 'dev':
            dev_samples.append(input_example)
        elif split == 'test':
            test_samples.append(input_example)
        else:
            train_samples.append(input_example)

    # We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=False, batch_size=train_batch_size)

    # print(len(train_dataloader.dataset))

    # We add an evaluator, which evaluates the performance during training
    evaluator_dev, _ = create_evaluator(df_transformer, "cpu")

    # We use bert-base-cased as base model and set num_labels=1, which predicts a continuous score between 0 and 1
    model = DocumentBiEncoder('squeezebert/squeezebert-uncased', num_labels=1, max_length=32, device="cuda:0")

    # Configure the training
    warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
    logger.info("Warmup-steps: {}".format(warmup_steps))
    # Train the model
    # mp.spawn(fit_model, args=(model, train_dataloader,
    #                           None,  # evaluator,
    #                           4,  # epochs
    #                           warmup_steps,
    #                           model_save_path,
    #                           True),  # use amp
    #          nprocs=8, join=True)

    model.fit(train_dataloader=train_dataloader,
              evaluator=evaluator_dev, evaluation_steps=10,
              epochs=num_epochs,
              warmup_steps=warmup_steps,
              output_path=model_save_path, use_amp=True)

    ##### Load model and eval on test set
    # model = DocumentCrossEncoder(model_save_path)

    # evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='agritrop-test')
    # evaluator(model)
