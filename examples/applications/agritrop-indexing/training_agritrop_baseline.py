import argparse
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

from sentence_transformers import InputExampleDocument, BiEncoder
from sentence_transformers import LoggingHandler

from .eval_agritrop import create_evaluator

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

    parser = argparse.ArgumentParser(description='Train / evaluate baseline indexing system on abstracts')

    parser.add_argument('--dataset', '-d', type=str, nargs=1,
                        help='Path to the TSV corpus to use', dest='dataset',
                        default=['datasets/corpus_agritrop_transformers_abstract.tsv'])
    parser.add_argument('--save-prefix', '-s', type=str, nargs=1,
                        help='Prefix for the model save directory', dest='save_prefix',
                        default=['output/training_agritrop_transformer_baseline-'])
    parser.add_argument('--epochs', '-e', type=int, nargs=1, help="The number of epochs (for training)", dest='epochs',
                        default=[100])

    parser.add_argument('--eval', '-l', type=str, nargs=1, help="Load model from directory and evaluate", dest='eval', default=[])

    args = parser.parse_args()

    # dataset's path
    agritrop_dataset_path = args.dataset[0]

    # Define our Cross-Encoder
    train_batch_size = 1
    num_epochs = args.epochs[0]

    load = len(args.eval) > 0
    model_save_path = args.save_prefix[0] + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Read Agritrop's dataset
    logger.info("Read Agritrop's train dataset")
    df_transformer = pd.read_csv(agritrop_dataset_path, sep='\t')

    # list sample
    train_samples = []
    dev_samples = []
    test_samples = []

    df_document_groups = df_transformer.groupby("doc_ids")

    for group in tqdm(df_document_groups):

        split_document_sentences = nltk.tokenize.sent_tokenize(group[1]['abstract'].iloc[0])
        concept_labels = []
        labels = []
        for index, row in group[1].iterrows():
            split_concept_labels = list(row['sentence2'].split(","))
            concate_concept = " ".join(split_concept_labels)
            concept_labels.append([concate_concept])
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

    # We use bert-base-cased as base model and set num_labels=1, which predicts a continuous score between 0 and 1
    if not load:
        logger.info("Training model using 'squeezebert/squeezebert-uncased'...")
        model = BiEncoder('squeezebert/squeezebert-uncased', num_labels=1, max_length=512, device="cuda:0",
                          freeze_transformer=False)
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
        model.save(model_save_path)
        model.fit(train_dataloader=train_dataloader,
                  epochs=num_epochs,
                  warmup_steps=warmup_steps,
                  output_path=model_save_path, use_amp=False)
        model.save(model_save_path)
    else:
        load_path = args.eval[0]
        logger.info(f"Loading model from {load_path}")
        model = BiEncoder(load_path, num_labels=1, max_length=512, device="cuda:0",
                          freeze_transformer=False)

        logger.info("Evaluating...")
        evaluator_dev, evaluator_test = create_evaluator(df_transformer, "cuda:0")
        evaluator_dev(model)
        evaluator_test(model)
