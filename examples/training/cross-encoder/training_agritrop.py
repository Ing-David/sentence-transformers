from torch.utils.data import DataLoader
import math
from sentence_transformers import LoggingHandler, util
from sentence_transformers.cross_encoder import DocumentCrossEncoder
from sentence_transformers.cross_encoder.evaluation import CECorrelationEvaluator
from sentence_transformers import InputExampleDocument
import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import pandas as pd
from tqdm import tqdm
import nltk

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
#### /print debug information to stdout

# dataset's path
agritrop_dataset_path = 'datasets/corpus_agritrop_training.tsv'

# Define our Cross-Encoder
train_batch_size = 1
num_epochs = 4
model_save_path = 'output/training_agritrop_transformer-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# We use bert-base-cased as base model and set num_labels=1, which predicts a continuous score between 0 and 1
model = DocumentCrossEncoder('squeezebert/squeezebert-uncased', num_labels=1, max_length=512)

# Read Agritrop's dataset
logger.info("Read Agritrop's train dataset")
df_transformer = pd.read_csv(agritrop_dataset_path, sep='\t')

# list sample
train_samples = []
dev_samples = []
test_samples = []

for i in tqdm(range(0, len(df_transformer))):

    if df_transformer['split'][i] == 'dev':
        split_sentences = nltk.tokenize.sent_tokenize(df_transformer['sentence1'][i])
        split_concept_labels = list(df_transformer['sentence2'][i].split(","))
        dev_samples.append(InputExampleDocument(document_sentences=split_sentences, concept_labels=split_concept_labels,
                                                label=df_transformer['score'][i]))

    elif df_transformer['split'][i] == 'test':
        split_sentences = nltk.tokenize.sent_tokenize(df_transformer['sentence1'][i])
        split_concept_labels = list(df_transformer['sentence2'][i].split(","))
        test_samples.append(
            InputExampleDocument(document_sentences=split_sentences, concept_labels=split_concept_labels,
                                 label=df_transformer['score'][i]))

    else:
        split_sentences = nltk.tokenize.sent_tokenize(df_transformer['sentence1'][i])
        split_concept_labels = list(df_transformer['sentence2'][i].split(","))
        train_samples.append(
            InputExampleDocument(document_sentences=split_sentences, concept_labels=split_concept_labels,
                                 label=df_transformer['score'][i]))

# We wrap train_samples (which is a List[InputExample]) into a pytorch DataLoader
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)

# print(len(train_dataloader.dataset))

# We add an evaluator, which evaluates the performance during training
#evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='agritrop-dev')


# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=None,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          sub_batches=1)

##### Load model and eval on test set
# model = DocumentCrossEncoder(model_save_path)

# evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='agritrop-test')
# evaluator(model)
