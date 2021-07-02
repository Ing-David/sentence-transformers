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
agritrop_dataset_path = 'datasets/corpus_agritrop_training_transformers.tsv'

# Define our Cross-Encoder
train_batch_size = 1
num_epochs = 4
model_save_path = 'output/training_agritrop_transformer-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# We use bert-base-cased as base model and set num_labels=1, which predicts a continuous score between 0 and 1
model = DocumentCrossEncoder('squeezebert/squeezebert-uncased', num_labels=1, max_length=32)

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
# evaluator = CECorrelationEvaluator.from_input_examples(dev_samples, name='agritrop-dev')


# Configure the training
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logger.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_dataloader=train_dataloader,
          evaluator=None,
          epochs=num_epochs,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          sub_batches=2, use_amp=True)

##### Load model and eval on test set
# model = DocumentCrossEncoder(model_save_path)

# evaluator = CECorrelationEvaluator.from_input_examples(test_samples, name='agritrop-test')
# evaluator(model)
