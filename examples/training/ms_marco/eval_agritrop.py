"""
This script runs the evaluation of an SBERT msmarco model on the
MS MARCO dev dataset and reports different performances metrices for cossine similarity & dot-product.

Usage:
python eval_msmarco.py model_name [max_corpus_size_in_thousands]
"""

from sentence_transformers import  LoggingHandler, DocumentCrossEncoder, evaluation, util, models
import logging
import sys
import os
import pandas as pd
from tqdm import tqdm

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

#Name of the SBERT model
model_name = 'bert-base-uncased'

####  Load model

model = DocumentCrossEncoder(model_name)

### Data files
data_folder = 'agritrop-data'
os.makedirs(data_folder, exist_ok=True)

#dataframe-path

dataframe_path = "agritrop-data/corpus_agritrop_training_transformers.tsv"

# dataset
dataframe = pd.read_csv(dataframe_path, sep="\t")


### Load data

queries_dev = {}
queries_test = {}
relevant_concept_ids_dev ={}
relevant_concept_ids_test = {}
corpus_concept_ids_dev = {}
corpus_concept_ids_test = {}

# Charge data
for i in tqdm(range(0,len(dataframe))):

  if dataframe['split'][i] == 'dev':
    queries_dev[dataframe['doc_ids'][i]] = dataframe['sentence1'][i]
    corpus_concept_ids_dev[dataframe['concept_ids'][i]] = dataframe['sentence2'][i]

  elif dataframe['split'][i] == 'test':
    queries_test[dataframe['doc_ids'][i]] = dataframe['sentence1'][i]
    corpus_concept_ids_test[dataframe['concept_ids'][i]] = dataframe['sentence2'][i]

# relevant concept_ids for dev and test dataset
dataframe_new_column = dataframe.groupby(['doc_ids'])['concept_ids'].apply(','.join).reset_index().rename(columns={'concept_ids':'merge_concept_ids'})
dataframe_merge = dataframe.merge(dataframe_new_column, on='doc_ids')

# Charge relevant data
for i in tqdm(range(0,len(dataframe_merge))):

  if dataframe_merge['split'][i] == 'dev':
    if dataframe_merge['score'][i] == 1:
      relevant_concept_ids_dev[dataframe_merge['doc_ids'][i]] = set(dataframe_merge['merge_concept_ids'][i].split(","))
  if dataframe_merge['split'][i] == 'test':
    if dataframe_merge['score'][i] == 1:
      relevant_concept_ids_test[dataframe_merge['doc_ids'][i]] = set(dataframe_merge['merge_concept_ids'][i].split(","))

## Run evaluator
logging.info("Queries: {}".format(len(queries_dev)))
logging.info("Corpus: {}".format(len(corpus_concept_ids_dev)))

ir_evaluator = evaluation.InformationRetrievalEvaluator(queries_dev, corpus_concept_ids_dev, relevant_concept_ids_dev,
                                                        show_progress_bar=True,
                                                        corpus_chunk_size=100000,
                                                        precision_recall_at_k=[10, 100],
                                                        name="agritrop dev")

ir_evaluator(model)
