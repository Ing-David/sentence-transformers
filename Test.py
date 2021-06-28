from __future__ import unicode_literals, print_function
from sentence_transformers import DocumentTransformer
from spacy.lang.en import English  # updated
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import RNN
import re
import pandas as pd
import torch.nn as nn
import torch
import nltk
# nltk.download('punkt')
from transformers import AutoTokenizer

df = pd.read_csv("corpus_titres_abstracts_corps_eng_articles-type_1_2_1000_limit.csv")

#sentences = doc_to_sent(df['body_grobid'][0])
#model = SentenceTransformer('bert-base-cased')
#text_embeddings = model.encode(sentences,convert_to_tensor= True)



#print(type(text_embeddings))
#sentences = nltk.tokenize.sent_tokenize(df['body_grobid'][0])
#print(len(sentences))
'''
model = DocumentTransformer('bert-base-uncased')
text_embeddings = model.encode(df['body_grobid'][0])
print(text_embeddings.shape)
'''
#config =
#model = RNN('bert-base-cased',self.config,768, 1)
