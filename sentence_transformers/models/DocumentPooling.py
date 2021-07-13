import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json


class DocumentPooling(nn.Module):
    """Performs pooling (max or mean) on the sentence embeddings.

    Using pooling, it generates from a variable number of sentences, a fixed size document embedding. 
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode: Can be a string: mean/max. If set, overwrites the other pooling_mode_* settings
    :param pooling_mode_max_sentences: Use max in each dimension over all tokens.
    :param pooling_mode_mean_sentences: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_sentences: Perform mean-pooling, but devide by sqrt(input_length).
    """

    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode: str = None,
                 pooling_mode_max_sentences: bool = False,
                 pooling_mode_mean_sentences: bool = True,
                 pooling_mode_mean_sqrt_len_sentences: bool = False,
                 ):
        super(DocumentPooling, self).__init__()

        self.config_keys = ['word_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
                            'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max']
            pooling_mode_max_sentences = (pooling_mode == 'max')
            pooling_mode_mean_sentences = (pooling_mode == 'mean')

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_mean_tokens = pooling_mode_mean_sentences
        self.pooling_mode_max_tokens = pooling_mode_max_sentences
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_sentences

        pooling_mode_multiplier = sum(
            [pooling_mode_max_sentences, pooling_mode_mean_sentences, pooling_mode_mean_sqrt_len_sentences])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Tensor):

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_max_tokens:
            max_over_time = torch.max(features, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            sum_embeddings = torch.sum(features, 1)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / features.shape[0])
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(features.shape[0]))

        features = torch.cat(output_vectors, 1)
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return DocumentPooling(**config)
