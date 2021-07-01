from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List, Iterable, Union
import transformers
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from .. import DocumentTransformer, util
from ..evaluation import SentenceEvaluator
from ..models import RNN, Transformer, Pooling
from collections import defaultdict
from sentence_transformers.losses import MultipleNegativesRankingLoss
import nltk
logger = logging.getLogger(__name__)
import GPUtil


class DocumentCrossEncoder():
    def __init__(self, model_name: str, num_labels: int = None, max_length: int = None, device: str = None,
                 tokenizer_args: Dict = {},
                 default_activation_function=None):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.
        It does not yield a sentence embedding and does not work for individually sentences.
        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        :param tokenizer_args: Arguments passed to AutoTokenizer
        :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1, else nn.Identity()
        """

        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        # Model RNN
        self.model_rnn = RNN()
        # Model BERT via Transformer
        self.model_transformer = Transformer(model_name)
        # Model Pooling
        self.model_pooling = Pooling(768, 'mean')
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)))
        elif hasattr(self.config,
                     'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(
                self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    # Helper functions for function smart_batching_collate
    @staticmethod
    def get_dimensions(array, level=0):
        yield level, len(array)
        try:
            for row in array:
                yield from DocumentCrossEncoder.get_dimensions(row, level + 1)
        except TypeError:  # not an iterable
            pass

    @staticmethod
    def get_max_shape(array):
        dimensions = defaultdict(int)
        for level, length in DocumentCrossEncoder.get_dimensions(array):
            dimensions[level] = max(dimensions[level], length)
        return [value for _, value in sorted(dimensions.items())]

    @staticmethod
    def iterate_nested_array(array, index=()):
        try:
            for idx, row in enumerate(array):
                yield from DocumentCrossEncoder.iterate_nested_array(row, (*index, idx))
        except TypeError:  # final level
            yield (*index, slice(len(array))), array

    @staticmethod
    def pad(array, fill_value):
        dimensions = DocumentCrossEncoder.get_max_shape(array)
        result = np.full(dimensions, fill_value)
        for index, value in DocumentCrossEncoder.iterate_nested_array(array):
            result[index] = value
        return result

    def smart_batching_collate(self, batch):

        # Initialisation
        document_input_ids = [[] for _ in range(len(batch))]
        document_token_type_ids = [[] for _ in range(len(batch))]
        document_attention_mask = [[] for _ in range(len(batch))]
        concept_input_ids = [[] for _ in range(len(batch))]
        concept_token_type_ids = [[] for _ in range(len(batch))]
        concept_attention_mask = [[] for _ in range(len(batch))]
        tokenized_document_sentences = {}
        tokenized_concept_labels = {}
        labels = []
        # iteration through each batch
        for a in range(len(batch)):
            # Document
            # padding sequence length for each document in order to have the same length for all phrases
            b = self.tokenizer(batch[a].document_sentences, padding=True, truncation='longest_first',
                               max_length=self.max_length)
            for c in b['input_ids']:
                document_input_ids[a].append(c)
            for d in b['token_type_ids']:
                document_token_type_ids[a].append(d)
            for e in b['attention_mask']:
                document_attention_mask[a].append(e)
            # Concept's labels
            # padding sequence length for each concept's labels to have the same length in each document
            f = self.tokenizer(batch[a].concept_labels, padding=True, truncation='longest_first',
                               max_length=self.max_length)
            for g in f['input_ids']:
                concept_input_ids[a].append(g)
            for h in f['token_type_ids']:
                concept_token_type_ids[a].append(h)
            for i in f['attention_mask']:
                concept_attention_mask[a].append(i)
            # Score for each line
            labels.append(batch[a].label)

        # Padding the 3D array global after combining all the documents and concept's labels in the same batch
        # Document
        padding_document_input_ids = self.pad(document_input_ids, fill_value=0)
        padding_document_token_type_ids = self.pad(document_token_type_ids, fill_value=0)
        padding_document_attention_mask = DocumentCrossEncoder.pad(document_attention_mask, fill_value=0)
        # Concept's labels
        padding_concept_input_ids = self.pad(concept_input_ids, fill_value=0)
        padding_concept_token_type_ids = self.pad(concept_token_type_ids, fill_value=0)
        padding_concept_attention_mask = self.pad(concept_attention_mask, fill_value=0)

        # Convert to tensor and put them into correspond dictionary's key
        # Document
        tensor_padding_document_input_ids = torch.tensor(padding_document_input_ids)
        tokenized_document_sentences['input_ids'] = tensor_padding_document_input_ids
        tensor_padding_document_token_type_ids = torch.tensor(padding_document_token_type_ids)
        tokenized_document_sentences['token_type_ids'] = tensor_padding_document_token_type_ids
        tensor_padding_document_attention_mask = torch.tensor(padding_document_attention_mask)
        tokenized_document_sentences['attention_mask'] = tensor_padding_document_attention_mask
        # Concept's labels
        tensor_padding_concept_input_ids = torch.tensor(padding_concept_input_ids)
        tokenized_concept_labels['input_ids'] = tensor_padding_concept_input_ids
        tensor_padding_concept_token_type_ids = torch.tensor(padding_concept_token_type_ids)
        tokenized_concept_labels['token_type_ids'] = tensor_padding_concept_token_type_ids
        tensor_padding_concept_attention_mask = torch.tensor(padding_concept_attention_mask)
        tokenized_concept_labels['attention_mask'] = tensor_padding_concept_attention_mask
        # Score for batch
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(
            self._target_device)

        for name in tokenized_document_sentences:
            tokenized_document_sentences[name] = tokenized_document_sentences[name].to(self._target_device)

        for name in tokenized_concept_labels:
            tokenized_concept_labels[name] = tokenized_concept_labels[name].to(self._target_device)

        return tokenized_document_sentences, tokenized_concept_labels, labels

    def _sub_batching(self, sub_batch_size, document_sentences_batch):
        input_ids = document_sentences_batch['input_ids']
        token_type_ids = document_sentences_batch['token_type_ids']
        attention_mask = document_sentences_batch['attention_mask']
        batch_list = []

        for i in range(0, input_ids.shape[0], sub_batch_size):
            start = i
            end = i + sub_batch_size
            batch_list.append({'input_ids': input_ids[start:end, :],
                               'token_type_ids': token_type_ids[start:end, :],
                               'attention_mask': attention_mask[start:end, :]})
        remainder = input_ids.shape[0] % sub_batch_size
        # if remainder > 0:
        #     start = sub_batch_size *len(batch_list)
        #     batch_list.append({'input_ids': input_ids[start:, :],
        #                        'token_type_ids': token_type_ids[start:, :],
        #                        'attention_mask': attention_mask[start:, :]})
        return batch_list

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            sub_batches=-1,
            loss_fct=None,
            activation_fct=nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        """
        train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model_rnn.to(self._target_device)
        self.model_transformer.to(self._target_device)
        self.model_pooling.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model_rnn.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = DocumentTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps,
                                                           t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = MultipleNegativesRankingLoss(model=self.model_rnn)

        skip_scheduler = False

        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0
            self.model_rnn.zero_grad()
            self.model_rnn.train()

            for document_sentences, concept_labels, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):

                # Initialisation list
                list_concept_labels_pooling = []
                list_document_sentences_pooling = []
                list_scores = []

                for i in tqdm(range(0, len(document_sentences['input_ids'])), desc="Each document in batch"):
                    # dictionary
                    document_sentences_batch = {}
                    concept_labels_batch = {}
                    # Document
                    document_sentences_batch['input_ids'] = document_sentences['input_ids'][i, :, :]
                    document_sentences_batch['token_type_ids'] = document_sentences['token_type_ids'][i, :, :]
                    document_sentences_batch['attention_mask'] = document_sentences['attention_mask'][i, :, :]

                    if sub_batches > 0:
                        # token_embeddings = features['token_embeddings']
                        # cls_token = features['cls_token_embeddings']
                        # attention_mask = features['attention_mask']
                        token_embeddings = []
                        cls_tokens = []
                        attention_masks = []
                        sub_batch_list = self._sub_batching(sub_batches, document_sentences_batch)
                        for sub_batch in tqdm(sub_batch_list, desc="Processing sub-batch"):
                            GPUtil.showUtilization()
                            local_output = self.model_transformer(sub_batch)
                            token_embeddings.append(local_output['token_embeddings'].cpu())
                            cls_tokens.append(local_output['cls_token_embeddings'].cpu())
                            attention_masks.append(local_output['attention_mask'].cpu())

                        document_sentences_batch_output = {
                            'token_embeddings': torch.stack(token_embeddings, dim=0),
                            'cls_token_embeddings': torch.stack(cls_tokens, dim=0),
                            'attention_mask': torch.stack(attention_masks, dim=0)
                        }
                    else:
                        # BERT process via Transformer for document
                        document_sentences_batch_output = self.model_transformer(document_sentences_batch)
                    # Pooling procedure for document
                    output_pooling_document_sentences = self.model_pooling(document_sentences_batch_output)

                    # Concept's labels
                    concept_labels_batch['input_ids'] = concept_labels['input_ids'][i, :, :]
                    concept_labels_batch['token_type_ids'] = concept_labels['token_type_ids'][i, :, :]
                    concept_labels_batch['attention_mask'] = concept_labels['attention_mask'][i, :, :]
                    # BERT process via Transformer for concept's labels
                    concept_labels_batch_output = self.model_transformer(concept_labels_batch)
                    # Pooling procedure for concept's labels
                    output_pooling_concept_labels = self.model_pooling(concept_labels_batch_output)

                    # Put them into the list
                    list_document_sentences_pooling.append(output_pooling_document_sentences['token_embeddings'].data)
                    list_concept_labels_pooling.append(output_pooling_concept_labels['sentence_embedding'].data)

                    # Scores
                    list_scores.append(labels[i].item())

                # Combine each document and concept's labels in the batch
                combine_batch_document_sentences = torch.stack(list_document_sentences_pooling, dim=0)
                combine_batch_concept_labels = torch.stack(list_concept_labels_pooling, dim=0)
                # scores
                tensor_scores = torch.tensor(list_scores, dtype=torch.float)
                # Put Documents and Concept's labels into RNN
                output_batch_document_sentences_rnn = self.model_rnn(combine_batch_document_sentences)
                output_batch_concept_labels_rnn = self.model_rnn(combine_batch_concept_labels)

                if use_amp:
                    with autocast():

                        loss_value = loss_fct([output_batch_document_sentences_rnn, output_batch_concept_labels_rnn],
                                              tensor_scores)

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model_rnn.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    skip_scheduler = scaler.get_scale() != scale_before_step
                else:

                    loss_value = loss_fct([output_batch_document_sentences_rnn, output_batch_concept_labels_rnn],
                                          tensor_scores)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model_rnn.parameters(), max_grad_norm)
                    optimizer.step()

                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()

                training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, output_path, save_best_model, epoch, training_steps, callback)

                    self.model_rnn.zero_grad()
                    self.model_rnn.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, output_path, save_best_model, epoch, -1, callback)


    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):              #{key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):      #Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):    #Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])

    def encode(self, documents: Union[List[str], List[int]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               device: str = None) -> Tensor:
        """
        Computes document embeddings

        :param documents: the documents to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings after Pooling.
        :param device: Which torch.device to use for the computation
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned.
        """
        #self.eval()
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)

        if device is None:
            device = self._target_device

        #self.to(device)

        # Sort documents depend on their length
        length_sorted_idx = np.argsort([-self._text_length(doc) for doc in documents])
        # New list of documents after sorted
        documents_sorted = [documents[idx] for idx in length_sorted_idx]

        embeddings = []

        for start_index in trange(0, len(documents), batch_size, desc="Batches", disable=not show_progress_bar):
            # batch of documents
            documents_batch = documents_sorted[start_index:start_index+batch_size]
            # Initialisation
            document_input_ids = [[] for _ in range(len(documents_batch))]
            document_token_type_ids = [[] for _ in range(len(documents_batch))]
            document_attention_mask = [[] for _ in range(len(documents_batch))]
            tokenized_document_sentences = {}
            # Loop through
            for i in range(len(documents_batch)):
                # Convert document to list of sentences
                list_sentences = nltk.tokenize.sent_tokenize(documents_batch[i])
                # tokenizer the list of sentences
                a = self.tokenizer(list_sentences, padding=True, truncation='longest_first', max_length=512)
                for b in a['input_ids']:
                    document_input_ids[i].append(b)
                for c in a['token_type_ids']:
                    document_token_type_ids[i].append(c)
                for d in a['attention_mask']:
                    document_attention_mask[i].append(d)

            # Padding Document
            padding_document_input_ids = self.pad(document_input_ids, fill_value=0)
            padding_document_token_type_ids = self.pad(document_token_type_ids, fill_value=0)
            padding_document_attention_mask = self.pad(document_attention_mask, fill_value=0)
            # Convert to torch
            tensor_padding_document_input_ids = torch.tensor(padding_document_input_ids)
            tokenized_document_sentences['input_ids'] = tensor_padding_document_input_ids
            tensor_padding_document_token_type_ids = torch.tensor(padding_document_token_type_ids)
            tokenized_document_sentences['token_type_ids'] = tensor_padding_document_token_type_ids
            tensor_padding_document_attention_mask = torch.tensor(padding_document_attention_mask)
            tokenized_document_sentences['attention_mask'] = tensor_padding_document_attention_mask
            # list pooling
            list_document_sentences_pooling = []

            # embeddings' part
            for i in tqdm(range(0, len(tokenized_document_sentences['input_ids'])), desc="Each document in batch"):
                # dictionary
                document_sentences_batch = {}
                # Document
                document_sentences_batch['input_ids'] = tokenized_document_sentences['input_ids'][i, :, :]
                document_sentences_batch['token_type_ids'] = tokenized_document_sentences['token_type_ids'][i, :, :]
                document_sentences_batch['attention_mask'] = tokenized_document_sentences['attention_mask'][i, :, :]
                # BERT process via Transformer for document
                document_sentences_batch_output = self.model_transformer(document_sentences_batch)
                # Pooling procedure for document
                output_pooling_document_sentences = self.model_pooling(document_sentences_batch_output)
                if output_value:
                    list_document_sentences_pooling.append(output_pooling_document_sentences[output_value].data)

            # Combine each document in the batch to create tensor
            combine_batch_document_sentences = torch.stack(list_document_sentences_pooling, dim=0)
            # Put Documents and Concept's labels into RNN
            output_batch_document_sentences_rnn = self.model_rnn(combine_batch_document_sentences)

            embeddings.append(output_batch_document_sentences_rnn['sentence_embedding'])

        all_embeddings = torch.stack(embeddings)

        return all_embeddings


    def predict(self, sentences: List[List[str]],
                batch_size: int = 32,
                show_progress_bar: bool = None,
                num_workers: int = 0,
                activation_fct=None,
                apply_softmax=False,
                convert_to_numpy: bool = True,
                convert_to_tensor: bool = False
                ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.
        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimensions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only,
                                    num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (
                        logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores

    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)


