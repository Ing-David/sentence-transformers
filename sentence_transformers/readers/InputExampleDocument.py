from typing import Union, List


class InputExampleDocument:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', document_sentences: List[str] = None, concept_labels: List[List[str]] = None,  labels: List[Union[int, float]] = None):
        """
        Creates one InputExample with the given texts, guid and label
        :param guid
            id for the example
        :param texts
            the texts for the example. Note, str.strip() is called on the texts
        :param label
            the label for the example
        """
        self.guid = guid
        self.document_sentences = document_sentences
        self.concept_labels = concept_labels
        self.labels = labels

    def __str__(self):
        return "<InputExample> label: {}, document_sentences: {}, concept_labels: {}".format(str(self.labels), "; ".join(self.document_sentences), "; ".join([",".join(labels) for labels in self.concept_labels]))
