import torch
import torch.nn as nn

from sentence_transformers.models.DocumentPooling import DocumentPooling


class DocumentEmbeddingGRU(nn.Module):

    def __init__(self, input_size=768, hidden_size=384, num_layers=2, dropout=0.3, use_attention=True):
        super(DocumentEmbeddingGRU, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.BiGRU = nn.GRU(self.input_size, self.hidden_size, dropout=dropout, num_layers=self.num_layers,
                            bidirectional=True, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(self.input_size, num_heads=1)
        self.document_pooling_layer = DocumentPooling(self.input_size, 'mean')

    def forward(self, x):
        # BiGRU
        # x = x.squeeze()
        gru_out, hid = self.BiGRU(x)
        h2 = hid[2 * self.num_layers - 2, :, :]
        h3 = hid[2 * self.num_layers - 1, :, :]
        h = torch.cat((h2, h3), dim=1)
        # attention's elements
        query = x.transpose(-2, -3)
        key = h.expand(gru_out.shape[1], x.shape[0], h.shape[1])
        value = gru_out.transpose(-2, -3)
        # Multi-head attention
        attention_output, attention_output_weights = self.multihead_attn(query, key, value)
        attention_output = attention_output.transpose(0, 1)
        attention_output = self.document_pooling_layer(attention_output)

        return {'sentence_embedding': attention_output}
