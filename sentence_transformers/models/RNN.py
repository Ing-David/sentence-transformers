import torch
import torch.nn as nn


class RNN(nn.Module):

    def __init__(self, input_size=768, hidden_size=384, num_layers=2):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.BiGRU = nn.GRU(self.input_size, self.hidden_size, dropout=0.5, num_layers=self.num_layers,
                            bidirectional=True, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(self.input_size, num_heads=1)

    def forward(self, x):
        # BiGRU
        gru_out, hid = self.BiGRU(x)
        h2 = hid[2, :, :]
        h3 = hid[3, :, :]
        h = torch.cat((h2, h3), dim=1)
        # attention's elements
        query = x.transpose(-2, -3)
        key = h.expand(gru_out.shape[1], x.shape[0], h.shape[1])
        value = gru_out.transpose(-2, -3)
        # Multi-head attention
        attn_output, attn_output_weights = self.multihead_attn(query, key, value)
        attention_output = attn_output.transpose(0, 1)

        return {'sentence_embedding': attention_output}
