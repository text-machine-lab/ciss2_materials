import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        """
        gets the querys keys, and values for each attention head.

        the queries and keys are multiplied, and this is scaled, masked,
         softmaxed, and dropouted to get the weights

         these weights are applied to the values via matrix multiplication

        Args:
            q: Query
            k: Key
            v: Value
            mask:

        Returns:

        """

        # MatMul
        attn = torch.bmm(q, k.transpose(1, 2))
        # Scale
        attn = attn / self.temperature

        # Mask
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        # softmax/dropout
        attn = self.softmax(attn)
        attn = self.dropout(attn)

        # Matmul
        output = torch.bmm(attn, v)

        return output, attn
