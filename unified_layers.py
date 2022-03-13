""" Unifed QANet layers
"""

import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import layers
import qanet_layers
from qanet_layers import FeedForward


class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1): 
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.char_conv = qanet_layers.CharacterConv(char_vectors, hidden_size, drop_prob)
        self.conv1d_word = qanet_layers.FeedForward(word_vectors.size(1), hidden_size, bias=False)
        self.conv1d = qanet_layers.FeedForward(2*hidden_size, hidden_size, bias=False)
        self.hwy = layers.HighwayEncoder(2, hidden_size)
        self.seg_emb = nn.Embedding(2, hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        seg_c = torch.zeros_like(cw_idxs)
        seg_q = torch.ones_like(qw_idxs)
        seg = torch.cat([seg_c, seg_q], dim=-1)
        x1 = torch.cat([cw_idxs, qw_idxs], dim=-1)
        x2 = torch.cat([cc_idxs, qc_idxs], dim=1)

        word_emb = self.word_emb(x1)
        word_emb = F.dropout(word_emb, p=self.drop_prob, training=self.training)
        word_emb = self.conv1d_word(word_emb)
        char_emb = self.char_conv(x2)
        
        emb = torch.cat([word_emb, char_emb], dim=-1)
        emb = self.conv1d(emb)
        emb = self.hwy(emb)   
        seg_emb = self.seg_emb(seg)
        out = emb + seg_emb
        return out 


class VerifierOutput(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = FeedForward(hidden_size*2, 1)
        self.w2 = FeedForward(hidden_size*2, 1)
        self.w3 = FeedForward(hidden_size*4, 1)
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=-1)
        X2 = torch.cat([M1, M3], dim=-1)
        log_p1 = masked_softmax(self.w1(X1).squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(self.w2(X2).squeeze(), mask, log_softmax=True)
        p1 = masked_softmax(self.w1(X1).squeeze(), mask, log_softmax=False).view(X1.shape[0], X1.shape[1], 1)
        p2 = masked_softmax(self.w2(X2).squeeze(), mask, log_softmax=False).view(X2.shape[0], X2.shape[1], 1)
        A = torch.sum(p1*X1, dim=1)
        B = torch.sum(p2*X2, dim=1)
        C = torch.cat([A,B], dim=-1)
        log_pna = self.log_sigmoid(self.w3(C)).squeeze()
        return log_p1, log_p2, log_pna
