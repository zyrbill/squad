""" QANet layers
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax
import layers

class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1):   # use proj instead of Conv1D
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.char_emb = nn.Embedding.from_pretrained(char_vectors)
        self.conv2d = nn.Conv2d(64, hidden_size, kernel_size=(1, 5))
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        #self.proj = nn.Linear(word_vectors.size(1) + 200, hidden_size, bias=False)
        self.conv1d_word = Initialized_Conv1d(word_vectors.size(1), hidden_size, bias=False)
        self.conv1d = Initialized_Conv1d(2*hidden_size, hidden_size, bias=False)
        self.hwy = layers.HighwayEncoder(2, hidden_size)

    def forward(self, x1, x2):
        word_emb = self.word_emb(x1)
        word_emb = F.dropout(word_emb, p=self.drop_prob, training=self.training)
        word_emb = word_emb.transpose(1,2)
        word_emb = self.conv1d_word(word_emb)
        
        char_emb = self.char_emb(x2)
        char_emb = char_emb.permute(0, 3, 1, 2)    # batch, char_channel, seq_len, char_limit
        char_emb = F.dropout(char_emb, p=self.drop_prob, training=self.training)
        char_emb = self.conv2d(char_emb)
        char_emb = F.relu(char_emb)
        char_emb, idx = torch.max(char_emb, dim=-1)

        emb = torch.cat([word_emb, char_emb], dim=1)
        emb = self.conv1d(emb)
        emb = emb.transpose(1,2)
        emb = self.hwy(emb)   
        #emb = emb.transpose(1,2) # (batch_size, seq_len, hidden_size)
        return emb

class EncoderBlock(nn.Module):  # do not understando posEncoder
    def __init__(self, conv_num=4, hidden_size=128, num_head=8, kernel_size=7, drop_prob=0.1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(hidden_size, hidden_size, kernel_size) for _ in range(conv_num)])
        #self.self_att = SelfAttention(hidden_size, num_head, drop_prob=drop_prob)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_head, dropout=drop_prob, batch_first=True)
        self.ffn_1 = Initialized_Conv1d(hidden_size, hidden_size, relu=True, bias=True)
        self.ffn_2 = Initialized_Conv1d(hidden_size, hidden_size, bias=True)
        self.layer_norm_convs = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(conv_num)])
        self.layer_norm_attention = nn.LayerNorm(hidden_size)
        self.layer_norm_ffn = nn.LayerNorm(hidden_size)
        self.conv_num = conv_num
        self.drop_prob = drop_prob

    def forward(self, x, mask):
        #total_layers = (self.conv_num + 1) * blks
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.layer_norm_convs[i](out)
            if i % 2 == 0:
                out = F.dropout(out, p=self.drop_prob, training=self.training)
            out = conv(out.transpose(1,2)).transpose(1,2)
            out = self.residual_block(out, res)

        res = out
        out = self.layer_norm_attention(out)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        #print('self_attention', out.size())
        out, _ = self.self_attention(out, out, out, mask)
        #print('self_attention 2', out.size())
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.residual_block(out, res)

        res = out
        out = self.layer_norm_ffn(out)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.ffn_1(out.transpose(1,2)).transpose(1,2)
        out = self.ffn_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.residual_block(out, res)
        return out

    def residual_block(self, out, res):
        if self.training:
            return F.dropout(out, self.drop_prob, training=self.training) + res
        else:
            return out + res

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, padding=kernel_size//2, bias=False)
        self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)             # bad mistake, no depthwise conv
        return out

class Output(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = Initialized_Conv1d(hidden_size*2, 1)
        self.w2 = Initialized_Conv1d(hidden_size*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=-1)
        X2 = torch.cat([M1, M3], dim=-1)
        Y1 = mask_logits(self.w1(X1.transpose(1,2)).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2.transpose(1,2)).squeeze(), mask)
        p1, p2 = F.log_softmax(Y1, dim=1), F.log_softmax(Y2, dim=1)
        return p1, p2

def mask_logits(target, mask):
    mask = mask.type(torch.float32)
    return target * mask + (1 - mask) * (-1e30)  # !!!!!!!!!!!!!!!  do we need * mask after target?


class Initialized_Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, groups=1,
                 relu=False, bias=False):
        super().__init__()
        self.out = nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding, groups=groups, bias=bias)
        if relu is True:
            self.relu = True
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            self.relu = False
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        if self.relu is True:
            return F.relu(self.out(x))
        else:
            return self.out(x)

def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):   # previous wrong (no added signal)
    #print('in posEncoder !!!!!!!!!', x.size())
    #x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    #print('signal size', signal.size())
    return (x + signal.to(x.get_device()))#.transpose(1, 2)

def get_timing_signal(length, channels,
                      min_timescale=1.0, max_timescale=1.0e4):
    position = torch.arange(length).type(torch.float32)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales).type(torch.float32) * -log_timescale_increment)
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim = 1)
    m = nn.ZeroPad2d((0, (channels % 2), 0, 0))
    signal = m(signal)
    signal = signal.view(1, length, channels)
    return signal



