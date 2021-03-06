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
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1): 
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.proj_word = FeedForward(word_vectors.size(1), hidden_size, bias=False)
        self.char_conv = CharacterConv(char_vectors, hidden_size, drop_prob)
        self.proj = FeedForward(2*hidden_size, hidden_size, bias=False)
        self.hwy = layers.HighwayEncoder(2, hidden_size)

    def forward(self, x1, x2):
        word_emb = self.word_emb(x1)
        word_emb = F.dropout(word_emb, p=self.drop_prob, training=self.training)
        word_emb = self.proj_word(word_emb)
        char_emb = self.char_conv(x2)
        emb = torch.cat([word_emb, char_emb], dim=-1)
        emb = self.proj(emb)
        emb = self.hwy(emb)   
        return emb


class CharacterConv(nn.Module):
    def __init__(self, char_vectors, hidden_size, drop_prob=0.1):
        super(CharacterConv, self).__init__()
        self.drop_prob = drop_prob
        self.char_emb = nn.Embedding.from_pretrained(char_vectors)
        self.conv2d = nn.Conv2d(64, hidden_size, kernel_size=(1, 5))
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')

    def forward(self, x):
        char_emb = self.char_emb(x)
        char_emb = char_emb.permute(0, 3, 1, 2)   
        char_emb = F.dropout(char_emb, p=self.drop_prob, training=self.training)
        char_emb = self.conv2d(char_emb)
        char_emb = F.relu(char_emb)
        char_emb, idx = torch.max(char_emb, dim=-1)
        char_emb = char_emb.transpose(1,2)
        return char_emb


class EncoderBlock(nn.Module):  
    def __init__(self, conv_num=4, hidden_size=128, num_head=8, kernel_size=7, drop_prob=0.1):
        super().__init__()
        self.conv_num = conv_num
        self.drop_prob = drop_prob
        self.residual_connection = ResidualConnection(drop_prob)
        self.layer_norm_attention = nn.LayerNorm(hidden_size)
        self.layer_norm_ffn = nn.LayerNorm(hidden_size)
        self.repeated_convs = RepeatedDepthwiseSeparableConv(conv_num, hidden_size, hidden_size, kernel_size, drop_prob)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_head, dropout=drop_prob, batch_first=True)
        self.ffn_1 = FeedForward(hidden_size, hidden_size, relu=True, bias=True)
        self.ffn_2 = FeedForward(hidden_size, hidden_size, bias=True)

    def forward(self, x, mask):
        out = PosEncoder(x)
        out = self.repeated_convs(out)

        res = out
        out = self.layer_norm_attention(out)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out, _ = self.self_attention(out, out, out, mask)
        out = self.residual_connection(out, res)

        res = out
        out = self.layer_norm_ffn(out)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.ffn_1(out)
        out = self.ffn_2(out)
        out = self.residual_connection(out, res)
        return out


class RepeatedDepthwiseSeparableConv(nn.Module):
    def __init__(self, repeat, in_channels, out_channels, kernel_size, drop_prob=0.1):
        super().__init__()
        self.drop_prob = drop_prob
        self.convs = nn.ModuleList([DepthwiseSeparableConv(in_channels, out_channels, kernel_size) for _ in range(repeat)])
        self.residual_connection = ResidualConnection(drop_prob)
        self.layer_norm_convs = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(repeat)])

    def forward(self, x):
        out = x
        for i, conv in enumerate(self.convs):
            res = out
            out = self.layer_norm_convs[i](out)
            if i % 2 == 0:
                out = F.dropout(out, p=self.drop_prob, training=self.training)
            out = conv(out.transpose(1,2)).transpose(1,2)
            out = F.relu(out)
            out = self.residual_connection(out, res)
        return out


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, 
                                   groups=in_channels, padding=kernel_size//2, bias=False)
        self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)           
        return out


class FeedForward(nn.Module):
    def __init__(self, in_features, out_features, relu=False, bias=False):
        super().__init__()
        self.relu = relu
        self.out = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        if self.relu is True:
            nn.init.kaiming_normal_(self.out.weight, nonlinearity='relu')
        else:
            nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        return F.relu(self.out(x)) if self.relu else self.out(x)


class ResidualConnection(nn.Module):
    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, res):
        if self.training:
            return F.dropout(x, self.drop_prob, training=self.training) + res
        else:
            return x + res


class Output(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = FeedForward(hidden_size*2, 1)
        self.w2 = FeedForward(hidden_size*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1, X2 = torch.cat([M1, M2], dim=-1), torch.cat([M1, M3], dim=-1)
        log_p1 = masked_softmax(self.w1(X1).squeeze(), mask, log_softmax=True)
        log_p2 = masked_softmax(self.w2(X2).squeeze(), mask, log_softmax=True)
        return log_p1, log_p2


# Adapted from https://github.com/BangLiu/QANet-PyTorch ##############################################################
def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):  
    length = x.shape[1]
    channels = x.shape[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(x.get_device()))

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
######################################################################################################################
