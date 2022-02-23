""" QANet layers
"""

import math


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from util import masked_softmax

class Embedding(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1):
        super(Embedding, self).__init__()
        self.drop_prob = drop_prob
        self.word_emb = nn.Embedding.from_pretrained(word_vectors)
        self.char_emb = nn.Embedding.from_pretrained(char_vectors)
        self.conv2d = nn.Conv2d(64, 200, kernel_size=(1, 5))
        nn.init.kaiming_normal_(self.conv2d.weight, nonlinearity='relu')
        self.conv1d = Initialized_Conv1d(word_vectors.size(1) + 200, hidden_size, bias=False)
        self.hwy = HighwayEncoder(2, hidden_size)

    def forward(self, x1, x2):
        word_emb = self.word_emb(x1)
        word_emb = F.dropout(word_emb, p=self.drop_prob, training=self.training)
        word_emb = word_emb.transpose(1,2)
        
        char_emb = self.char_emb(x2)

        char_emb = char_emb.permute(0, 3, 1, 2)
        char_emb = F.dropout(char_emb, p=self.drop_prob, training=self.training)
        char_emb = self.conv2d(char_emb)
        char_emb = F.relu(char_emb)
        char_emb, idx = torch.max(char_emb, dim=-1)

        emb = torch.cat([word_emb, char_emb], dim=1)
        emb = F.dropout(emb, self.drop_prob, training=self.training)
        emb = self.conv1d(emb)
        emb = F.relu(emb)
        emb = self.hwy(emb)   # (batch_size, seq_len, hidden_size)
        return emb

class EncoderBlock(nn.Module):
    def __init__(self, conv_num, hidden_size, num_head, k, drop_prob=0.1):
        super().__init__()
        self.convs = nn.ModuleList([DepthwiseSeparableConv(hidden_size, hidden_size, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(hidden_size, num_head, drop_prob=drop_prob)
        self.FFN_1 = Initialized_Conv1d(hidden_size, hidden_size, relu=True, bias=True)
        self.FFN_2 = Initialized_Conv1d(hidden_size, hidden_size, bias=True)
        self.norm_C = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(conv_num)])
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.conv_num = conv_num
        self.drop_prob = drop_prob

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 1) * blks
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out.transpose(1,2)).transpose(1,2)
            if (i) % 2 == 0:
                out = F.dropout(out, p=self.drop_prob, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, self.drop_prob*float(l)/total_layers)
            l += 1
        res = out
        out = self.norm_1(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.self_att(out, mask)
        out = self.layer_dropout(out, res, self.drop_prob*float(l)/total_layers)
        l += 1
        res = out

        out = self.norm_2(out.transpose(1,2)).transpose(1,2)
        out = F.dropout(out, p=self.drop_prob, training=self.training)
        out = self.FFN_1(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, self.drop_prob*float(l)/total_layers)
        return out

    def layer_dropout(self, inputs, residual, drop_prob):
        if self.training == True:
            pred = torch.empty(1).uniform_(0,1) < self.drop_prob
            if pred:
                return residual
            else:
                return F.dropout(inputs, self.drop_prob, training=self.training) + residual
        else:
            return inputs + residual

class CQAttention(nn.Module):
    def __init__(self, hidden_size, drop_prob=0.1):
        super().__init__()
        w4C = torch.empty(hidden_size, 1)
        w4Q = torch.empty(hidden_size, 1)
        w4mlu = torch.empty(1, 1, hidden_size)
        nn.init.xavier_uniform_(w4C)
        nn.init.xavier_uniform_(w4Q)
        nn.init.xavier_uniform_(w4mlu)
        self.w4C = nn.Parameter(w4C)
        self.w4Q = nn.Parameter(w4Q)
        self.w4mlu = nn.Parameter(w4mlu)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)
        self.drop_prob = drop_prob

    def forward(self, C, Q, Cmask, Qmask):
        C = C.transpose(1, 2)
        Q = Q.transpose(1, 2)
        batch_size_c = C.size()[0]
        batch_size, Lc, hidden_size = C.shape
        batch_size, Lq, hidden_size = Q.shape
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.view(batch_size_c, Lc, 1)
        Qmask = Qmask.view(batch_size_c, 1, Lq)
        S1 = F.softmax(mask_logits(S, Qmask), dim=2)
        S2 = F.softmax(mask_logits(S, Cmask), dim=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out.transpose(1, 2)

    def trilinear_for_attention(self, C, Q):
        batch_size, Lc, hidden_size = C.shape
        batch_size, Lq, hidden_size = Q.shape
        drop_prob = self.drop_prob
        C = F.dropout(C, p=drop_prob, training=self.training)
        Q = F.dropout(Q, p=drop_prob, training=self.training)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, Lq])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, Lc, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1,2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res

class Pointer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w1 = Initialized_Conv1d(hidden_size*2, 1)
        self.w2 = Initialized_Conv1d(hidden_size*2, 1)

    def forward(self, M1, M2, M3, mask):
        X1 = torch.cat([M1, M2], dim=1)
        X2 = torch.cat([M1, M3], dim=1)
        Y1 = mask_logits(self.w1(X1).squeeze(), mask)
        Y2 = mask_logits(self.w2(X2).squeeze(), mask)
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

def PosEncoder(x, min_timescale=1.0, max_timescale=1.0e4):
    x = x.transpose(1, 2)
    length = x.size()[1]
    channels = x.size()[2]
    signal = get_timing_signal(length, channels, min_timescale, max_timescale)
    return (x + signal.to(x.get_device())).transpose(1, 2)

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

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        return F.relu(self.pointwise_conv(self.depthwise_conv(x)))


class HighwayEncoder(nn.Module):
    def __init__(self, layer_num, size):
        super().__init__()
        self.n = layer_num
        self.linear = nn.ModuleList([Initialized_Conv1d(size, size, relu=False, bias=True) for _ in range(self.n)])
        self.gate = nn.ModuleList([Initialized_Conv1d(size, size, bias=True) for _ in range(self.n)])

    def forward(self, x):
        #x: shape [batch_size, hidden_size, length]
        drop_prob = 0.1
        for i in range(self.n):
            gate = F.sigmoid(self.gate[i](x))
            nonlinear = self.linear[i](x)
            nonlinear = F.dropout(nonlinear, p=drop_prob, training=self.training)
            x = gate * nonlinear + (1 - gate) * x
        return x

class SelfAttention(nn.Module):
    def __init__(self, d_model, num_head, drop_prob):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.drop_prob = drop_prob
        self.mem_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model*2, kernel_size=1, relu=False, bias=False)
        self.query_conv = Initialized_Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=1, relu=False, bias=False)

        bias = torch.empty(1)
        nn.init.constant_(bias, 0)
        self.bias = nn.Parameter(bias)

    def forward(self, queries, mask):
        memory = queries

        memory = self.mem_conv(memory)
        query = self.query_conv(queries)
        memory = memory.transpose(1, 2)
        query = query.transpose(1, 2)
        Q = self.split_last_dim(query, self.num_head)
        K, V = [self.split_last_dim(tensor, self.num_head) for tensor in torch.split(memory, self.d_model, dim=2)]

        key_depth_per_head = self.d_model // self.num_head
        Q *= key_depth_per_head**-0.5
        x = self.dot_product_attention(Q, K, V, mask = mask)
        return self.combine_last_two_dim(x.permute(0,2,1,3)).transpose(1, 2)

    def dot_product_attention(self, q, k ,v, bias = False, mask = None):
        """dot-product attention.
        Args:
        q: a Tensor with shape [batch, heads, length_q, depth_k]
        k: a Tensor with shape [batch, heads, length_kv, depth_k]
        v: a Tensor with shape [batch, heads, length_kv, depth_v]
        bias: bias Tensor (see attention_bias())
        is_training: a bool of training
        scope: an optional string
        Returns:
        A Tensor.
        """
        logits = torch.matmul(q,k.permute(0,1,3,2))
        if bias:
            logits += self.bias
        if mask is not None:
            shapes = [x  if x != None else -1 for x in list(logits.size())]
            mask = mask.view(shapes[0], 1, 1, shapes[-1])
            logits = mask_logits(logits, mask)
        weights = F.softmax(logits, dim=-1)
        # dropping out the attention links for each of the heads
        weights = F.dropout(weights, p=self.drop_prob, training=self.training)
        return torch.matmul(weights, v)

    def split_last_dim(self, x, n):
        """Reshape x so that the last dimension becomes two dimensions.
        The first of these two dimensions is n.
        Args:
        x: a Tensor with shape [..., m]
        n: an integer.
        Returns:
        a Tensor with shape [..., n, m/n]
        """
        old_shape = list(x.size())
        last = old_shape[-1]
        new_shape = old_shape[:-1] + [n] + [last // n if last else None]
        ret = x.view(new_shape)
        return ret.permute(0, 2, 1, 3)

    def combine_last_two_dim(self, x):
        """Reshape x so that the last two dimension become one.
        Args:
        x: a Tensor with shape [..., a, b]
        Returns:
        a Tensor with shape [..., ab]
        """
        old_shape = list(x.size())
        a, b = old_shape[-2:]
        new_shape = old_shape[:-2] + [a * b if a and b else None]
        ret = x.contiguous().view(new_shape)
        return ret


