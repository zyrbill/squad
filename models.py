"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import qanet_layers
import unified_layers
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0., use_char_emb=False):
        super(BiDAF, self).__init__()
        if use_char_emb:
            self.emb = qanet_layers.Embedding(word_vectors=word_vectors, 
                                              char_vectors=char_vectors, 
                                              hidden_size=hidden_size, 
                                              drop_prob=drop_prob)
        else:
            self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs, cc_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs, qc_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)
        #print('BiDAF Attention', print(att.size()))

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1, num_head=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.num_head = num_head
        
        self.emb = qanet_layers.Embedding(word_vectors, char_vectors, hidden_size, drop_prob=drop_prob)
        self.emb_enc = qanet_layers.EncoderBlock(conv_num=4, hidden_size=hidden_size, num_head=num_head, kernel_size=7, drop_prob=0.1)
        self.cq_att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)
        self.proj = qanet_layers.FeedForward(hidden_size * 4, hidden_size)
        #self.cq_resizer = qanet_layers.FeedForward(hidden_size * 4, hidden_size)
        self.model_enc_blks = nn.ModuleList([qanet_layers.EncoderBlock(conv_num=2, hidden_size=hidden_size, num_head=num_head, kernel_size=5, drop_prob=0.1) for _ in range(5)])
        self.out = qanet_layers.Output(hidden_size)
        #self.out = qanet_layers.CondOutput(hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float()
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()
        
        c_emb, q_emb = self.emb(cw_idxs, cc_idxs), self.emb(qw_idxs, qc_idxs)
        c_enc, q_enc = self.emb_enc(c_emb, c_mask), self.emb_enc(q_emb, q_mask)
        
        x = self.cq_att(c_enc, q_enc, c_mask, q_mask)
        m0 = self.proj(x)
        #m0 = self.cq_resizer(x)

        m0 = F.dropout(m0, p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             m0 = blk(m0, c_mask)
        m1 = m0

        #m0 = F.dropout(m0, p=self.drop_prob, training=self.training)   ##  do we need to add it?
        for i, blk in enumerate(self.model_enc_blks):
             m0 = blk(m0, c_mask)
        m2 = m0

        m0 = F.dropout(m0, p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             m0 = blk(m0, c_mask)
        m3 = m0
        p1, p2 = self.out(m1, m2, m3, c_mask)
        return p1, p2


class UnifiedQANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1, num_head=8, num_emb_encoder=1, num_mdl_encoder=5, use_verifier=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob
        self.num_head = num_head
        self.use_verifier = use_verifier
        
        self.emb = unified_layers.Embedding(word_vectors, char_vectors, hidden_size, drop_prob=drop_prob)
        #encoder_blk = unified_layers.TransformerEncoderBlock(hidden_size=hidden_size, num_head=num_head, drop_prob=drop_prob)
        #self.model_enc_blks = nn.ModuleList([copy.deepcopy(encoder_blk) for _ in range(4)])
        self.emb_enc_blks = nn.ModuleList([qanet_layers.EncoderBlock(conv_num=4, hidden_size=hidden_size, num_head=num_head, kernel_size=7, drop_prob=0.1) for _ in range(num_emb_encoder)])
        self.cq_att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)
        self.proj = qanet_layers.FeedForward(hidden_size * 4, hidden_size)
        self.model_enc_blks = nn.ModuleList([qanet_layers.EncoderBlock(conv_num=2, hidden_size=hidden_size, num_head=num_head, kernel_size=5, drop_prob=0.1) for _ in range(num_mdl_encoder)])
        if self.use_verifier:
            self.out = unified_layers.VerifierOutput(hidden_size)
        else:
            self.out = qanet_layers.Output(hidden_size)

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float()
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()

        c_len, q_len = cw_idxs.shape[1], qw_idxs.shape[1]

        mask = torch.cat([c_mask, q_mask], dim=-1)
        
        enc = self.emb(cw_idxs, qw_idxs, cc_idxs, qc_idxs)
        for i, blk in enumerate(self.emb_enc_blks):
             enc = blk(enc, mask)
        
        c_enc, q_enc = enc[:,:c_len,:], enc[:,c_len:,:] 

        m0 = self.cq_att(c_enc, q_enc, c_mask, q_mask)
        m0 = self.proj(m0)

        m0 = F.dropout(m0, p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             m0 = blk(m0, c_mask)
        m1 = m0

        #m0 = F.dropout(m0, p=self.drop_prob, training=self.training)   ##  do we need to add it?
        for i, blk in enumerate(self.model_enc_blks):
             m0 = blk(m0, c_mask)
        m2 = m0

        m0 = F.dropout(m0, p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             m0 = blk(m0, c_mask)
        m3 = m0

        if self.use_verifier:
            p1, p2, pna = self.out(m1, m2, m3, c_mask)
            return p1, p2, pna
        else:
            p1, p2 = self.out(m1, m2, m3, c_mask)
            return p1, p2
