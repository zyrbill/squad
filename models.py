"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import qanet_layers
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob, 
                                    use_char_emb=use_char_emb)        

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
        print('BiDAF Attention', print(att.size()))

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class QANet(nn.Module):
    def __init__(self, word_vectors, char_vectors, hidden_size, drop_prob=0.1, num_head=8):  # !!! notice: set it to be a config parameter later.
        super().__init__()
        self.emb = qanet_layers.Embedding(word_vectors, char_vectors, hidden_size, drop_prob=drop_prob)
        self.num_head = num_head
        self.emb_enc = qanet_layers.EncoderBlock(conv_num=4, hidden_size=hidden_size, num_head=num_head, kernel_size=7, drop_prob=0.1)
        self.cq_att = layers.BiDAFAttention(hidden_size=hidden_size, drop_prob=drop_prob)
        #self.cq_att = qanet_layers.CQAttention(hidden_size=hidden_size)
        #self.cq_resizer = nn.Linear(hidden_size*4, hidden_size)
        self.cq_resizer = qanet_layers.Initialized_Conv1d(hidden_size * 4, hidden_size)
        self.model_enc_blks = nn.ModuleList([qanet_layers.EncoderBlock(conv_num=2, hidden_size=hidden_size, num_head=num_head, kernel_size=5, drop_prob=0.1) for _ in range(5)])
        self.out = qanet_layers.Output(hidden_size)
        self.hidden_size = hidden_size
        self.drop_prob = drop_prob

    def forward(self, cw_idxs, qw_idxs, cc_idxs, qc_idxs):
        c_mask = (torch.zeros_like(cw_idxs) != cw_idxs).float()  #  different
        q_mask = (torch.zeros_like(qw_idxs) != qw_idxs).float()
        
        c_emb, q_emb = self.emb(cw_idxs, cc_idxs), self.emb(qw_idxs, qc_idxs)

        c_enc = self.emb_enc(c_emb, c_mask)
        q_enc = self.emb_enc(q_emb, q_mask)
        x = self.cq_att(c_enc, q_enc, c_mask, q_mask)
        M0 = self.cq_resizer(x.transpose(1,2)).transpose(1,2)    # not the same dim as bang liu
        M0 = F.dropout(M0, p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, c_mask)
        M1 = M0
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, c_mask)
        M2 = M0
        M0 = F.dropout(M0, p=self.drop_prob, training=self.training)
        for i, blk in enumerate(self.model_enc_blks):
             M0 = blk(M0, c_mask)
        M3 = M0
        p1, p2 = self.out(M1, M2, M3, c_mask)
        return p1, p2