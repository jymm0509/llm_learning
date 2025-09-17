import torch
import torch.nn as nn
from d2l import torch as d2l
import os
import random


# 将两个句子变成BERT的输入
def get_tokens_and_segmemts(tokens_a, tokens_b = None):
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segmments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segmments += [1] * (len(tokens_b) + 1)

    return tokens, segmments

class BERTEndocer(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len = 1000, key_size = 768, query_size = 768, value_size = 768,
                 **kwargs):
        super(BERTEndocer, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f'{i}', d2l.TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout,
                True))
        #可以学习的位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))

    def forward(self, tokens, segmments, valid_lens):
        X = self.token_embedding(tokens) + self.segment_embedding(segmments)
        X = X + self.pos_embedding.data[:, :X.shape[1]]
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X

# vocab_size, num_hiddens, ffn_num_hiddens, num_heads = 10000, 786, 2024, 4
# norm_shape, ffn_num_input, num_layers, dropout = [786], 786, 2, 0.2
# encoder = BERTEndocer(vocab_size, num_hiddens, norm_shape, ffn_num_input,
#                  ffn_num_hiddens, num_heads, num_layers, dropout)
# tokens = torch.randint(0, vocab_size, (2, 8))
# segments = torch.tensor([[0,0,0,0,1,1,1,1], [0,0,0,1,1,1,1,1]])
# encoder_X = encoder(tokens, segments, None)
# print(encoder_X)
# print(encoder(tokens, segments))

class MaskLM(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_inputs=768, **kwargs):
        super(MaskLM, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )
    # 将pred_position出输出的特征抽出来进行预测
    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat


class NextSentencePred(nn.Module):
    def __init__(self, num_inputs, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        return self.output(X)

class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len = 1000, key_size = 768, query_size = 768, value_size = 768,
                 hid_in_features=786, mlm_in_features = 768, nsp_in_feature = 768):
        super(BERTModel, self).__init__()
        self.encoder = BERTEndocer(vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len, key_size, query_size, value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)
        self.nsp = NextSentencePred(nsp_in_feature)

    def forward(self, tokens, segmments, valid_lens = None,
                pred_positions = None):
        encoded_X = self.encoder(tokens, segmments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None

        nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        return encoded_X, mlm_Y_hat, nsp_Y_hat
