import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

# 1. 随机数据设置
#序列建模
batch_size = 2
#单词表的数目
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8 #特征大小,为什么512最好？？？
max_position_len = 5

#最大序列长度
max_src_seq_len = 5
max_tgt_seq_len = 5

src_len = torch.randint(2, max_src_seq_len, (batch_size,))
trg_len = torch.randint(2, max_tgt_seq_len, (batch_size,))

#生成随机序列，由单词索引构成
# pad --- 填充到最大长度，默认填充0（因为输入长度要一致）
# cat --- 转换成tensor
# 先用unsqueeze将数据维度变成[1,5]，然后用cat进行拼接
src_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_src_words, (L,)), (0, max_src_seq_len-L)), 0)\
                    for L in src_len])
tgt_seq = torch.cat([torch.unsqueeze(F.pad(torch.randint(1, max_num_tgt_words, (L,)), (0, max_tgt_seq_len-L)), 0)\
                     for L in trg_len])


# 2. word embedding
# table 的第0行是给padding的，第1-9行给8个字母的
src_embedding_table = nn.Embedding(max_num_src_words+1, model_dim)
trg_embedding_table = nn.Embedding(max_num_tgt_words+1, model_dim)

# print(src_embedding_table.weight)
# print(src_seq)
# print(src_embedding_table(src_seq))

# 3. position embedding
pos_mat = torch.arange(max_position_len).reshape((-1, 1))
i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1, -1))/model_dim)
pos_embedding_table = torch.zeros(max_position_len, model_dim)
pos_embedding_table[:, 0::2] = torch.sin(pos_mat/i_mat)
pos_embedding_table[:, 1::2] = torch.cos(pos_mat/i_mat)

#直接加上位置信息
# print(src_embedding_table(src_seq)+pos_embedding_table)

# #  softmax演示————为什么要除以dk
# score = torch.randn(5) # 模拟QK'
# a = 0.1 #模拟除以d
# b = 10 # 模拟不处理
# prob1 = F.softmax(score*a, dim=-1)
# prob2 = F.softmax(score*b, dim=-1)
# print(score)
# print(prob1)
# print(prob2) # 明显可见prob2的值相差很大，因为softmax是非线性的，大的会越来越大

# 4. encoder Self-Attention的mask
valid_encoder_pos = torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max_src_seq_len - L)), 0) for L in src_len])
valid_encoder_pos = torch.unsqueeze(valid_encoder_pos, 2)
valid_encoder_pos_matrix = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1,2))
invalid_encoder_pos_matrix = 1 - valid_encoder_pos_matrix
mask_encoder_self_attention = invalid_encoder_pos_matrix.to(torch.bool)

score = torch.randn(batch_size, max_src_seq_len, max_src_seq_len)
mask_score = score.masked_fill(mask_encoder_self_attention, -1e9)
prob = F.softmax(mask_score, dim=-1)

# print(prob)

# 5. Step5: 构造intra-attention的mask
valid_encoder_pos = torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max_src_seq_len - L)), 0) for L in src_len])
valid_decoder_pos = torch.cat([torch.unsqueeze(F.pad(torch.ones(L), (0, max_tgt_seq_len - L)), 0) for L in trg_len])
valid_encoder_pos = torch.unsqueeze(valid_encoder_pos, 2)
valid_decoder_pos = torch.unsqueeze(valid_decoder_pos, 2)
valid_cross_pos_matrix = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(1, 2))
invalid_cross_pos_matrix = 1 - valid_cross_pos_matrix
valid_encoder_self_attention = invalid_cross_pos_matrix.to(torch.bool)


# print(valid_encoder_pos)
# print(valid_decoder_pos)


# 6. Step6: 构造decoder self-attention的mask
valid_decoder_tri_matrix = torch.cat([torch.unsqueeze(F.pad(torch.tril(torch.ones(L, L)), (0, max_tgt_seq_len-L, 0, max_tgt_seq_len -L)), 0)for L in trg_len])
invalid_decoder_tri_matrix = 1 - valid_decoder_tri_matrix
invalid_decoder_tri_matrix = invalid_decoder_tri_matrix.to(torch.bool)

score = torch.randn(batch_size, max_tgt_seq_len, max_src_seq_len)
score_mask = score.masked_fill(invalid_decoder_tri_matrix, -1e9)
prob = F.softmax(score_mask, -1)
print(prob)

# Step7: scaled self-attention
def scaled_dot_product_attention(q, k, v, mask=None):
    score = torch.bmm(q, k.transpose(-2, -1))/torch.sqrt(model_dim)
    masked_score = score.masked_fill(mask, -1e9)
    F.softmax(masked_score)
    context = torch.bmm(prob, v)
    return context


