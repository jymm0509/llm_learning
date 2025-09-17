import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 官方API
bs, T, i_size, h_size = 2, 3, 4, 5
input = torch.randn(bs, T, i_size) #输入序列 (2, 3, 4)
c0 = torch.randn(bs, h_size) #初始值
h0 = torch.randn(bs, h_size) #初始值
lstm_layer = nn.LSTM(input_size=i_size, hidden_size=h_size, batch_first=True)
output1, h_n = lstm_layer(input, (h0.unsqueeze(0), c0.unsqueeze(0)))
print(output1)

# for k, v in lstm_layer.named_parameters():
#     print(k, v.shape)

#input_size , output_size 指的是输入输出的特征维度，T代表的是每个batch的样本个数（每一时刻的样本），代表Time
def lstm_forward(input, initial_states, w_ih, w_hh, b_ih, b_hh):
    h0, c0 = initial_states #初始状态
    bs, T, i_size = input.shape
    h_size = w_ih.shape[0]

    h_prev = h0
    c_prev = c0

    # 对权重进行扩维度
    batch_w_ih = w_ih.unsqueeze(0).tile(bs, 1, 1)  # (4*h_size, i_size) ---> (bs, 4*h_size, i_size)
    batch_w_hh = w_hh.unsqueeze(0).tile(bs, 1, 1)

    output_size = h_size
    output = torch.zeros(bs, T, output_size) # 初识化输出序列

    for t in range(T):
        x = input[:, t, :] #当前时刻的输入向量(bs, i_size)
        #对x记性扩维度
        w_time_x = torch.bmm(batch_w_ih, x.unsqueeze(-1)) # (bs, 4*h_size, i_size) * (bs, i_size, 1) = (bs, 4*h_size, 1)
        w_time_x = w_time_x.squeeze(-1) # 去掉最后一维度

        w_time_h_prev = torch.bmm(batch_w_hh, h_prev.unsqueeze(-1))
        w_time_h_prev = w_time_h_prev.squeeze(-1)

        # 内部计算
        i_t = torch.sigmoid(w_time_h_prev[:, :h_size] + w_time_x[:, :h_size] \
                            + b_hh[:h_size] + b_ih[:h_size])
        f_t = torch.sigmoid(w_time_h_prev[:, h_size:2*h_size] + w_time_x[:, h_size:2*h_size] \
                            + b_hh[h_size:2*h_size] + b_ih[h_size:2*h_size])
        g_t = torch.tanh(w_time_h_prev[:, 2*h_size:3*h_size] + w_time_x[:, 2*h_size:3*h_size] \
                         + b_hh[2*h_size:3*h_size] + b_ih[2*h_size:3*h_size])
        o_t = torch.sigmoid(w_time_h_prev[:, 3*h_size:4*h_size] + w_time_x[:, 3*h_size:4*h_size] + b_hh[3*h_size:4*h_size] + b_ih[3*h_size:4*h_size])

        c_prev = f_t * c_prev + i_t * g_t
        h_prev = o_t * torch.tanh(c_prev)

        output[:, t, :] = h_prev

    return output, (h_prev, c_prev)


output2 = lstm_forward(input, (h0, c0), lstm_layer.weight_ih_l0, lstm_layer.weight_hh_l0, lstm_layer.bias_ih_l0, lstm_layer.bias_hh_l0)

