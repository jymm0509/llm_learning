import torch
import torch.nn as nn
import torch.nn.functional as F

# # 1.单层单向
# single_rnn = nn.RNN(4, 3, 1, batch_first=True)
# input = torch.randn(1, 2, 4)
# output, h_n= single_rnn(input)
# print(output)
# print(h_n)
#
# # 2. 双向单层RNN
# bidirectional_rnn = nn.RNN(4, 3, 1, batch_first=True, bidirectional=True)
# bi_output, bi_h_n = bidirectional_rnn(input)
# print(bi_output)
# print(bi_h_n)

bs, T = 2, 3
input_size, hidden_size =2, 3
input = torch.randn(bs, T, input_size)
h_prev = torch.randn(bs, hidden_size)

#调用API
rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
rnn_output, h_n = rnn(input, h_prev.unsqueeze(0))

#矩阵乘法实现rnn_forward函数
# 单向的RNN
def rnn_forward(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev):
    bs, T, input_size = input.shape
    h_dim = weight_ih.shape[0]
    h_out = torch.zeros(bs, T, h_dim)

    for t in range(T):
        x = input[:, t, :].unsqueeze(2) #获取当前时刻输入,取得一个向量，扩充到三维与w相乘
        w_ih_batch = weight_ih.unsqueeze(0).tile(bs, 1, 1) #bs * h_dim * input_size
        w_hh_batch = weight_hh.unsqueeze(0).tile(bs, 1, 1) #bs * h_dim * h_dim
        w_time_x = torch.bmm(w_ih_batch, x).squeeze(-1) #去掉最后一维，bs*h_dim
        w_time_h = torch.bmm(w_hh_batch, h_prev.unsqueeze(2)).squeeze(-1)
        h_prev = torch.tanh(w_time_x + bias_hh + bias_ih + w_time_h)
        h_out[:, t, :] = h_prev

        return h_out, h_prev.unsqueeze(0)













