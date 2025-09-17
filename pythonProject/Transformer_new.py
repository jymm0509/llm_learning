import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
import random

# token embedding
class TokenEmbedding(nn.Embedding):
    def __init__(self, seq_len, d_model):
        super(TokenEmbedding, self).__init__(seq_len, d_model, padding_idx=1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        # encoding 矩阵
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad_(False)

        # 位置编码
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(1)

        # 偶数位置索引
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/d_model)))
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/d_model)))

    def forward(self, x):
        seq_len = x.size(1)
        return self.encoding[:seq_len, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob) # 随机丢弃神经元防止过拟合

    def forward(self, x):
        tok_emb = self.token_embedding(x)
        pos_emb = self.positional_encoding(x)
        return self.drop_out(tok_emb + pos_emb)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_combine = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # 1. 线性变换得到Q, K, V [batch_size, seq_len, d_model]
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # 分割 --- 相当于将Q K V各自投影成64 * d_k
        d_k = self.d_model // self.num_heads
        # view 改变张量的形状不改变数据
        # [bs, seq_len, d_model] --- [bs, seq_len, num_heads, d_k] --- [bs, num_heads, seq_len, d_k]
        Q = Q.contiguous().view(bs, -1, self.num_heads, d_k).transpose(1, 2)
        K = K.contiguous().view(bs, -1, self.num_heads, d_k).transpose(1, 2)
        V = V.contiguous().view(bs, -1, self.num_heads, d_k).transpose(1, 2)

        # 3. 计算注意力分数 --- [bs, num_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1) # [bs, seq_len, seq_len] --- [bs, 1, seq_len, seq_len]
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weight = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weight, V) # [bs, num_heads, seq_len, d_k]

        #合并
        attn_output = attn_output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.w_combine(attn_output)

        return output

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        # 缩放 偏移参数
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased = False, keepdim=True)
        out = (x - mean)/(var + self.eps) # 归一化
        out = self.gamma * out + self.beta # 缩放平移

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask = None):
        _x = x
        x_attn = self.self_attention(x,x,x,mask=mask)
        x_attn = self.dropout1(x_attn)
        x_attn = self.norm1(x_attn + _x) # 残差连接

        _x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        return x

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model,ffn_hidden, num_heads,n_layer,device, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = TokenEmbedding(enc_voc_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_len, device=device)
        self.layers = nn.ModuleList([
            TransformerEncoder(d_model, num_heads, ffn_hidden)
            for _ in range(n_layer)
        ])

    def forward(self, x, mask = None):
        x = self.embedding(x)
        x = self.position_encoder(x)
        for layer in self.layers:
            x = layer(x, mask)

        return x

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, ffn_hidden, num_heads, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.self_attention = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attention = nn.MultiheadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = LayerNorm(d_model)

    def forward(self, dec ,enc, self_attn_mask=None, cross_attn_mask=None):
        _x = dec
        x = self.self_attention(dec, dec, dec, self_attn_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        _x = x
        x = self.cross_attention(x, enc, enc, cross_attn_mask)
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)

        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, d_model,max_len, ffn_hidden, num_heads, n_layers, dropout, device):
        super(Decoder, self).__init__()
        self.embedding = TokenEmbedding(dec_voc_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, device=device)
        self.layers = nn.ModuleList([
            TransformerDecoder(d_model, ffn_hidden, num_heads)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden)

    def forward(self, dec, enc, self_attn_mask=None, cross_attn_mask=None):
        x = self.embedding(dec)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, enc, self_attn_mask, cross_attn_mask)

        decoder_output = self.ffn(x)
        return decoder_output

class Transformer(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 n_layer,
                 max_len,
                 ffn_hidden,
                 num_heads,
                 dropout,
                 device):
        super(Transformer, self).__init__()
        self.device = device
        self.encoder = Encoder(enc_voc_size, max_len, d_model,ffn_hidden, num_heads,n_layer,device)
        self.decoder = Decoder(dec_voc_size, d_model,max_len, ffn_hidden, num_heads, n_layer, dropout, device)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q = q.repeat(1, 1, 1, len_k)
        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)
        mask = q & k # [bs, 1, len_q, len_k] 注意力机制的时候是四维张量
        return mask

    def make_decoder_self_mask(self, q, k):
        len_q, len_k = q.size(1), k.size(1)
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor).to(self.device)
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg, trg, self.trg_pad_idx, self.trg_pad_idx)*self.make_decoder_self_mask(trg, trg)
        enc = self.encoder(src, src_mask)
        out = self.decoder(trg, enc, trg_mask, src_mask)
        return out


class ConversationDataset(Dataset):
    def __init__(self, pairs, vocab, max_len):
        self.pairs = pairs
        self.vocab = vocab
        self.max_len = max_len
        self.sos_token = vocab['<SOS>']
        self.eos_token = vocab['<EOS>']
        self.pad_token = vocab['<PAD>']

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        question, answer = self.pairs[idx]

        # 将文本转换为索引序列
        ques_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in question]
        ans_ids = [self.vocab.get(word, self.vocab['<UNK>']) for word in answer]

        # 添加特殊标记并截断/填充
        ques_ids = [self.sos_token] + ques_ids + [self.eos_token]
        ans_ids = [self.sos_token] + ans_ids + [self.eos_token]

        ques_ids = ques_ids[:self.max_len]
        ans_ids = ans_ids[:self.max_len]

        ques_ids = ques_ids + [self.pad_token] * (self.max_len - len(ques_ids))
        ans_ids = ans_ids + [self.pad_token] * (self.max_len - len(ans_ids))

        return torch.tensor(ques_ids, dtype=torch.long), torch.tensor(ans_ids, dtype=torch.long)


# 2. 定义训练函数
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch_idx, (src, trg) in enumerate(train_loader):
        src, trg = src.to(device), trg.to(device)

        # 解码器输入是trg[:-1]，目标是trg[1:]
        decoder_input = trg[:, :-1]
        target = trg[:, 1:].contiguous().view(-1)

        optimizer.zero_grad()

        # 前向传播
        output = model(src, decoder_input)
        output = output.view(-1, output.size(-1))

        # 计算损失
        loss = criterion(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


# 3. 定义评估函数
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in val_loader:
            src, trg = src.to(device), trg.to(device)
            decoder_input = trg[:, :-1]
            target = trg[:, 1:].contiguous().view(-1)

            output = model(src, decoder_input)
            output = output.view(-1, output.size(-1))

            loss = criterion(output, target)
            total_loss += loss.item()

    return total_loss / len(val_loader)


# 4. 定义生成回复的函数（用于测试）
def generate_response(model, input_text, vocab, inv_vocab, max_len, device):
    model.eval()
    with torch.no_grad():
        # 将输入文本转换为索引
        input_ids = [vocab.get(word, vocab['<UNK>']) for word in input_text]
        input_ids = [vocab['<SOS>']] + input_ids + [vocab['<EOS>']]
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        # 初始化解码器输入（只有SOS token）
        decoder_input = torch.tensor([[vocab['<SOS>']]], dtype=torch.long).to(device)

        # 逐步生成回复
        for i in range(max_len):
            output = model(input_tensor, decoder_input)
            next_word = output.argmax(dim=-1)[:, -1].item()

            if next_word == vocab['<EOS>']:
                break

            decoder_input = torch.cat([
                decoder_input,
                torch.tensor([[next_word]], dtype=torch.long).to(device)
            ], dim=1)

        # 将索引转换回文本
        response_ids = decoder_input[0].cpu().numpy()
        response_text = [inv_vocab[idx] for idx in response_ids if idx not in [vocab['<SOS>'], vocab['<PAD>']]]

        return ''.join(response_text)


# 5. 主训练流程
def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 超参数设置
    batch_size = 32
    max_len = 20
    d_model = 256
    n_layer = 3
    ffn_hidden = 512
    num_heads = 8
    dropout = 0.1
    learning_rate = 0.0005
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建简单的对话数据集[2,4](@ref)
    train_pairs = [
        ("你好", "你好！有什么我可以帮助你的吗？"),
        ("今天天气怎么样", "今天天气很好，阳光明媚。"),
        ("你会做什么", "我可以和你聊天，回答你的问题。"),
        ("你叫什么名字", "我是一个聊天机器人。"),
        ("再见", "再见！祝你有个愉快的一天！"),
        ("谢谢", "不客气，很高兴能帮助你。"),
        ("你好吗", "我很好，谢谢关心。你呢？"),
        ("今天星期几", "今天是星期五。"),
        ("你多大了", "我是一个程序，没有年龄。"),
        ("你喜欢什么", "我喜欢学习和帮助人们。")
    ]

    # 创建验证集
    val_pairs = [
        ("你好啊", "你好！很高兴见到你。"),
        ("天气如何", "今天天气不错，适合出门。"),
        ("你是谁", "我是一个智能聊天机器人。")
    ]

    # 创建词汇表[2,4](@ref)
    vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    for question, answer in train_pairs + val_pairs:
        for word in question:
            if word not in vocab:
                vocab[word] = len(vocab)
        for word in answer:
            if word not in vocab:
                vocab[word] = len(vocab)

    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    # 创建数据加载器
    train_dataset = ConversationDataset(train_pairs, vocab, max_len)
    val_dataset = ConversationDataset(val_pairs, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    src_pad_idx = vocab['<PAD>']
    trg_pad_idx = vocab['<PAD>']
    enc_voc_size = vocab_size
    dec_voc_size = vocab_size

    model = Transformer(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        d_model=d_model,
        n_layer=n_layer,
        max_len=max_len,
        ffn_hidden=ffn_hidden,
        num_heads=num_heads,
        dropout=dropout,
        device=device
    ).to(device)

    # 定义损失函数和优化器[1,6](@ref)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

    # 学习率调度器[7](@ref)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # 训练循环
    print("开始训练...")
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)

        scheduler.step()

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_transformer_model.pth')

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f}')

        # 每隔几个epoch测试生成效果
        if (epoch + 1) % 10 == 0:
            test_input = "你好"
            response = generate_response(model, test_input, vocab, inv_vocab, max_len, device)
            print(f'测试生成: 输入="{test_input}", 输出="{response}"')

    print("训练完成！")

    # 加载最佳模型并进行最终测试
    model.load_state_dict(torch.load('best_transformer_model.pth'))
    print("最终测试生成结果:")

    test_inputs = ["你好", "天气如何", "再见"]
    for test_input in test_inputs:
        response = generate_response(model, test_input, vocab, inv_vocab, max_len, device)
        print(f'输入: "{test_input}" -> 输出: "{response}"')


if __name__ == '__main__':
    main()



