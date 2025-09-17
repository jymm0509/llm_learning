import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
nn.Transformer
class Seq2SeqEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, source_vocab_size):
        super(Seq2SeqEncoder, self).__init__()
        #示例化两个对象，lstm层、embedding_table
        self.lstm_layer = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.embedding_table = nn.Embedding(source_vocab_size, embedding_dim)

    def forward(self, input_ids): # input_ids是将sentence中的单词用位置index表示后的序列
        input_sequence = self.embedding_table(input_ids) #input --- (bs, source_len, embedding_dim)
        output_states, (final_h, final_c) = self.lstm_layer(input_sequence) # lstm的输入序列维度是(bs, source_len, input_size)
        return output_states, final_h

class Seq2SeqAttentionMechanism(nn.Module):
    def __init__(self):
        super(Seq2SeqAttentionMechanism, self).__init__()

    def forward(self, decoder_state_t, encoder_states): #encoder_states --- (bs, source_len, hidden_dim)
        bs, source_len, hidden_size = encoder_states.shape
        decoder_state_t = decoder_state_t.unsqueeze(1) #(bs, hidden_dim) --- (bs,, source_len, hidden_dim)
        decoder_state_t = torch.tile(decoder_state_t, (1, source_len, 1))
        score = torch.sum(decoder_state_t*encoder_states, dim=-1) #内积操作 --- (bs, source_length)
        atte_prob = F.softmax(score, dim=-1) # (bs, source_len)
        context = torch.sum(atte_prob.unsqueeze(-1) * encoder_states, 1) # (bs, hidden_size) 求加权和
        return atte_prob, context


class Seq2SeqDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_classes, target_vocab_size, start_id, end_id):
        super(Seq2SeqDecoder, self).__init__()
        self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
        self.proj_layer = nn.Linear(hidden_size*2, num_classes) #lstm的输出，要映射到分类的分布上，其输入是[context; h_t] == 2*hidden_size
        self.attention = Seq2SeqAttentionMechanism()
        self.num_classes = num_classes # 类别数目
        self.embedding_table = nn.Embedding(target_vocab_size, embedding_dim)
        self.start_id = start_id
        self.end_id = end_id # 结束符号

    #训练阶段调用，teacher forcing，基于正确的答案进行训练
    def forward(self, shift_target_ids, encoder_states): #shift_target_ids 第一个是start_id
        shift_target_ids = self.embedding_table(shift_target_ids)
        bs, target_len, embedding_dim = shift_target_ids.shape
        bs, source_len, hidden_size = encoder_states.shape

        logits = torch.zeros(bs, target_len, self.num_classes)
        probs = torch.zeros(bs, target_len, source_len) #每一个target都有一个对source的probs

        for t in range(target_len): #串行计算
            decoder_input = shift_target_ids[:, t, :]
            if t == 0: # 取到的是start_id的embedding
                h_t, c_t = self.lstm_cell(decoder_input)
            else:
                h_t, c_t = self.lstm_cell(decoder_input, (h_t, c_t))

            attention_prob, context = self.attention(h_t, encoder_states)

            decoder_output = torch.cat((context, h_t), dim=-1)
            logits[:, t, :] = self.proj_layer(decoder_output)
            probs[:, t, :] = attention_prob

        return probs, logits

    #推理
    def inference(self, encoder_states):
        target_id = self.start_id
        h_t = None
        result = []

        while True:
            decoder_input_t = self.embedding_table(target_id)
            if h_t is None:
                h_t, c_t = self.lstm_cell(decoder_input_t)
            else:
                h_t, c_t = self.lstm_cell(decoder_input_t, (h_t, c_t))

            attention_prob, context = self.attention(h_t, encoder_states)

            decoder_output = torch.cat((context,h_t), dim=-1)
            logits = self.proj_layer(decoder_output)
            target_id = torch.argmax(logits, dim=-1)
            result.append(target_id)

            if torch.any(target_id == self.end_id):
                print("stop decoding.......")
                break
        predicted_ids = torch.stack(result, dim=0)
        return predicted_ids

class Model(nn.Module):
    def __init__(self ,embedding_dim, hidden_size, num_classes, source_vocab_size, target_vocab_size, start_id, end_id):
        super(Model, self).__init__()
        self.encoder = Seq2SeqEncoder(embedding_dim, hidden_size, source_vocab_size)
        self.decoder = Seq2SeqDecoder(embedding_dim, hidden_size, num_classes, target_vocab_size, start_id, end_id)

    def forward(self, input_sequence_ids, shifted_target_sequence_ids):
        # 训练
        encoder_states, final_h = self.encoder(input_sequence_ids)
        probs, logits = self.decoder(shifted_target_sequence_ids, encoder_states)
        return probs, logits
    def infer(self, input_sequence_ids):
        encoder_states, final_h = self.encoder(input_sequence_ids)
        predicted_ids = self.decoder.inference(encoder_states)
        return predicted_ids



if __name__ == '__main__':
    # 尝试设置后端 (添加到代码开头，在导入plt之前)
    try:
        import matplotlib
        matplotlib.use('TkAgg')  # 或 'Qt5Agg'
        print(f"Using matplotlib backend: {matplotlib.get_backend()}")
    except Exception as e:
        print(f"Error setting backend: {e}")

    import matplotlib.pyplot as plt
    source_vocab_size = 100
    target_vocab_size = 100
    num_classes = target_vocab_size  # 确保与目标词汇表大小一致
    embedding_dim = 8
    hidden_size = 16
    source_len = 3  # 输入序列长度
    target_len = 4  # 目标序列基础长度（不包括EOS）
    batch_size = 2
    num_epochs = 1000
    learning_rate = 0.001

    # 特殊标记
    start_id = 0  # SOS标记，假设为0
    end_id = 99  # EOS标记，假设为99（确保在词汇表范围内）

    # 初始化模型
    model = Model(embedding_dim, hidden_size, num_classes, source_vocab_size, target_vocab_size, start_id, end_id)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 忽略索引处理可选
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 初始化绘图
    plt.ion()  # 开启交互模式
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Over Time')
    ax.grid(True)  # 添加网格
    # 初始化一个空列表和一条空线
    epochs_list = []
    losses_list = []
    loss_line, = ax.plot([], [], 'b-', label='Training Loss')
    ax.legend()

    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0.0

        # 生成随机训练数据（示例数据）
        input_sequence_ids = torch.randint(0, source_vocab_size, (batch_size, source_len))
        target_ids = torch.randint(0, target_vocab_size, (batch_size, target_len))

        # 添加EOS标记到目标序列末尾
        target_ids_with_eos = torch.cat((target_ids, end_id * torch.ones(batch_size, 1, dtype=torch.long)), dim=-1)

        # 创建解码器输入：移位目标序列（以SOS开头）
        shifted_target_sequence_ids = torch.cat(
            (start_id * torch.ones(batch_size, 1, dtype=torch.long), target_ids_with_eos[:, :-1]), dim=-1)

        # 前向传播
        probs, logits = model(input_sequence_ids, shifted_target_sequence_ids)

        # 计算损失：logits形状(batch_size, target_len+1, num_classes)，目标形状(batch_size, target_len+1)
        epoch_loss = criterion(logits.view(-1, num_classes), target_ids_with_eos.view(-1))

        # 记录损失
        current_loss = epoch_loss.item()
        losses_list.append(current_loss)
        epochs_list.append(epoch + 1)

        # --- 更新图表 (修改部分) ---
        loss_line.set_data(epochs_list, losses_list)
        ax.relim()  # 重设坐标轴范围
        ax.autoscale_view()  # 自动缩放

        # 短暂暂停并更新图表，但避免过于频繁的更新
        if epoch % 1 == 0:  # 每个epoch都更新，如果慢可以改为每N个epoch
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.01)  # 短暂暂停

        # 反向传播和优化
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()

        total_loss += epoch_loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss.item():.4f}')

    print("训练完成！")
    plt.ioff()  # 训练结束后关闭交互模式
    plt.show()  # 显示最终的图表