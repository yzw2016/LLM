import torch
from torch.nn import ModuleList, Linear, LayerNorm
import torch.nn as nn

# Transformer解码器层类
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, vocab_size, dim_feedforward=2048, dropout=0.1, activation="relu"):
        """
        初始化Transformer解码器层。
        
        参数:
        - d_model: 模型的维度
        - nhead: 多头注意力的头数
        - dim_feedforward: 前馈神经网络的维度，默认为2048
        - dropout: Dropout比例，默认为0.1
        - activation: 激活函数名称，默认为"relu"
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        前向传播函数。
        
        参数:
        - tgt: 目标序列
        - memory: 记忆序列（来自编码器的输出）
        - tgt_mask: 目标序列的掩码，用于在注意力机制中忽略某些位置
        - memory_mask: 记忆序列的掩码
        - tgt_key_padding_mask: 目标序列的键填充掩码，用于忽略填充的位置
        - memory_key_padding_mask: 记忆序列的键填充掩码
        
        返回值:
        - 加工后的目标序列
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_activation_fn(activation):
    """
    根据字符串名称返回激活函数。
    
    参数:
    - activation: 激活函数的名称
    
    返回值:
    - 对应的激活函数
    
    异常:
    - 如果给定的名称不支持，则抛出ValueError
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "tanh":
        return nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {activation}")


# Transformer解码器类
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        vocab_size,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
    ):
        """
        初始化Transformer解码器。
        
        参数:
        - num_layers: 解码器层数
        - d_model: 模型的维度
        - nhead: 多头注意力的头数
        - dim_feedforward: 前馈神经网络的维度，默认为2048
        - dropout: Dropout比例，默认为0.1
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.linear = Linear(d_model, vocab_size)
        self.layers = ModuleList([TransformerDecoderLayer(d_model, nhead, vocab_size, dim_feedforward, dropout, activation) for _ in range(num_layers)])
        self.norm = LayerNorm(d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tgt, memory, **kwargs):
        """
        前向传播函数。
        
        参数:
        - tgt: 目标序列
        - memory: 记忆序列（来自编码器的输出）
        - kwargs: 其他参数
        
        返回值:
        - 返回最后一步预测概率分布
        """
        tgt = self.embed(tgt).transpose(0, 1)    #词嵌入层，只是为了演示推理过程的bean search,所以用词嵌入作为doecoder输入
        for layer in self.layers:
            tgt = layer(tgt, memory)
        tgt = tgt.transpose(0, 1)
        tgt = self.linear(tgt)            
        decoded_sequence = self.softmax(tgt[:, -1, :])   
        return decoded_sequence
    
# Greedy搜索解码类
class GreedySearch:
    def __init__(self, memory_tensor, max_len = 5):
        """
        初始化Greedy搜索解码器。
        
        参数:
        - memory_tensor: 编码后的记忆张量，用于解码过程中的输入。
        - max_len: 最大解码长度，默认为5。
        """
        self.decoder = TransformerDecoder(num_layers=6, d_model=512, nhead=8, vocab_size = 1024)
        self.max_len = max_len
        self.memory_tensor = memory_tensor
        
    def decode(self, start_id, end_id):
        """
        使用贪婪搜索进行解码。
        
        参数:
        - start_id: 起始标识符，解码序列的起始字符。
        - end_id: 结束标识符，解码序列终止的条件。
        
        返回值:
        - decoded_sequence: 解码得到的序列。
        """
        action_seq = torch.tensor([[start_id]])  # 初始化解码序列
        time_step = 0
        while time_step < self.max_len:  # 控制解码步数，防止过长
            logits = self.decoder(action_seq, self.memory_tensor)  # 通过解码器获取概率分布
            logits = logits.squeeze(0)  # 调整logits形状以进行下一步操作
            if torch.argmax(logits) == end_id:  # 如果预测到结束标识符，则终止解码
                break
            action_seq = torch.cat((action_seq, torch.argmax(logits).unsqueeze(0).unsqueeze(0)), dim=-1)  # 根据概率分布选择最大概率的字符并添加到解码序列中
            time_step += 1
        decoded_sequence = action_seq.squeeze().tolist()  # 处理解码序列，转换为列表形式
        return decoded_sequence

# Beam搜索解码类
class BeamSearch:
    def __init__(self, memory_tensor, beam_width=2, max_len=5):
        """
        初始化Beam搜索解码器。
        
        参数:
        - beam_width: Beam的宽度，即同时保留的最有希望的解码序列的数量，默认为2。
        - max_len: 最大解码长度，即终止解码前允许的最大序列长度，默认为5。
        - memory_tensor: 缓存张量，用于存储解码过程中的中间状态。
        """
        self.beam_width = beam_width
        self.max_len = max_len
        self.memory_tensor = memory_tensor
        self.decoder = TransformerDecoder(num_layers=6, d_model=512, nhead=8, vocab_size = 1024)  # 初始化解码器模型
            
    def decode(self, start_id, end_id):
        """
        使用Beam搜索进行解码。
        
        参数:
        - start_id: 起始标记的ID，用于序列的初始化。
        - end_id: 结束标记的ID，用于标识解码序列的结束。
        
        返回值:
        - 解码的序列列表: 返回得分最高的解码序列。
        """
        # 初始化活跃序列
        active_sequences = [(torch.tensor([[start_id]]), torch.log(torch.tensor([1.0])), False)]
        time_step = 0        
        while len(active_sequences) > 0 and time_step < self.max_len:
            new_sequences = []
            for previous_sequence, previous_score , finished in active_sequences:
                if finished:  # 如果序列已经结束，则跳过
                    new_sequences.append((previous_sequence, previous_score, True))
                    continue
                # 获取当前时间步的输入序列
                input_sequence = previous_sequence
                # 通过解码器获取下一个时间步的概率分布
                next_logits = self.decoder(input_sequence, self.memory_tensor)
                next_probs = next_logits
                # 在概率分布上进行Beam搜索
                topk_probs, topk_ids = torch.topk(next_probs, k=self.beam_width, dim=-1)  
                for id in topk_ids.squeeze().tolist():
                    if id == end_id:  # 如果生成了结束标记，则结束该序列
                        new_sequences.append((torch.cat([previous_sequence, torch.tensor([[id]])], dim=-1), previous_score + torch.log(torch.Tensor([next_probs.squeeze().tolist()[id]])), True))
                    else:  # 继续生成序列
                        new_sequences.append((torch.cat([previous_sequence, torch.tensor([[id]])], dim=-1), previous_score + torch.log(torch.Tensor([next_probs.squeeze().tolist()[id]])), False))
            # 更新活跃序列为新的序列列表
            active_sequences = new_sequences
            time_step += 1
        
        # 对所有完成的序列按得分进行排序，返回得分最高的序列
        completed_sequences = sorted(active_sequences, key=lambda x: x[1], reverse=True)
        return completed_sequences[0][0].squeeze().tolist()
            
# 以下代码演示了如何使用两种不同的搜索策略：贪心搜索（Greedy Search）和束搜索（Beam Search），
# 来解码来自一个记忆张量（memory_tensor）的信息。

# 示例用法:
memory_tensor = torch.randn(10, 1, 512)  # 从编码器获取的记忆张量，维度为(批大小, 序列长度, 特征维度)

# 使用贪心搜索策略
greedy_search = GreedySearch(memory_tensor)
print(greedy_search.decode(0, 1))  # 对记忆张量进行解码，输出结果
print("*********************")

# 使用束搜索策略
bean_search = BeamSearch(memory_tensor)
print(bean_search.decode(0, 1))  # 同样对记忆张量进行解码，但使用不同的搜索策略
