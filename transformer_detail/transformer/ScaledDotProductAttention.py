import torch
import torch.nn as nn
import torch.nn.functional as F
author = "尧志文"

class ScaledDotProductAttention(nn.Module):
    """
     实现缩放点积注意力机制。

    参数:
    - temperature: 温度参数，用于缩放点积注意力的计算。
    - attn_dropout: 注意力dropout的比例，默认为0.1。
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
       前向传播函数。

       参数:
       - q: 查询向量矩阵。
       - k: 关键向量矩阵。
       - v: 值向量矩阵。
       - mask: 可选，掩码矩阵，用于在注意力计算中屏蔽某些位置。

       返回:
       - output: 注意力机制后的输出。
       - attn: 注意力权重矩阵。
       """
        # 计算注意力权重
        attention = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # 如果有掩码，则将掩码位置的注意力权重置为负无穷大,这里设置为-1e9
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)

        # 应用softmax函数，得到注意力权重
        attention = self.dropout(F.softmax(attention, dim=-1))  # 注意力分布
        # 计算注意力机制后的输出
        output = torch.matmul(attention, v)  # 根据注意力分别加权求和

        return output, attention


