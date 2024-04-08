import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ScaledDotProductAttention import ScaledDotProductAttention
author = "尧志文"

class MulitheadAttention(nn.Module):
    """
    多头注意力层的实现。

    参数:
    - n_head: int, 注意力头的数量。
    - d_model: int, 模型的维度。
    - d_k: int, 关键向量的维度。
    - d_v: int, 值向量的维度。
    - dropout: float, Dropout比例，默认为0.1。
    """
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.fc = nn.Linear(n_head * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        前向传播函数。

        参数:
        - q: 查询向量的张量。
        - k: 关键向量的张量。
        - v: 值向量的张量。
        - mask: 可选，掩码张量，默认为None。

        返回:
        - q: 加工后的查询向量。
        - attention: 注意力权重分布。
        """
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        batch_size, q_len, k_len, v_len = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # 对查询、关键和值向量进行线性变换，并重新排列为多头形式
        q = self.w_qs(q).view(batch_size, q_len, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(batch_size, k_len, n_head, d_k).transpose(1, 2)
        v = self.w_ks(v).view(batch_size, v_len, n_head, d_v).transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, n_head, -1, -1)
        # 计算注意力权重并应用到值向量上
        q, attention = self.attention(q, k, v, mask=mask)
        q = q.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        # 通过全连接层、dropout和层归一化恢复向量维度
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attention

class PositionwiseFeedForward(nn.Module):
    """
    位置wise的前馈神经网络层。

    参数:
    - d_in: int, 输入的维度。
    - d_hid: int, 隐藏层的维度。
    - dropout: float, Dropout比例，默认为0.1。
    """
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x: 输入的张量。

        返回:
        - x: 加工后的张量。
        """
        residual = x
        # 通过两层线性变换和ReLU激活函数处理输入
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x

        









