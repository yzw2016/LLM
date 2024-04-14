import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ScaledDotProductAttenion import ScaledDotProductAttention
author = 'yaozhiwen'
class MulitheadAttention(nn.Layer):
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
        super(MulitheadAttention, self).__init__()
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
        batch_size, q_len, k_len, v_len = q.shape[0], q.shape[1], k.shape[1], v.shape[1]
        resdiual = q
        q = self.w_qs(q).view([batch_size, q_len, n_head, d_k]).transpose([0, 2, 1, 3])
        k = self.w_ks(k).view([batch_size, k_len, n_head, d_k]).transpose([0, 2, 1, 3])
        v = self.w_vs(v).view([batch_size, v_len, n_head, d_v]).transpose([0, 2, 1, 3])
        if mask is not None:
            mask = mask.unsqueeze(1)
        q, attention = self.attention(q, k, v, mask=mask)
        q = q.transpose([0, 2, 1, 3]).contiguous().reshape([batch_size, -1, n_head * d_v])
        q = self.dropout(self.fc(q))
        q = self.layer_norm(q + resdiual)
        return q, attention
    
class PositionwiseFeedForward(nn.Layer):
    """
    位置wise的前馈神经网络层。

    参数:
    - d_in: int, 输入的维度。
    - d_hid: int, 隐藏层的维度。
    - dropout: float, Dropout比例，默认为0.1。
    """
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
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
        resdiual = x
        return self.layer_norm(self.w_2(self.dropout(F.relu(self.w_1(x)))) + resdiual)

if __name__ == '__main__':
    q = paddle.randn([2, 3, 27])
    k = paddle.randn([2, 4, 27])
    v = paddle.randn([2, 4, 27])
    mask = paddle.randn([2, 1, 4])
    mha = MulitheadAttention(n_head=3, d_model=27, d_k=9, d_v=9)
    q, attention = mha(q, k, v, mask)
    print(q)
    print(attention)
    pff = PositionwiseFeedForward(d_in=27, d_hid=54)
    print(pff(q))

