import paddle
import paddle.nn as nn
import paddle.nn.functional as F
author = "yaozhiwen"

class ScaledDotProductAttention(nn.Layer):
    """
     实现缩放点积注意力机制。

    参数:
    - temperature: 温度参数，用于缩放点积注意力的计算。
    - attn_dropout: 注意力dropout的比例，默认为0.1。
    """
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
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
        attnention = paddle.matmul(q / self.temperature, k, transpose_y=True)
        if mask is not None:
            attnention = attnention.masked_fill(mask == 0, -1e9)
        output = paddle.matmul(self.dropout(F.softmax(attnention, axis=-1)), v)
        return output, attnention
    
if __name__ == "__main__":
    q = paddle.randn([2, 3, 5, 10])
    k = paddle.randn([2, 3, 5, 10])
    v = paddle.randn([2, 3, 5, 10])
    mask = (1-paddle.triu(paddle.ones([2, 1, 5, 5]), diagonal=1)).astype('bool')
    attention = ScaledDotProductAttention(temperature=0.1)
    output, attn = attention(q,k, v, mask)
    print(output)
    print(attn)
