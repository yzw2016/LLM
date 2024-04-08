import paddle
import paddle.nn as nn
import math

class MultiHeadSelfAttention(nn.Layer):
    """
    多头注意力模块，用于实现transformer模型中的注意力机制。
    
    参数:
        model_dim: 模型维度，即输入和输出的向量维度。
        num_heads: 注意力头的数量。
        dropout_rate: Dropout率，防止模型过拟合，默认为0.1。
    """
    def __init__(self, model_dim, num_heads, dropout_rate=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        assert model_dim % num_heads == 0, "model_dim 必须能整除注意力头的数量。"
        self.query_projection = nn.Linear(model_dim, model_dim)
        self.key_projection = nn.Linear(model_dim, model_dim)
        self.value_projection = nn.Linear(model_dim, model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(axis=-1)
    
    def forward(self, inputs, attention_mask=None, target=None):
        """
        前向传播函数。

        参数：
        - inputs: 输入张量，形状为(batch_size, sequence_length, model_dim)。
        - mask: 掩码张量，形状为(batch_size, sequence_length, sequence_length)。

        返回：
        - output: 输出张量，形状为(batch_size, sequence_length, model_dim)。
        """

        batch_size, sequence_length, _ = inputs.shape

        # 对Query、Key和Value进行线性变换
        querys = self.query_projection(inputs)
        keys = self.key_projection(inputs)
        values = self.value_projection(inputs)

        # 进行矩阵分割以实现多头注意力
        querys = querys.reshape([batch_size, sequence_length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        keys = keys.reshape([batch_size, sequence_length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        values = values.reshape([batch_size, sequence_length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])

        # 计算scaled dot-product attention,考虑注意力掩码
        attention_scores = paddle.matmul(querys, keys, transpose_y=True) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand([-1, self.num_heads, sequence_length, -1])
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_probs = self.softmax(attention_scores)
        # 应用训练阶段的dropout
        if target is not None:
            attention_probs = self.dropout(attention_probs) 
        attention_weights = paddle.matmul(attention_probs, values).transpose([0, 2, 1, 3]).reshape([batch_size, sequence_length, self.model_dim])
        output = self.output(attention_weights)
        return output, attention_probs

# 使用示例：
model_dim = 512
num_heads = 8
mask_attention = paddle.to_tensor([[1 if i < 8 else 0 for i in range(10)]], dtype='int64')
attention_layer = MultiHeadSelfAttention(model_dim, num_heads)
inputs = paddle.randn((1, 10, model_dim))  # 假设我们有一个批次大小为1，序列长度为10，模型维度为512的输入
outputs, attention_weight = attention_layer(inputs, mask_attention)
print(outputs)
print(attention_weight)