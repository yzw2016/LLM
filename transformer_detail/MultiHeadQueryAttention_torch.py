import torch
import torch.nn as nn

class SharedQueryMultiHeadAttention(nn.Module):
    """
    实现共享查询的多头注意力机制
    
    参数:
    - embed_dim: 嵌入维度
    - num_heads: 头的数量
    - dropout: Dropout比例，默认为0.0
    
    方法:
    - forward: 前向传播方法
    """
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(SharedQueryMultiHeadAttention, self).__init__()
        
        # 确保嵌入维度可以被头数整除
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads."
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # 定义线性变换层，用于计算查询、键和值
        self.q = nn.Linear(embed_dim, embed_dim)
        self.k = nn.Linear(embed_dim, self.head_dim)
        self.v = nn.Linear(embed_dim, self.head_dim)
        
        # 计算缩放因子，用于缩放点积注意力得分
        self.scale = self.head_dim ** -0.5
        
        # 定义残差连接和层归一化的组件
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, query, key, value, mask=None):
        """
        前向传播方法
        
        参数:
        - query: 查询向量
        - key: 键向量
        - value: 值向量
        - mask: 注意力掩码，可选，默认为None
        
        返回:
        - output: 注意力机制后的输出向量
        - attn_weights: 注意力权重
        """
        batch_size, seq_len, _ = input.shape  # input应该是query, key, value中任意一个的shape
        
        # 对查询和键值进行线性变换，并重新排列为多头形式
        query_j = self.q(query).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_j = self.k(key).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        value_j = self.v(value).unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # 计算注意力得分，并应用缩放因子
        scores = torch.matmul(query_j, key_j) * self.scale
        
        # 如果存在掩码，则应用到注意力得分上
        if mask is not None:
            scores.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        # 计算注意力权重，并应用softmax激活函数
        attn_weights = nn.functional.softmax(scores, dim=-1)
        
        # 根据注意力权重计算加权值，并恢复到原始维度
        context = torch.matmul(attn_weights, value_j).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        
        # 应用dropout和层归一化
        context = self.dropout(context)
        output = self.out_proj(context)
        
        return output, attn_weights