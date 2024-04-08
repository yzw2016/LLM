import torch.nn as nn
from Sublayer import MulitheadAttention, PositionwiseFeedForward
author = "尧志文"

class EncoderLayer(nn.Module):
    """
    Transformer模型的编码器层。

    参数:
    - d_model: 模型的维度
    - d_inner: PositionwiseFeedForward内部的维度
    - n_head: 多头注意力的头数
    - d_k: 关键向量的维度
    - d_v: 值向量的维度
    - dropout: Dropout比例，默认为0.1
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.n_head = n_head
        self.slf_attn = MulitheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        """
        前向传播函数。

        参数:
        - enc_input: 编码器的输入
        - slf_attn_mask: 自注意力掩码，用于限制注意力的范围

        返回:
        - enc_output: 编码器层的输出
        - enc_slf_attn: 自注意力的权重分布
        """
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)  # 通过PositionwiseFeedForward网络
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    """
    Transformer模型的解码器层。

    参数:
    - d_model: 模型的维度
    - d_inner: PositionwiseFeedForward内部的维度
    - n_head: 多头注意力的头数
    - d_k: 关键向量的维度
    - d_v: 值向量的维度
    - dropout: Dropout比例，默认为0.1
    """
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MulitheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MulitheadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        """
        前向传播函数。

        参数:
        - dec_input: 解码器的输入
        - enc_output: 编码器的输出
        - slf_attn_mask: 自注意力掩码，用于限制注意力的范围
        - dec_enc_attn_mask: 解码器-编码器注意力掩码，用于限制解码器注意的范围

        返回:
        - dec_output: 解码器层的输出
        - dec_slf_attn: 自注意力的权重分布
        - dec_enc_attn: 解码器-编码器注意力的权重分布
        """
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)  # 通过PositionwiseFeedForward网络
        return dec_output, dec_slf_attn, dec_enc_attn



