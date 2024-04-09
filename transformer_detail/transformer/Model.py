import torch
import torch.nn as nn
from Layers import EncoderLayer, DecoderLayer
import numpy as np
author = "尧志文"

class PositionalEncoding(nn.Module):
    """
    位置编码类，用于增加模型对位置信息的感知能力。

    参数:
    - d_hid: 隐藏层维度
    - n_position: 位置编码的最大长度，默认为200
    """
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # 注册位置编码表为模型的buffer，不会被优化器更新
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """
        生成正弦编码表。

        参数:
        - n_position: 位置编码的最大长度
        - d_hid: 隐藏层维度

        返回值:
        - 返回一个形状为[1, n_position, d_hid]的张量，包含位置编码信息。
        """
        # 根据位置和维度生成位置编码
        pe_table = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_hid) for j in range(d_hid)]
            if pos != 0 else np.zeros(d_hid) for pos in range(n_position)])
        # 对编码进行正弦和余弦操作
        pe_table[:, 0::2] = np.sin(pe_table[:, 0::2])  # 对偶数索引应用正弦函数
        pe_table[:, 1::2] = np.cos(pe_table[:, 1::2])  # 对奇数索引应用余弦函数
        return torch.FloatTensor(pe_table).unsqueeze(0)

    def forward(self, x):
        """
        前向传播过程，将位置编码添加到输入张量x中。

        参数:
        - x: 输入张量，形状为[B, L, d_hid]，其中B为批次大小，L为序列长度

        返回值:
        - 添加了位置编码的输入张量x。
        """
        # 在位置编码表中选择与输入序列长度相同的编码，并与输入张量相加
        return x + self.pos_table[:, :x.size(1), :].clone().detach()
    
def get_pad_mask(seq, pad_idx):
    """
    生成一个掩码张量，用于表示序列中每个位置是否为填充位置。

    参数:
    - seq: 输入序列张量，形状为[B, L]，其中B为批次大小，L为序列长度

    返回值:
    - 掩码张量，形状为[B, 1, L]，其中B为批次大小，L为序列长度
   """
 #   return (seq != pad_idx).unsqueeze(1).expand(-1, seq.size(1), -1)
    return (seq != pad_idx).unsqueeze(-2)

def get_subsequent_mask(seq):
    """
    生成一个掩码张量，用于表示序列中每个位置是否为未来位置。
    参数:
    - seq: 输入序列张量，形状为[B, L]，其中B为批次大小，L为序列长度
    返回值:
    - 掩码张量，形状为[B, L, L]，其中B为批次大小，L为序列长度
    """
    mask = torch.triu(torch.ones(seq.size(1), seq.size(1)), diagonal=1)
    mask = mask.bool().unsqueeze(0).expand(seq.size(0), -1, -1)
    return mask


class Encoder(nn.Module):
    """
    用于定义一个编码器，可以处理变长输入序列。
    
    参数:
    - n_src_vocab: 源词汇表大小。
    - d_word_vec: 词向量维度。
    - n_layers: 编码器层的数量。
    - n_head: 注意力头的数量。
    - d_k: 关键向量的维度。
    - d_v: 值向量的维度。
    - d_model: 模型的维度。
    - d_inner: 内部隐藏层的维度。
    - pad_idx: 填充符号的索引。
    - dropout: Dropout比例，默认为0.1。
    - n_position: 位置编码的最大位置数，默认为200。
    - scale_emb: 是否对嵌入层的输出进行缩放，默认为False。
    """
    def __init__(self, n_src_vocab,  d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        super().__init__()
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)  # 词嵌入层
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)  # 位置编码层
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])  # 编码器层堆栈
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化
        self.scale_emb = scale_emb  # 是否缩放嵌入层输出
        self.d_model = d_model  # 模型维度

    def forward(self, src_seq, src_mask=None, return_attns=False):
        """
        前向传播函数。
        
        参数:
        - src_seq: 输入的源序列。
        - src_pad_mask: 填充掩码，用于指示哪些位置是填充的。
        - src_mask: 自注意力掩码，用于遮蔽未来位置的信息。
        - return_attns: 是否返回每一层的注意力权重，默认为False。
        
        返回:
        - 如果return_attns为True，则返回编码器输出和注意力权重列表。
        - 如果return_attns为False，则仅返回编码器输出。
        """
        enc_output = self.src_word_emb(src_seq)  # 词嵌入
        if self.scale_emb:
            enc_output = enc_output * np.sqrt(self.d_model)  # 缩放嵌入层输出
        enc_output = self.dropout(self.position_enc(enc_output))  # 加入位置编码
        enc_output = self.layer_norm(enc_output)  # 层归一化
        enc_slf_attn_list = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]  # 收集注意力权重
        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder(nn.Module):
    """
    解码器类，用于处理序列解码任务。
    
    参数:
    - n_tgt_vocab: 目标词汇表大小。
    - d_word_vec: 词向量维度。
    - n_layers: 解码器层的数量。
    - n_head: 注意力头的数量。
    - d_k: 注意力键的维度。
    - d_v: 注意力值的维度。
    - d_model: 模型的维度。
    - d_inner: 解码器层中FFN的内部维度。
    - pad_idx: 填充符号的索引。
    - dropout: Dropout比例，默认为0.1。
    - n_position: 位置编码的最大位置数，默认为200。
    - scale_emb: 是否对嵌入层输出进行缩放，默认为False。
    """
    def __init__(self, n_tgt_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
                 d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False):
        super().__init__()
        self.trg_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=pad_idx)  # 目标词嵌入
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)  # 位置编码
        self.dropout = nn.Dropout(dropout)  # Dropout层
        self.layer_stack = nn.ModuleList([  # 解码器层堆栈
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)  # 层归一化
        self.scale_emb = scale_emb  # 是否缩放嵌入层
        self.d_model = d_model  # 模型维度
    
    def forward(self, trg_seq, enc_output, src_mask=None, trg_mask=None,  return_attns=False):
        """
        前向传播函数。
        
        参数:
        - tgt_seq: 目标序列。
        - enc_output: 编码器的输出。
        - non_pad_mask: 非填充掩码，用于关注非填充位置。
        - src_mask: 源序列的掩码，用于注意力机制。
        - trg_mask: 目标序列的掩码，用于防止当前位置预测未来的单词。
        - return_attns: 是否返回注意力权重，默认为False。
        
        返回值:
        - 如果return_attns为True，则返回解码器输出、自注意力权重列表和编码器-解码器注意力权重列表。
        - 如果return_attns为False，则仅返回解码器输出。
        """
        dec_slf_attn_list, dec_enc_attn_list = [], []  # 用于存储注意力权重的列表
        dec_output = self.trg_word_emb(trg_seq)  # 目标词嵌入
        if self.scale_emb:
            dec_output = dec_output * np.sqrt(self.d_model)  # 缩放嵌入层输出
        dec_output = self.dropout(self.position_enc(dec_output))  # 位置编码和dropout
        dec_output = self.layer_norm(dec_output)  # 层归一化
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=src_mask, dec_enc_attn_mask=trg_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,
    
class Transformer(nn.Module):
    """
    实现一个Transformer模型，用于序列到序列的学习任务。
    
    参数:
    - n_src_vocab: 源语言词汇表大小。
    - n_tgt_vocab: 目标语言词汇表大小。
    - src_pad_idx: 源语言填充符号的索引。
    - trg_pad_idx: 目标语言填充符号的索引。
    - len_max_seq: 最大序列长度。
    - d_word_vec: 词向量维度，默认为512。
    - d_model: 模型的维度，默认为512。
    - d_inner: 隐藏层的维度，默认为2048。
    - n_layers: 压缩层的数量，默认为6。
    - n_head: 注意力头的数量，默认为8。
    - d_k: 注意力机制中键的维度，默认为64。
    - d_v: 注意力机制中值的维度，默认为64。
    - dropout: Dropout比例，默认为0.1。
    - tgt_emb_prj_weight_sharing: 是否共享目标语言的嵌入和投影层的权重，默认为True。
    - emb_src_trg_weight_sharing: 是否共享源和目标语言的嵌入层权重，默认为True。
    - scale_emb_or_prj: 是否对嵌入层或投影层进行缩放，默认为'none'。
    
    返回:
    - 无
    """
    def __init__(self, n_src_vocab, n_tgt_vocab, src_pad_idx, trg_pad_idx, len_max_seq, d_word_vec=512, d_model=512, d_inner=2048, 
                 n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, tgt_emb_prj_weight_sharing=True, 
                 emb_src_trg_weight_sharing=True, scale_emb_or_prj='none'):
        super().__init__()
        self.src_pad_id = src_pad_idx
        self.trg_pad_id = trg_pad_idx
        
        # 确认缩放选项的正确性
        assert scale_emb_or_prj in ['none', 'emb', 'prj', 'emb+prj']
        # 根据缩放选项设置嵌入和投影是否缩放
        scale_emb = (scale_emb_or_prj == 'emb') or (scale_emb_or_prj == 'emb+prj')
        self.scale_prj = (scale_emb_or_prj == 'prj') or (scale_emb_or_prj == 'emb+prj')
        self.dmodel = d_model
        
        # 构建编码器和解码器
        self.encoder = Encoder(n_src_vocab=n_src_vocab, d_word_vec=d_word_vec, n_layers=n_layers, n_head=n_head, 
                               d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=src_pad_idx, 
                               dropout=dropout, scale_emb=scale_emb)
        self.decoder = Decoder(n_tgt_vocab=n_tgt_vocab, d_word_vec=d_word_vec, n_layers=n_layers, n_head=n_head, 
                               d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, pad_idx=trg_pad_idx, 
                              dropout=dropout, scale_emb=scale_emb)
        self.trg_word_proj = nn.Linear(d_model, n_tgt_vocab, bias=False)

        # 根据参数共享权重
        if tgt_emb_prj_weight_sharing:
            self.trg_word_proj.weight = self.decoder.trg_word_emb.weight
        if emb_src_trg_weight_sharing:
            self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight

    def forward(self, src_seq, trg_seq):
        """
        前向传播过程。
        
        参数:
        - src_seq: 源语言序列。
        - trg_seq: 目标语言序列。
        
        返回:
        - seq_logits: 序列的logits，形状为(-1, n_tgt_vocab)。
        """
        # 生成掩码
        src_mask = get_pad_mask(src_seq, self.src_pad_id)
        trg_mask = get_pad_mask(trg_seq, self.trg_pad_id) & get_subsequent_mask(trg_seq)
        # 编码输入序列
        enc_output, *_ = self.encoder(src_seq, src_mask)
        # 解码输出序列
        dec_output, *_ = self.decoder(trg_seq, enc_output, trg_mask, src_mask)
        # 序列logits
        seq_logits = self.trg_word_proj(dec_output)
        # 如果需要，对logits进行缩放
        if self.scale_prj:
            seq_logits *= self.dmodel ** -0.5
        return seq_logits.view(-1, seq_logits.size(-1))