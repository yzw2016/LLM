import paddle
from Model import Encoder, Decoder, Transformer
author = "尧志文"

# 定义源序列和目标序列的张量，这里使用IntTensor，并给出示例数据
src_seq = paddle.to_tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 5, 6, 4, 3, 9, 5, 2, 0]])
tgret_seq = paddle.to_tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 5, 6, 4, 3, 9, 5, 2, 0]])

# 初始化Transformer模型，设置源词汇表大小、目标词汇表大小、源和目标的填充索引、最大序列长度和层数
model = Transformer(n_src_vocab=10, n_tgt_vocab=20, src_pad_idx=0, trg_pad_idx=0, len_max_seq=10, n_layers=1)

# 使用模型对输入的源序列和目标序列进行处理，获取模型输出
a = model(src_seq, tgret_seq)

# 打印模型输出的形状，以供检查
print(a.shape)