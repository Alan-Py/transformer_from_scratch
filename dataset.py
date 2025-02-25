import torch
from torch.utils.data import Dataset, DataLoader
from util import bpe_tokenizer,build_vocab,text_to_sequence_bpe
from torch.nn.utils.rnn import pad_sequence
UNK_IDX = 0
PAD_IDX = 1
START_IDX = 2
END_IDX = 3
class TranslationDataset(Dataset):
    def __init__(self, source_texts, target_texts, src_vocab, tgt_vocab, tokenizer, bpe, max_len=512):
        self.source_texts = source_texts
        self.target_texts = target_texts
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer
        self.bpe = bpe
        self.max_len = max_len

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        src_text = self.source_texts[idx]
        tgt_text = self.target_texts[idx]

        src_sequence = text_to_sequence_bpe(src_text, self.src_vocab, self.tokenizer, self.bpe)
        tgt_sequence = text_to_sequence_bpe(tgt_text, self.tgt_vocab, self.tokenizer, self.bpe)
        # print(type(src_sequence), type(tgt_sequence))
        # print(len(src_sequence),len(tgt_sequence))
        # 在每个序列前后添加 <START> 和 <END> token
        src_sequence = [START_IDX] + src_sequence + [END_IDX]
        tgt_sequence = [START_IDX] + tgt_sequence + [END_IDX]

        # 手动填充序列到最大长度
        src_padding_len = self.max_len - len(src_sequence)
        tgt_padding_len = self.max_len - len(tgt_sequence)
        # 裁剪长度，以便在每个序列前后添加 <START> 和 <END> token
        max_content_len = self.max_len - 2  # 2 tokens for <START> and <END>
        if len(src_sequence) > max_content_len:
            src_sequence = src_sequence[:max_content_len]
        if len(tgt_sequence) > max_content_len:
            tgt_sequence = tgt_sequence[:max_content_len]        
        # 添加 <START> 和 <END> token
        src_sequence = [START_IDX] + src_sequence + [END_IDX]
        tgt_sequence = [START_IDX] + tgt_sequence + [END_IDX]

        # 手动填充序列到最大长度
        src_padding_len = self.max_len - len(src_sequence)
        tgt_padding_len = self.max_len - len(tgt_sequence)
        
        # 如果长度不足，填充
        if src_padding_len > 0:
            src_sequence = src_sequence + [PAD_IDX] * src_padding_len
        if tgt_padding_len > 0:
            tgt_sequence = tgt_sequence + [PAD_IDX] * tgt_padding_len

        return torch.tensor(src_sequence), torch.tensor(tgt_sequence)
# def collate_fn(data_batch):
#     # 从 dataset 获取已经处理过 padding 的数据
#     en_batch, de_batch = zip(*data_batch)
#     # 计算目标语言和源语言的最大长度
#     max_len_de = max([len(de_tensor) for de_tensor in de_batch]) + 2
#     max_len_en = max([len(en_tensor) for en_tensor in en_batch]) + 2

#     # 填充目标语言（de_batch）和源语言（en_batch）
#     padded_en_batch = []
#     padded_de_batch = []
    
#     for en_tensor, de_tensor in zip(en_batch, de_batch):
#         # 计算补齐的长度
#         en_padding_len = max_len_en - len(en_tensor)
#         de_padding_len = max_len_de - len(de_tensor)
        
#         # 对源语言 (en) 和目标语言 (de) 进行填充
#         if en_padding_len > 0:
#             en_tensor = torch.cat([en_tensor, torch.tensor([PAD_IDX] * en_padding_len)])
#         if de_padding_len > 0:
#             de_tensor = torch.cat([de_tensor, torch.tensor([PAD_IDX] * de_padding_len)])
        
#         padded_en_batch.append(en_tensor)
#         padded_de_batch.append(de_tensor)

#     # 将列表转换为 Tensor
#     en_batch = torch.stack(padded_en_batch, dim=0)
#     de_batch = torch.stack(padded_de_batch, dim=0)
#     return en_batch, de_batch