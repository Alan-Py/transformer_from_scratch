import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
from collections import Counter
import string
import pickle

# 定义特殊符号
START_TOKEN = '<start>'
END_TOKEN = '<end>'
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'

# special_symbols列表
special_symbols = [UNK_TOKEN, PAD_TOKEN, START_TOKEN, END_TOKEN]

# 预先为特殊符号分配索引
UNK_IDX = 0
PAD_IDX = 1
START_IDX = 2
END_IDX = 3

def bpe_tokenizer(text, bpe):
    """
    对输入的文本应用BPE编码
    """
    return bpe.process_line(text).split()

# 创建词汇表
def build_vocab(texts, tokenizer,bpe):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text, bpe))
    # 创建一个空的词汇表，先加入特殊符号
    vocab = {word: idx for idx, word in enumerate(special_symbols)}
    # 为其他单词分配索引，从 4 开始
    current_idx = len(special_symbols)
    for word, count in counter.items():
        if word not in vocab:  # 避免特殊符号被覆盖
            vocab[word] = current_idx
            current_idx += 1

    return vocab

# 将BPE文本转换为索引序列
def text_to_sequence_bpe(text, vocab, tokenizer, bpe):
    return [vocab.get(word, vocab['<unk>']) for word in tokenizer(text, bpe)]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# bpe decoder
def bpe_decode(translated_words):
    """
    将 BPE 编码的子词列表解码为完整的句子。
    """
    sentence = []
    current_word = ""
    
    for subword in translated_words:
        if subword.endswith("@@"):
            current_word += subword[:-2]  # 去掉 '@@' 并拼接到当前单词
        else:
            current_word += subword
            sentence.append(current_word)
            current_word = ""
    
    # 如果最后还有未处理的单词，加入句子
    if current_word:
        sentence.append(current_word)
    
    return " ".join(sentence)

# 保存词汇表和反向词汇表
def save_vocab(vocab, rev_vocab, vocab_path, rev_vocab_path):
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    with open(rev_vocab_path, 'wb') as f:
        pickle.dump(rev_vocab, f)

# 加载词汇表和反向词汇表
def load_vocab(vocab_path, rev_vocab_path):
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    with open(rev_vocab_path, 'rb') as f:
        rev_vocab = pickle.load(f)
    return vocab, rev_vocab
