from datasets import load_dataset
from subword_nmt import apply_bpe
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from collections import Counter
import string
from util import bpe_tokenizer,build_vocab,text_to_sequence_bpe,count_parameters,bpe_decode,save_vocab,load_vocab
from dataset import TranslationDataset,collate_fn
from model import Transformers,get_src_mask,get_trg_mask
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import torch.nn as nn
from accelerate import Accelerator
import os
import pickle
from train import evaluate

import warnings
warnings.filterwarnings("ignore")
# 检查是否有多个 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你可能需要下载一些 `nltk` 数据包，尤其是 BLEU 评估相关的
nltk.download('punkt')
UNK_IDX = 0
PAD_IDX = 1
START_IDX = 2
END_IDX = 3
if __name__ == "__main__":
    vocab = True
    
    # 假设数据集已下载并解压到./data/wmt14目录
    dataset = load_dataset("csv", data_files={
        "test": "./dataset/wmt14_translate_de-en_test.csv"
    })
    print('dataset loaded.')


    # 加载BPE编码文件
    bpe = apply_bpe.BPE(open('bpe.codes', 'r'))

    # 设定开始和结束标记符
    START_TOKEN = '<start>'
    END_TOKEN = '<end>'
    UNK_TOKEN = '<unk>'
    PAD_TOKEN = '<pad>'  # 这里定义padding的token

    src_vocab, src_rev_vocab = load_vocab('./dictionary/src_vocab.pkl', './dictionary/src_rev_vocab.pkl')
    tgt_vocab, tgt_rev_vocab = load_vocab('./dictionary/tgt_vocab.pkl', './dictionary/tgt_rev_vocab.pkl')
    src_pad_id = src_vocab[PAD_TOKEN]  # 获取源语言的 padding token id
    tgt_pad_id = tgt_vocab[PAD_TOKEN]  # 获取目标语言的 padding token id

    print("loaded")
    model = Transformers(
            src_seq_len = len(src_vocab),
            trg_seq_len = len(tgt_vocab),
            d_model = 512,
            num_head = 8,
            dropout_rate = 0.1
        )
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_vocab[PAD_TOKEN])
    
    # 加载测试集
    test_texts = dataset['test']
    test_src_texts = [item['en'] for item in test_texts]
    test_tgt_texts = [item['de'] for item in test_texts]
    test_dataset = TranslationDataset(test_src_texts, test_tgt_texts, src_vocab, tgt_vocab,bpe_tokenizer,bpe)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

    # 测试模型
    model_path = "./checkpoint/transformer_epoch_best.pth"  # 训练好的模型权重路径
     # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    # model.eval()  # 设置模型为评估模式
    bleu_score = evaluate(model, test_loader, src_rev_vocab, tgt_rev_vocab, device)
    print(f"BLEU: {bleu_score * 100:.2f}%")