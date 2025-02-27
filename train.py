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
from dataset import TranslationDataset
from model import Transformers,get_src_mask,get_trg_mask
import torch
import nltk
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import torch.nn as nn
from accelerate import Accelerator
import os
import pickle
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import warnings
warnings.filterwarnings("ignore")
# 初始化 Accelerator
accelerator = Accelerator()

# 检查是否有多个 GPU
device = accelerator.device
nltk.download('punkt')
UNK_IDX = 0
PAD_IDX = 1
START_IDX = 2
END_IDX = 3
# # 初始化分布式环境
# def setup(rank, world_size):
#     os.environ['MASTER_ADDR'] = 'localhost'
#     os.environ['MASTER_PORT'] = '12345'
#     dist.init_process_group("gloo", rank=rank, world_size=world_size)

# # 清理分布式环境
# def cleanup():
#     dist.destroy_process_group()
# greedy decode
def greedy_decode(model, src_tensor, max_len,device):
    src_tensor = src_tensor.to(device)
    src_mask = get_src_mask(src_tensor, PAD_IDX).to(device)
    encode_out = model.encode(src_tensor, src_mask)
    trg_tensor = torch.ones(1, 1).fill_(START_IDX).type(torch.long).to(device)
    for _ in range(max_len-1):
        trg_mask = get_trg_mask(trg_tensor, PAD_IDX).to(device)
        out = model.decode(encode_out, trg_tensor, src_mask, trg_mask)

        _, next_word_id = torch.max(out[:,-1,:], dim = 1)
        next_word_id = next_word_id.item()
        trg_tensor = torch.cat([trg_tensor, torch.ones(1, 1).fill_(next_word_id).to(device)], dim=-1).type(torch.long)
        if next_word_id == END_IDX:
            break
    return trg_tensor

# 定义评估函数
def evaluate(model, dataloader,src_rev_vocab, tgt_rev_vocab,device=None, max_len=512):
    model.eval()  # Set model to evaluation mode
    generated_translations = []
    reference_translations = []
    
    with torch.no_grad():
        with tqdm(dataloader, unit="batch", desc="eval") as pbar:
            for i, (src_batch, tgt_batch) in enumerate(pbar):
                src_batch = src_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                generated_batch = []
                for src in src_batch:
                    # Convert the source sentence (English) to text
                    # src_sentence = src.cpu().numpy().tolist()
                    # src_text = [src_rev_vocab.get(idx, '<unk>') for idx in src_sentence if idx != PAD_IDX]
                    # src_text = bpe_decode(src_text)
                    # Get the translation using greedy decoding
                    translated_tensor = greedy_decode(model, src.unsqueeze(0), max_len, START_IDX)
                    translated_sentence = translated_tensor.squeeze(0).cpu().numpy().tolist()
                    gen_text = [tgt_rev_vocab.get(idx, '<unk>') for idx in translated_sentence if idx != PAD_IDX]
                    generated_batch.append(bpe_decode(gen_text))
                # Add the generated batch to the results
                generated_translations.extend(generated_batch)
                # Add the reference (target) translations
                for tgt_sentence in tgt_batch:
                    tgt_sentence = tgt_sentence.cpu().numpy().tolist()
                    tgt_text = [tgt_rev_vocab.get(idx, '<unk>') for idx in translated_sentence if idx != PAD_IDX]
                    reference_translations.append(bpe_decode(tgt_text))
    bleu_score = corpus_bleu(reference_translations, generated_translations)
    # print(f"BLEU score: {bleu_score * 100:.2f}%")
    return bleu_score


def train(model, train_loader, optimizer, criterion, num_epochs=10,eval_loader=None,src_pad_id=0,tgt_pad_id=0):
    
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch+1}/{num_epochs}",disable=not accelerator.is_local_main_process) as pbar:
            for i, (src, tgt) in enumerate(pbar):
                tgt_token_ids_batch_input = tgt[:,:-1]
                # print("srcshape",src.shape)
                # print("tgtshape",tgt.shape)

                # 生成src和trg的mask
                src_mask = get_src_mask(src, src_pad_id).to(accelerator.device)  # 获取源序列mask
                trg_mask = get_trg_mask(tgt_token_ids_batch_input, tgt_pad_id).to(accelerator.device)  # 获取目标序列mask
                src = src.to(accelerator.device)
                tgt = tgt.to(accelerator.device)
                optimizer.zero_grad()
                
                # 获取模型的输出
                output = model(src, tgt_token_ids_batch_input.to(accelerator.device), src_mask=src_mask, trg_mask=trg_mask)  # 去掉目标序列的最后一位用于输入
                
                
                # 计算损失
                loss = criterion(output.view(-1, output.size(-1)), tgt[:, 1:].contiguous().view(-1))  # 忽略<start> token
                for p in model.parameters():
                    loss += 0.0*p.sum()
                accelerator.backward(loss)
                # for name,param in model.named_parameters():
                #     if param.grad is None:
                #         print(name)
                optimizer.step()
                
                total_loss += loss.item()
                
                if (i + 1) % 10000 == 0:  # 每10000个batch打印一次损失
                    accelerator.print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {total_loss / (i+1):.4f}")
        if accelerator.is_local_main_process:  # 只在主进程保存模型
            accelerator.save(model.state_dict(), f"./checkpoint/transformer_epoch_{epoch+1}.pth")
        # 每个epoch结束后，进行验证
        if eval_loader and accelerator.is_local_main_process:
            print(f"Evaluating after epoch {epoch + 1}...")
            # 将模型移动到单卡设备
            model = accelerator.unwrap_model(model)
    
            # 将模型设置为评估模式
            model.eval()
            
            # 使用 accelerator.device 获取当前设备
            device = accelerator.device
            # 调用评估函数
            bleu_score = evaluate(model, eval_loader, src_rev_vocab, tgt_rev_vocab, device)
            print(f"BLEU: {bleu_score * 100:.2f}%")


        

if __name__ == "__main__":
    vocab = True
    
    # 假设数据集已下载并解压到./data/wmt14目录
    dataset = load_dataset("csv", data_files={
        "train": "./dataset/wmt14_translate_de-en_train_fixed.csv",
        "validation": "./dataset/wmt14_translate_de-en_validation.csv",
        "test": "./dataset/wmt14_translate_de-en_test.csv"
    })
    print('dataset loaded.')


    # 加载BPE编码文件
    bpe = apply_bpe.BPE(open('./preprocess/bpe.codes', 'r'))

    # 设定开始和结束标记符
    START_TOKEN = '<start>'
    END_TOKEN = '<end>'
    UNK_TOKEN = '<unk>'
    PAD_TOKEN = '<pad>'  # 这里定义padding的token


    # 构建BPE词汇表
    train_texts = dataset['train']
    print(len(train_texts))
    src_texts = [item['en'] for item in train_texts]
    tgt_texts = [item['de'] for item in train_texts]

    if vocab == False:
        src_vocab = build_vocab(src_texts, bpe_tokenizer, bpe)
        tgt_vocab = build_vocab(tgt_texts, bpe_tokenizer, bpe)
        # 获取 padding_id
        src_pad_id = src_vocab[PAD_TOKEN]  # 获取源语言的 padding token id
        tgt_pad_id = tgt_vocab[PAD_TOKEN]  # 获取目标语言的 padding token id
        print(f"Source padding token id: {src_pad_id}")
        print(f"Target padding token id: {tgt_pad_id}")
        # 反向词汇表
        src_rev_vocab = {idx: word for word, idx in src_vocab.items()}
        tgt_rev_vocab = {idx: word for word, idx in tgt_vocab.items()}

        save_vocab(src_vocab, src_rev_vocab, './dictionary/src_vocab.pkl', './dictionary/src_rev_vocab.pkl')
        save_vocab(tgt_vocab, tgt_rev_vocab, './dictionary/tgt_vocab.pkl', './dictionary/tgt_rev_vocab.pkl')

        print("data preprocess done")
    else:
        print("loading")
        src_vocab, src_rev_vocab = load_vocab('./dictionary/src_vocab.pkl', './dictionary/src_rev_vocab.pkl')
        tgt_vocab, tgt_rev_vocab = load_vocab('./dictionary/tgt_vocab.pkl', './dictionary/tgt_rev_vocab.pkl')
        src_pad_id = src_vocab[PAD_TOKEN]  # 获取源语言的 padding token id
        tgt_pad_id = tgt_vocab[PAD_TOKEN]  # 获取目标语言的 padding token id

    print("loaded")
    world_size = torch.cuda.device_count()  # 使用所有可用的 GPU

    train_dataset = TranslationDataset(src_texts, tgt_texts, src_vocab, tgt_vocab,bpe_tokenizer,bpe)
    train_loader = DataLoader(train_dataset, batch_size=16,num_workers=8,drop_last=False, shuffle=True)
    print(len(train_loader))
    # print(1/0)

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
   

    # 加载验证集
    val_texts = dataset['validation']
    val_src_texts = [item['en'] for item in val_texts]
    val_tgt_texts = [item['de'] for item in val_texts]
    val_dataset = TranslationDataset(val_src_texts, val_tgt_texts, src_vocab, tgt_vocab,bpe_tokenizer,bpe)
    val_loader = DataLoader(val_dataset, batch_size=8,num_workers=8, shuffle=False)
    

    train(model, train_loader, optimizer, criterion, num_epochs=10, eval_loader=val_loader,src_pad_id=src_pad_id,tgt_pad_id=tgt_pad_id)