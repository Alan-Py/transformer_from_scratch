import torch
import math
import torch.nn as nn
import copy
from torch.autograd import gradcheck
# clone modules
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Embeddings(nn.Module):
    def __init__(self,vocab_size,dmodel=512) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.dmodel = dmodel
        self.embed_layer = nn.Embedding(self.vocab_size,self.dmodel)
    def forward(self,x):
        embedout = self.embed_layer(x)
        # print(embedout.shape)
        # print(embedout)
        return embedout * math.sqrt(self.dmodel)
class PositionEmbeddings(nn.Module):
    def __init__(self,max_seq_len,dropout,d_model=512) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        pos = torch.arange(0, max_seq_len,dtype = torch.float).unsqueeze(1)
        # we know a^-x  is equals to 1/a^x
        div = torch.pow(10000,-torch.arange(0,d_model,2,dtype=torch.float)/self.d_model)
        pe = torch.zeros((max_seq_len,d_model))
        pe[:,0::2] = torch.sin(pos * div)
        pe[:,1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,embed_vec):
        pe = self.pe[:,:embed_vec.size()[1]]
        return embed_vec + pe
# 计算注意力分数
def attention(q,k,v,mask=None,dropout=None):
    # calculate attenction score
    # query = (BS,NH,S/T,HD) , key.transpose(-2,-1) = (BS,NH,HD,S/T) 
    # BS（Batch Size）：批次大小
    # NH（Num Heads）：注意力头数
    # S/T（Sequence Length / Time Steps）：序列长度
    # S（Source Length）：Encoder 侧的序列长度（输入序列的长度）
    # T（Target Length）：Decoder 侧的序列长度（输出序列的长度）
    # HD（Head Dimension）：每个注意力头的维度
    # attention score size for encoder attention = (BS,NH,S,S) , decoder attention = (BS,NH,T,T), encoder-decoder attention = (BS,NH,T,S)
    d_k = q.size(-1)
    # (BS, NH, Query序列长度, Key序列长度)
    attention_score = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attention_score = attention_score.masked_fill(mask == 0, -1e9)
    p_attn = attention_score.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    score = p_attn@v
    return score,attention_score
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512,n_head=8, dropout_rate=0.1) -> None:
        super().__init__()
        assert d_model%n_head == 0
        self.d_k = d_model // n_head
        # 每个头的 query、key、value 维度都是 d_k。
        self.dropout = nn.Dropout(p=dropout_rate)
        self.head = n_head
        self.d_model = d_model
        # qkv and output
        self.w_key = nn.Linear(d_model,d_model)
        self.w_query = nn.Linear(d_model,d_model)
        self.w_value = nn.Linear(d_model,d_model)
        self.output_project = nn.Linear(d_model,d_model)
    def forward(self,q,k,v,mask=None):
        B = k.shape[0]
        key, query, value = self.w_key(k), self.w_query(q), self.w_value(v)
        #key = (B,seq_len,d_model)
        # split vector by number of head and transpose
        key = key.view(B,-1,self.head,self.d_k).transpose(1,2)#(batch_size, n_head, seq_len, head_dim)
        query = query.view(B,-1,self.head,self.d_k).transpose(1,2)
        value = value.view(B,-1,self.head,self.d_k).transpose(1,2)
        
        attention_score,_ = attention(query,key,value,mask,dropout=self.dropout)
        
        attention_score = attention_score.transpose(1,2)
        attention_score = attention_score.reshape(B,-1,self.d_model)
        
        attention_out = self.output_project(attention_score)
        return attention_out# (B,seq_len,d_model)
# PositionwiseFeedForward(x) = max(0,w1x+b1)w2 + b2
# w1 = (d_model,hidden_dim)
# w2 = (hidden_dim,d_model)
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model=512,hidden_dim=2048,dropout_rate = 0.1):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(d_model,hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,d_model)
    def forward(self,x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
# sublayer residual
class SubLayerConnection(nn.Module):
    def __init__(self,d_model,dropout):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,out_x):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a Tensor, but got {type(x)}")
        if not isinstance(out_x, torch.Tensor):
            raise TypeError(f"Expected out_x to be a Tensor, but got {type(out_x)}")

        return self.norm(x + self.dropout(out_x))
        
# sublayer for encoder
class EncoderLayer(nn.Module):
    def __init__(self,multi_head_attention,Position_wise_FeedForward,d_model=512,dropout_rate = 0.1) ->None:
        super().__init__()
        self.d_model = d_model
        self.muti_head_attention = multi_head_attention
        self.position_wise_FeedForward = Position_wise_FeedForward
        self.subLayer1 = SubLayerConnection(d_model,dropout_rate)
        self.sublayer2 = SubLayerConnection(d_model,dropout_rate)
    def forward(self,x,mask=None):
        
        atten_out = self.subLayer1(x,self.muti_head_attention(x, x, x, mask))
        return self.sublayer2(atten_out,self.position_wise_FeedForward(atten_out))
        # 返回encoder out (B,seq_len,d_model)
# Encoder
class EncoderBlock(nn.Module):
    def __init__(self, encoder_layer, N):
        super().__init__()
        self.layers = clones(encoder_layer,N)
    def forward(self,x,mask):
        for layer in self.layers:
            x = layer(x,mask)
        return x
# sublayer for decoder
class DecoderLayer(nn.Module):
    def __init__(self, multi_head_attention,Position_wise_FeedForward,d_model=512,dropout_rate = 0.1) ->None:
        super().__init__()
        self.d_model = d_model
        self.decoder_attention = copy.deepcopy(multi_head_attention)
        self.encoder_decoder_attention = copy.deepcopy(multi_head_attention)
        self.position_wise_FeedForward = Position_wise_FeedForward
        self.subLayer1 = SubLayerConnection(d_model,dropout_rate)
        self.sublayer2 = SubLayerConnection(d_model,dropout_rate)
        self.sublayer3 = SubLayerConnection(d_model,dropout_rate)
    def forward(self,enc,dec,src_mask=None,trg_mask=None):
        decoder_attention_out = self.decoder_attention(dec,dec,dec,trg_mask)
        decoder_attention_out = self.subLayer1(dec,decoder_attention_out)
        
        enc_dec_attention_out = self.encoder_decoder_attention(decoder_attention_out,enc,enc,src_mask)
        enc_dec_attention_out = self.sublayer2(decoder_attention_out,enc_dec_attention_out)
        
        out = self.position_wise_FeedForward(enc_dec_attention_out)
        out = self.sublayer3(enc_dec_attention_out,out)
        return out
class DecoderBlock(nn.Module):
    def __init__(self, decoder_layer,N):
        super().__init__()
        self.layers = clones(decoder_layer,N)
        
    def forward(self,enc,dec,src_mask=None,trg_mask=None):
        dec_out = dec
        for layer in self.layers:
            dec_out = layer(enc,dec_out,src_mask,trg_mask)
        return dec_out
# decoder generate
class DecoderGenerator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.log_softmax = nn.LogSoftmax(dim=-1)
    def forward(self,x):
        return self.log_softmax(self.proj(x))
# Transformer Block
class Transformers(nn.Module):
    def __init__(self,src_seq_len,trg_seq_len,d_model,num_head,dropout_rate = 0.1) -> None:
        super().__init__()
        self.src_seq_len = src_seq_len
        self.trg_seq_len = trg_seq_len
        self.d_model = d_model
        self.num_head = num_head
        
        self.src_embedding = Embeddings(src_seq_len,d_model)
        self.src_pe = PositionEmbeddings(src_seq_len,dropout_rate,d_model)
        self.trg_embedding = Embeddings(trg_seq_len,d_model)
        self.trg_pe = PositionEmbeddings(trg_seq_len,dropout_rate,d_model)
        self.multi_head_attention = MultiHeadAttention(d_model,num_head,dropout_rate)
        self.position_wise_feedforward = PositionwiseFeedForward(d_model=d_model,dropout_rate=dropout_rate)
        self.encoder_layer = EncoderLayer(self.multi_head_attention,self.position_wise_feedforward,d_model,dropout_rate)
        self.encoder_block = EncoderBlock(self.encoder_layer,N=6)
        self.decoder_layer = DecoderLayer(self.multi_head_attention,self.position_wise_feedforward,d_model,dropout_rate)
        self.decoder_block = DecoderBlock(self.decoder_layer,N=6)
        
        self.decoder_out = DecoderGenerator(d_model,trg_seq_len)
    def forward(self,src_token_id,trg_token_id,src_mask=None,trg_mask=None):
        encoder_out = self.encode(src_token_id,src_mask)
        decoder_out = self.decode(encoder_out,trg_token_id,src_mask,trg_mask)
        return decoder_out
    
    def encode(self,src_token_id,src_mask):
        embed = self.src_embedding(src_token_id)
        pe_out = self.src_pe(embed)
        encoder_out = self.encoder_block(pe_out,src_mask)
        return encoder_out
    def decode(self,enc_out,trg_token_id,src_mask,trg_mask):
        embed = self.trg_embedding(trg_token_id)
        pe_out = self.trg_pe(embed)
        decoder_out = self.decoder_block(enc_out,pe_out,src_mask,trg_mask)
        # print(decoder_out.shape)
        decoder_out = self.decoder_out(decoder_out)
        return decoder_out

def get_src_mask(src_token_ids_batch,pad_tok_id):
    batch_size = src_token_ids_batch.size()[0]
    src_mask = (src_token_ids_batch!=pad_tok_id).view(batch_size, 1, 1,-1) #SIZE = (BS,1,1,S)
    return src_mask
def get_trg_mask(trg_token_ids_batch,pad_tok_id):
    batch_size = trg_token_ids_batch.size()[0]
    seq_len = trg_token_ids_batch.size()[1]
    trg_pad_mask = (trg_token_ids_batch!=pad_tok_id).view(batch_size, 1, 1,-1) #SIZE = (BS,1,1,T)
    # 创建目标序列的 look ahead mask（上三角矩阵）
    trg_look_forward = torch.triu(torch.ones(1, 1, seq_len, seq_len, device=trg_token_ids_batch.device), diagonal=1).to(torch.bool)
    trg_mask = trg_pad_mask & trg_look_forward
    return trg_mask
if __name__ == "__main__":

    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    low_bound = 3
    high_bound = 15
    batch_size = 32
    src_seq_len = 10
    trg_seq_len = 15

    src_tensor_size = (batch_size, src_seq_len)  
    trh_tensor_size = (batch_size, trg_seq_len)  

    src_seq = torch.randint(3, 16, size=src_tensor_size, dtype=torch.float32, requires_grad=True)
    trg_seq = torch.randint(3, 16, size=trh_tensor_size, dtype=torch.float32, requires_grad=True)
    transformer = Transformers(
        src_seq_len = 20,
        trg_seq_len = 20,
        d_model = 512,
        num_head = 8,
        dropout_rate = 0.2
    )
    src_mask = get_src_mask(src_seq,PAD_IDX)
    trg_mask = get_src_mask(trg_seq,PAD_IDX)
    output = transformer(src_seq, trg_seq,src_mask,trg_mask)
    print(output.shape)