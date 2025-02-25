# "Attention is all you need" code implement.
## data preprocess
**download**

Download wmt14 from https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german

**bpe tokenizer**

```bash
mkdir dataset
cd ..
cd preprocess
python fix.py
python merge_text.py
bash bpe.sh
```
## enviroment
```
conda create -n trans python==3.9
pip install torch
pip install accelerate
pip install nltk
pip install subword_nmt
pip install datasets
```
## train
**on one card**
```bash
python train_single.py
```
**on multi cards**
```bash
accelerate config
accelerate launch train.py
```

# Thanks
Thanks for https://medium.com/@ujjalkumarmaity1998/paper-implementation-attention-is-all-you-need-transformer-59b95a93195c
I implement some code based on this blog.
