# 合并英语和德语训练数据
cat train.en train.de > train.txt

# 使用 subword-nmt 来学习 BPE（假设你想要创建一个 30k 的词汇表）
subword-nmt learn-bpe -s 30000 < train.txt > bpe.codes

# 使用 bpe.codes 对英语和德语句子应用 BPE 编码
subword-nmt apply-bpe -c bpe.codes < train.en > train.en.bpe
subword-nmt apply-bpe -c bpe.codes < train.de > train.de.bpe