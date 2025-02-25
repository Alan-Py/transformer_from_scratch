from datasets import load_dataset

# 加载数据
dataset = load_dataset("csv", data_files={
    "train": "../dataset/wmt14_translate_de-en_train_fixed.csv",
    "validation": "../dataset/wmt14_translate_de-en_validation.csv",
    "test": "../dataset/wmt14_translate_de-en_test.csv"
})

# 提取英语和德语文本
# print(dataset['train'][0])
train_texts = dataset['train']
for item in train_texts:
    # print(item.keys())
    if item['en'] == None:
        print("yes")
    if item['de'] == None:
        print('NO')
        print(item)
print("finish")
src_texts = [item['en'] for item in train_texts]
tgt_texts = [item['de'] for item in train_texts]

# 保存英语和德语文本到文件
with open("train.en", "w") as f:
    for sentence in src_texts:
        f.write(sentence + "\n")

with open("train.de", "w") as f:
    for sentence in tgt_texts:
        f.write(sentence + "\n")