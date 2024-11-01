import os
import requests
import tiktoken
import numpy as np

# openai tokenizer的库，tiktoken
# 使用方法 https://zhuanlan.zhihu.com/p/629776230
data = ''
for i in range(1, 6, 1):
    input_file_path = os.path.join(os.path.dirname(__file__), f"{i}.txt")

    with open(input_file_path, 'r') as f:
        data = data + f.read()
    print(i)

print(type(data))
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
enc = tiktoken.get_encoding("gpt2")

train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train has 96,918 tokens
# val has 10,811 tokens

# 1-5
# train has 904,984 tokens
# val has 98,940 tokens