# NanoGPT_mss

**midterm project: NanoGPT_mss**

参考NanoGPT的模型结构，使用473M的英语小故事TinyStories数据集进行训练，生成了NanoGPT_mss，参数量约为29.94M

## Background & Environment

python 3.10

GPU RTX 3080 Ti(12GB) * 1

Ubuntu 22.04

CUDA 12.1

```
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install numpy tqdm wandb tiktoken transformers datasets
```

## Datasets——Tiny Stories

### Origin Dataset

原数据集：2.6G；valid: 23827条；train: 2,357,448条

原数据集形式：对话形式，用户输入一句话/想要写的故事的风格/包含的词语，模型回答一段连贯的故事文字

原始数据集文件：因为train很多，只给出了TinyStories-Instruct-valid.txt

### My_tinystories

数据集构建方式：从原数据集中直接选出模型写的故事文字作为预训练的训练集；原数据集中的故事有"Story: "开头字样开始，到"<|endoftext|>"结束，通过python代码get_stories.py获取

**Tinystories_20m**数据集：挑选了原来的valid中的所有数据作为数据集，按9：1的比例划分训练集和验证/测试集，大小约20M

**Path: data/Tinystories/tinystories_20m**

**Tinystories_400m**数据集：挑选了原来的数据集中184,026条数据作为数据集，按9：1的比例划分训练集和验证/测试集，大小约480M

**Path: data/Tinystories/tinystories_400m**

对数据集进行prepare.py的**按词编码**转为train.bin文件和val.bin文件；用于模型的从头开始的训练。

```
python data/Tinystories/prepare.py
```

tinystories_20M output

```
总字符长度: 20215482
train has 4,374,745 tokens
val has 482,253 tokens
```

tinystories_400M ouput

```
总字符长度: 496176751
train has 106,848,836 tokens
val has 11,875,168 tokens
```

## Model Architecture

参考NanoGPT的模型结构，见model.py

## Implement

基于构建的tinystories-20m和tinystories-400m进行模型的训练

### 模型和训练超参数的设定

在config/train_tinystories.py中修改

输出路径的含义：out-数据集名称-网络大小-循环迭代次数，例如：

```
out_dir = 'out-tinystories-400m-baby-40000'
# 指使用tinystories-400m的数据集，经过了40000次迭代训练的网络
# baby:
# n_layer = 6
# n_head = 6
# n_emd = 384
# big:
# n_layer = 12
# n_head = 6
# n_emd = 768
```

效果较好的训练（只列出主要参数）：

```
batch_size = 32 # 可能64会更好，但是所用的显卡只有12G内存，32已经最多了
block_size = 256 # 上下文长度

# baby model size:
n_layer = 6
n_head = 6
n_emd = 384

dropout = 0.2
gradient_accumulation_steps = 1
dataset = 'TinyStories'

learning_rate = 1e-3 # 小模型的学习率可以大一点
min_lr = 1e4 # 通常设为学习率的1/10
max_iters = 40000 # 15000/80000
```

### 训练模型（从头开始训练）

```
python train.py config/train_tinystories.py
```

### 输出

```
tokens per iteration will be: 8,192
Initializing a new model from scratch
defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)
number of parameters: 29.94M
num decayed parameter tensors: 26, with 30,031,872 parameters
num non-decayed parameter tensors: 13, with 4,992 parameters
using fused AdamW: True
compiling the model... (takes a ~minute)
step 0: train loss 10.9210, val loss 10.9209
iter 0: loss 10.9118, time 27488.56ms, mfu -100.00%
iter 40: loss 6.2081, time 48.67ms, mfu 10.07%
iter 80: loss 4.5387, time 49.66ms, mfu 10.05%
iter 120: loss 4.0706, time 48.02ms, mfu 10.07%
... ...
iter 19960: loss 1.8257, time 49.94ms, mfu 9.22%
step 20000: train loss 1.7219, val loss 1.7420
iter 20000: loss 1.7301, time 5985.59ms, mfu 8.30%
iter 20040: loss 1.9038, time 49.94ms, mfu 8.45%
... ...
iter 39920: loss 1.6739, time 50.22ms, mfu 9.17%
iter 39960: loss 1.6498, time 50.38ms, mfu 9.22%
step 40000: train loss 1.5748, val loss 1.5911
iter 40000: loss 1.6491, time 5955.15ms, mfu 8.31%
```

40K 循环，大概运行了40min

注：训练的详细train_loss和val_loss记录和图标在loss.xlsx文件中

## Result

通过sample.py来查看模型

```
python sample.py --out_dir=out-tinystories-400m-baby-40000
```

模型只有生成功能，无对话能力；可以通过在sample.py中修改参数，控制其输出的开头字符串

```
start = " " # 以空格为开头输出
num_samples = 10
max_new_tokens = 150
```

### 输出

```
Overriding: out_dir = out-tinystories-400m-baby-40000
number of parameters: 29.94M
No meta.pkl found, assuming GPT-2 encodings...

Once upon a time, there was a little girl named Lily. She loved to play with her toys, but one day she accidentally dropped her car and it broke. She was very sad and didn't know what to do. Lily's mom saw how sad her was and told her not to worry. She said that they could clean the car and turn it on. Lily was happy and they started to stir the car. It was a normal moment, but they worked together and it started to increase in size. After a few minutes, they had to work and the car was clean again. Lily was so happy and hugged her mom tightly. "Thank you for helping me make the car look great again," she said with a big smile.
---------------
```

## 其他数据集的尝试

除了上面提到的Tinystories数据集，还使用了很多数据集对模型进行训练，如下面所示

### tang poem

### HarryPotter

33343 paragraphs

### Shakespeare

### ROC-Stories

一个简短的故事数据集，每个故事只有五句话，下面的数据集中包含的故事数量。

| total | test | train | val  |
| ----- | ---- | ----- | ---- |
| 98161 | 4909 | 88344 | 4908 |

### west-journey

西游记的txt文件数据集，包含一整本书的内容