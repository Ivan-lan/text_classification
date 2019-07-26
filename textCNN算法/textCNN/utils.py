from collections import Counter

def read_file(path):
    """
    读取文件
    需指定文件格式:【标签\t文本】
    """
    texts, labels = [],[]
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            try:
                label, text = line.strip().split('\t')
                if text:
                    texts.append(list(text))  # 字符级CNN [['字','符','级']]
                    labels.append(label)
            except:
                pass
    return texts, labels


def build_vocab(train_path, vocab_path, vocab_size = 5000):
    """根据训练集构建字符表,保存本地"""
    texts, labels = read_file(train_path)
    all_data = []
    for text in texts:
        all_data.extend(text)
    counter = Counter(all_data) # 构建字符出现频数字典 Counter({'你': 1, '好': 2, '吗': 1, '我': 1})
    topn_pairs = counter.most_common(vocab_size-1) # 返回频数前N的 [('好', 2), ('你', 1)]
    words, _ = list(zip(*topn_pairs))  # [('好', '你'), (2, 1)]
    words = ['<PAD>'] + list(words)
    with open(vocab_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(words) + '\n')  # '你\n好\n吗\n我\n好\n'
    return texts, labels

import json

def word2id(vocab_path,word2id_path): # 词表路径和词-id字典路径
    """字符 to id，保存word2id字典"""
    with open(vocab_path, 'r', encoding='utf-8', errors='ignore') as f:
        words = [ _.strip() for _ in f.readlines()]
    word2id = dict(zip(words,range(len(words)))) # # {字：1，哈：2}
    # 保存映射字典
    tmp = json.dumps(word2id)
    with open(word2id_path, 'w') as f:
        f.write(tmp)
    return words, word2id

def cate2id(labels,cate2id_path,cate_path): 
    """将标签转为id，保存字典,同时保存类别信息"""
    cates = list(set(labels))
    cate2id = dict(zip(cates, range(len(cates))))
    # 类别-id 映射表
    tmp = json.dumps(cate2id)
    with open(cate2id_path, 'w') as f:
        f.write(tmp)
    # 类别信息
    with open(cate_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cates) + '\n')  # '你\n好\n吗\n我\n好\n'
    return  cates, cate2id

def id2word(content, texts):
    """数字序列转字符"""
    return ''.join(texts[i] for i in content)

def load_dict(path):
    """加载映射表json文件为字典"""
    with open(path, 'r') as f:
        dic = json.load(f)
    return dic

def load_cate(path):
    """加载类别信息"""
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        cates = [ _.strip() for _ in f.readlines()]
    return cates


import tensorflow.contrib.keras as kr

def preprocess(texts,labels, word2id, cat2id, max_len=600):
    """ 将字符序列转为数字id序列"""
    text_id, label_id = [], []
    for i in range(len(texts)):
        text_id.append(list(word2id[x] for x in texts[i] if x in word2id)) # 字符转id
        label_id.append([cat2id[labels[i]]]) # 标签
    # 转为等长的序列
    x_pad = kr.preprocessing.sequence.pad_sequences(text_id, max_len) # maxlen：None或整数，为序列的最大长度。大于此长度的序列将被截短，小于此长度的序列将在后部填0.
    y_pad = kr.utils.to_categorical(label_id, num_classes = len(cat2id)) # 将整型标签转为onehot

    return x_pad, y_pad


import numpy as np

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    rows = len(x)
    num_batch = int((rows - 1) / batch_size) + 1
    indices =np.random.permutation(np.arange(rows)) # permutation返回打乱顺序的数组
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start = i*batch_size
        end = min((i+1)*batch_size, rows)
        yield x_shuffle[start:end], y_shuffle[start:end]  ####### 生成批次数据，迭代器，但迭代完完就没了！！！！




if __name__ == "__main__":
    path = "data\\train_data.txt"
    texts, labels = read_file(path)
    print("文本：",texts[:1])
    print("标签：",labels[:1])