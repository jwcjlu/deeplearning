# -*- coding: utf-8 -*-
# 或
# coding: utf-8
import sys
import io

# 替换标准输出为 GBK 编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from nltk.corpus import wordnet as wn
synsets = wn.synsets('car', pos='n')  # 获取所有名词义项
print(synsets)
synset = wn.synset('car.n.01')

print("定义:", synset.definition())          # 定义
print("例子:", synset.examples())            # 例句
print("同义词:", [lemma.name() for lemma in synset.lemmas()])  # 同义词列表