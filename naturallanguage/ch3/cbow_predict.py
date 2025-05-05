
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
import numpy as np
from common.layers import MatMul
from common.utils import  create_contexts_target
from common.utils import preprocess
from common.utils import convert_one_hot
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)
layer_in = MatMul(W_in)
layer_out = MatMul(W_out)
h0 = layer_in.forward(c0)
h1 = layer_in.forward(c1)
h = 0.5 * (h0 + h1)
s = layer_out.forward(h)
print(h)
text = 'You say goodbye and I say hello.'
##语料库的准备
corpus,wordtoId,idtoWord=preprocess(text)
contexts,target=create_contexts_target(corpus,1)
print(contexts)
print(target)
vecob_size=len(wordtoId)
print("-------------------------------------------")
print(convert_one_hot(contexts,vecob_size))
print(convert_one_hot(target,vecob_size))
 