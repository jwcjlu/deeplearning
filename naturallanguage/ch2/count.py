
import imp
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from   common.utils import preprocess
from common.utils import create_co_matrix
from common.utils import cos_similarity
from common.utils import most_similar
from common.utils import ppmi
import matplotlib.pyplot as plt
text = 'You say goodbye and I say hello.'
##语料库的准备
corpus,wordtoId,idtoWord=preprocess(text)
print(corpus)
print(wordtoId)
print(idtoWord)
coMatrix=create_co_matrix(corpus,len(wordtoId))
print(coMatrix)
x=coMatrix[wordtoId['you']]
y=coMatrix[wordtoId['i']]
print(cos_similarity(x,y))

most_similar('you',wordtoId,idtoWord,coMatrix,top=5)

ppmi=ppmi(coMatrix)
print(ppmi)
U,S,V=np.linalg.svd(ppmi)
print(coMatrix[0])
print(ppmi[0])
print(U[0])
for word, word_id in wordtoId.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()