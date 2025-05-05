from re import M
import numpy as np

# 预处理文本
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    wordToId = {}
    idToWord = {}
    for word in words:
        if word not in wordToId:
            new_id = len(wordToId)
            wordToId[word]  = new_id
            idToWord[new_id] = word

    corpus = [wordToId[w] for w in words]
    corpus = np.array(corpus)
    return corpus, wordToId, idToWord

# 计算共现矩阵
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix

def cos_similarity(x,y,eps=1e-8):
    nx=x/np.sqrt(np.sum(x**2)+eps)#x的正规化
    ny=y/np.sqrt(np.sum(y**2)+eps)#y的正规化
    return np.dot(nx,ny)

def most_similar(query,wordtoId,idtoWord,wordMatrix,top=5):
   #取出查询词
    if query not in wordtoId:
        print("%s is not found"%query)
        return
    print('\n[query] '+query)
    query_id=wordtoId[query]
    query_vec=wordMatrix[query_id]
    vocab_size=len(idtoWord)
    similarity=np.zeros(vocab_size)
    #计算余弦相似度
    for i in range(vocab_size):
        similarity[i]=cos_similarity(wordMatrix[i],query_vec)
    count=0
    #基于余弦相似度输出，按降序输出值
    for i in (-1*similarity).argsort():
        if idtoWord[i]==query:
            continue
        print(' %s: %s'%(idtoWord[i],similarity[i]))
        count+=1
        if count>=top:
            return
#正的点互信息Positive Pointwise MutualInformation
def ppmi(C,verbost=False,eps=1e-8):
    M=np.zeros_like(C,dtype=np.float32)
    N=np.sum(C)
    S=np.sum(C,axis=0)
    total=C.shape[0]*C.shape[1]
    cnt=0
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi=np.log2(C[i,j]*N/(S[i]*S[j])+eps)
            M[i,j]=max(0,pmi)
            if verbost:
                cnt+=1
                if cnt%(total//100)==0:
                    print('%.1f%% done'% (100*cnt/total))
    return M

def create_contexts_target(corpus,window_size=1):
    target=corpus[window_size:-window_size]
    contexts=[]
    for idx in range(window_size,len(corpus)-window_size):
        cs=[]
        for t in range(-window_size,window_size+1):
            if t==0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    return np.array(contexts),np.array(target)

def convert_one_hot(corpus,vocab_size):
    N=corpus.shape[0]
    if corpus.ndim==1:
        one_hot=np.zeros((N,vocab_size),dtype=np.int32)
        for idx,word_id in enumerate(corpus):
            one_hot[idx,word_id]=1
    elif corpus.ndim==2:
        C=corpus.shape[1]
        one_hot=np.zeros((N,C,vocab_size),dtype=np.int32)
        for idx_0,word_ids in enumerate(corpus):
            for idx_1,word_id in enumerate(word_ids):
                one_hot[idx_0,idx_1,word_id]=1
    return one_hot