
# coding: utf-8

# In[21]:


import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jieba 
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
import multiprocessing
#get_ipython().magic('matplotlib inline')


# In[4]:


# set parameters:
vocab_dim = 200
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 100
cpu_count = multiprocessing.cpu_count()


# In[9]:


data=pd.read_csv('data/train_first.csv')
#data=data.drop(['Id'],axis=1)
#data.head()
#划分x  y
X=data['Discuss']
Y=data['Score']


# In[11]:


#加载停用词
def get_stopwords(path):
    return [line.strip() for line in open(path,'r',encoding='utf-8').readlines()]
#句子去停用词
def removestopwords(sentence):
        stopwords_list=get_stopwords('data/stopwords.txt')
        outstr=[]
        for word in sentence:
            if not word in stopwords_list:
                if word!='\n' and word!='\t':
                     outstr.append(word)
        return outstr


# In[12]:


#分词 并去掉停用词
def cut(sentence):
    return removestopwords(jieba.cut(sentence))
#分词后的word
cabs=[cut(x) for x in X]
#cabs[0]


# In[24]:


#word2vec

#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined,total_examples=model.corpus_count,epochs=model.iter)
    model.save('data/Word2vec.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')


# In[25]:


#根据词向量重新填充x
index_dict, word_vectors,combined=word2vec_train(cabs)


# In[ ]:


#搭建网络结构
input_x=tf.placeholder(shape=[None,None],dtype=tf.float32,name='input_x')
lable=tf.placeholder(shape=[None,None])


# In[ ]:


#训练


# In[ ]:


#预测

