# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os
import nltk
import nltk.corpus

print(os.listdir(nltk.data.find("corpora")))
nltk.corpus.gutenberg.fileids()
hamlet=nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')
hamlet
for word in hamlet [:1000]:
    print(word, sep='',end='')  

ram="ramesh Is a good boy, he is working in CRC pharma"
ram

type(ram)

from nltk.tokenize import word_tokenize


len(Ram1)

from nltk.probability import FreqDist
fdis=FreqDist()
fdis

for word in Ram1:
    fdist[word.lower()]+=1
fdist


mostcommon=fdist.most_common(10)
mostcommon

from nltk.tokenize import blankline_tokenize
Ram2=blankline_tokenize(Ram1)
Ram2

from nltk.util import bigrams, trigrams, ngrams

kavya='Kavya is born on 1996, and now she is working in capgemini'
kavya1=nltk.word_tokenize(kavya)
kavya1

len(kavya1)

kavya2=list(nltk.bigrams(kavya1))
kavya2

kavya2=list(nltk.trigrams(kavya1))
kavya2

kavya2=list(nltk.ngrams(kavya1,5))
kavya2


from nltk.stem import PorterStemmer
pst=PorterStemmer()

pst.stem('having')

#Lammitizer#

from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
RamL=WordNetLemmatizer()

RamL.lemmatize('giving','given','give')

#Stop words

from nltk.corpus import stopwords

stopwords.words('english')
len(stopwords.words('english'))

fdist(stopwords.words('english'))

fdis('kavya')

FreqDist('kavya')
FreqDist_top10


import re
punc=re.compile(r'[-?!;:()\|0-9]')

ramesh="ramesh,given!,gone.has given a milk in morning"


post_punctuation=[]
    for words in ramesh:
    words=punc.sub("",words)
    if len(word)>0:
        post_punctuation.append(word)

post_punctuation
len(post_punctuation)

#POS tagging
RamP='ramesh is having a duke bike'
Token1=word_tokenize(RamP)
Token1

for token in Token1:
    print(nltk.pos_tag([token]))

#SDTM-Mapper    
!pip install sas7bdat tensorflow-hub pathlib sdtm-mapper
pip install sas7bdat tensorflow-hub pathlib sdtm-mapper
pip install sdtm-mapper
pip install tensorflow

import tensorflow
import sas7bdat
import pathlib
import sdtm_mapper

import sdtm_mapper.SDTMMapper as mapp


createspec=sdtm_mapper.SDTMMapper('ae', False, 'ae.sas7bdat')

#####
!pip install sas7bdat tensorflow-hub pathlib
!pip install sdtm-mapper

import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras import backend as K
from keras.callbacks import ModelCheckpoint

# Here you import sdtm_mapper
import sdtm_mapper.SDTMModels as sdtm
import sdtm_mapper.SDTMMapper
from sdtm_mapper import samples

bucket='snvn-sagemaker-1' #data bucket
KEY='mldata/Sam/data/project/xxx-000/xxx/xxx-201/csr/data/raw/latest/'
# Initialize session





