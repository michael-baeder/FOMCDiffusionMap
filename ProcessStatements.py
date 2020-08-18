# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:47:33 2020

@author: Michael
"""

import numpy as np
import io
import os
import datetime
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from scipy import linalg as LA

from utils import throttle


ps = PorterStemmer() 

# Move to the statements folder
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
os.chdir('..\\Statements')

# Load the statements
statement_files = [file for file in os.listdir() if file.endswith('.txt')]
n_occ = dict() # mapping of words to number of documents they occur in
doc_data = []
for file_name in statement_files:
    with open(file_name,'r') as file:
        r = file.read()
        # keep only alphabetical words, make all lower-case
        clean_tokens = [w.lower() for w in nltk.word_tokenize(r) if w.isalpha()]
        # remove stop words and stem
        no_stops = [ps.stem(w) for w in clean_tokens if not w in stop_words]
        # Add occurences to the dictionary
        for w in set(no_stops):
            if w not in n_occ:
                n_occ[w]=1
            else:
                n_occ[w]+=1   
        doc_data = doc_data + [ 
                    {'date': datetime.datetime.strptime(file_name[:-4], '%Y%m%d'),
                     'raw_text': r,
                     'cleaned_words': no_stops}]

# Calculate the TF-IDF vector for each document
n_documents = len(doc_data)
idf = np.array([np.log(n_documents/v) for v in n_occ.values()]).transpose()

n_words = len(n_occ)
for doc in doc_data:
    tf = np.array([sum([1 for v in doc['cleaned_words'] if v==w])
                    for w in n_occ]).transpose()
    doc['tfidf'] = np.multiply(tf,idf)
    
# Calculate the cosine similarity BETWEEN each document
cosine_similarity = lambda a,b: np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))
    
D = np.zeros([n_documents,n_documents])+-np.Inf
for i in range(n_documents):
    for j in range(i,n_documents):
        D[i,j] = cosine_similarity(doc_data[i]['tfidf'],doc_data[j]['tfidf'])
D = np.maximum(D,D.transpose())

Dnorm = np.repeat(np.matrix(np.sum(D,axis=0)),np.size(D,0),axis=0)

D = np.multiply(D, 1./Dnorm)
t = 0.05
V,W = LA.eig(np.power(D,t))