# -*- coding: utf-8 -*-
"""
    Utilities for processing text data.
"""

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

stop_words = stopwords.words('english')
ps = PorterStemmer()

from utils import flatten_list

def standard_processing(text):
    """
        Standard processes to clean a set of text. Tokenize, lower case,
        remove non-alphabetic characeters and stop words, and stem.
    """
    clean_tokens = [w.lower() for w in nltk.word_tokenize(text) if w.isalpha()]
    return [ps.stem(w) for w in clean_tokens if not w in stop_words]
    
def make_corpus(file_list, process_fun = standard_processing):
    """
        Create a dictionary of documents from a list of file names. Apply
        a processing function to the raw text if desired.
    """
    corpus = dict()
    for file in file_list:
        with open(file) as f:
            txt = f.read()
        fname = file.split('\\')[-1].split('.')[0]
        corpus[fname] = process_fun(txt)
    return corpus

def calc_tf_idf(corpus):
    """
        Given a text corpus (dictionary of name/words), calculate the
        TF-IDF score for each document.
    """
    n_documents = len(corpus)
    all_words = flatten_list(corpus.values())
    unique_words = set(all_words)
    
    # Calculate document frequency
    df = dict()
    for w in unique_words:
        df[w] = sum([1 for k in corpus if w in corpus[k]])
    
    n_words = len(df)
    
    # Calculate IDF
    idf = np.array([np.log(n_documents/df[w]) for w in unique_words]).transpose()
    
    # Calculate TF-IDF for each document
    tfidf = dict()
    for key, text in corpus.items():
        # Use the document frequency order to make sure interpretation is consistent
        tf = np.array([sum([1 for v in text if v==w]) for w in unique_words]).transpose()
        tfidf[key] = np.multiply(tf,idf)
    
    return tfidf
    
def calc_cosine_similarity(tfidf, norm=True):
    """
        Given a dictionary of tfidf vectors, construct a diffusion matrix using
        cosine similarity. The order will follow the items() order of the
        input.

    """
    V = list(tfidf.values())
    cosine_similarity = lambda a,b: np.dot(a,b)/np.sqrt(np.dot(a,a)*np.dot(b,b))
    n_documents = len(tfidf)
    D = np.zeros([n_documents,n_documents])+-np.Inf
    for i in range(n_documents):
        for j in range(i,n_documents):
            D[i,j] = cosine_similarity(V[i],V[j])
    
    # Symmetrize the matrix and norm if desired
    D = np.maximum(D,D.transpose())
    if norm:
        Dnorm = np.repeat(np.matrix(np.sum(D,axis=0)),np.size(D,0),axis=0)
        D = np.multiply(D, 1./Dnorm)
    return D
