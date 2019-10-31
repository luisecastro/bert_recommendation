# Author: Luis Castro 2019
# While going throught Stanford's NLU course, I plan to write every function in my own way 
# to help familiarize with the math behind them
# they are a lot faster based on preliminary walltime runs

import numpy as np
import pandas as pd

###### Pending to optimize

def neighbors(word: str, df: pd.DataFrame, distfunc: object) -> pd.Series:
    if word not in df.index:
        raise ValueError('{} is not in this VSM'.format(word))
    w = df.loc[word]
    dists = df.apply(lambda x: distfunc(w, x), axis=1)
    return dists.sort_values()


def ngram_vsm(df, n=2):
    unigram2vecs = defaultdict(list)
    for w, x in df.iterrows():
        for c in char_ngrams(w, n):
            unigram2vecs[c].append(x)
    unigram2vecs = {c: np.array(x).sum(axis=0)
                    for c, x in unigram2vecs.items()}
    cf = pd.DataFrame(unigram2vecs).T
    cf.columns = df.columns
    return cf


def character_level_rep(word, cf, n=4):
    ngrams = char_ngrams(word, n)
    ngrams = [n for n in ngrams if n in cf.index]    
    reps = cf.loc[ngrams].values
    return reps.sum(axis=0)    
######

def pd2np(func):
    def inner(df):
        if isinstance(df, pd.DataFrame):
            columns = df.columns
            index = df.index
            return pd.DataFrame(func(df.values), columns=columns, index=index)
        else:
            return func(df)
    return inner    


def euclidean(u: list, v: list) -> float:
    return np.sqrt(np.sum(np.power(np.subtract(u, v), 2)))


def cosine_similarity(u: list, v: list) -> float:
    return np.divide(np.sum(np.multiply(u, v)), np.multiply(np.sqrt(np.sum(np.power(u, 2))), np.sqrt(np.sum(np.power(v, 2)))))


def cosine_distance(u: list, v: list) -> float:
    return np.subtract(1, cosine_similarity(u, v))


def matching(u: list, v: list) -> float:
    return np.sum(np.minimum(u, v))


def jaccard(u: list, v: list) -> float:
    return np.subtract(1, np.divide(matching(u, v), np.sum(np.maximum(u, v))))


def dice(u: list, v: list) -> float:
    return np.subtract(1, np.divide(np.multiply(2, matching(u, v)), np.sum([u, v])))


def overlap(u: list, v: list) -> float:
    return np.divide(matching(u, v), np.min([np.sum(u), np.sum(v)]))


@pd2np
def observed_expected(u: list) -> list:
    return np.divide(np.multiply(np.sum(u), u), np.multiply(np.sum(u, axis=1).reshape(u.shape[0], 1), np.sum(u, axis=0).reshape(1, u.shape[1])))


@pd2np
def pmi(u: list) -> list:
    return np.log(observed_expected(u))


@pd2np
def ppmi(u: list) -> list:
    return np.nan_to_num(np.maximum(pmi(u), 0), posinf=0)


@pd2np
def term_freq(u: list) -> list:
    return np.divide(u, np.sum(u, axis=1))


@pd2np
def inv_doc_freq(u: list) -> list:
    docs = u.shape[1]
    freqs = np.sum(u.astype(bool), axis=1)
    return np.nan_to_num(np.log(np.divide(docs, freqs)), posinf=0)


@pd2np
def tfidf(u: list) -> list:
    return np.transpose(np.multiply(term_freq(u), inv_doc_freq(u)))


@pd2np
def normalize(u: list) -> float:
    return np.divide(u, np.sqrt(np.dot(u, u)))


def char_ngrams(w: str, n: int) -> list:
    w = list(w)
    if n > 1:
        w.insert(0, '<w>')
        w.append('</w>')
    return [''.join(w[i: i+n]) for i in range(len(w) - n+1)]