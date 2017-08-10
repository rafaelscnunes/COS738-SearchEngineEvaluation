#!/Library/Frameworks/Python.framework/Versions/3.6/bin/Python3.6
# -*- coding: utf-8 -*-
"""
Created on 04/Jul/2017 with PyCharm Community Edition
@title:  IMIR - vsm.py
@author: rafaenune - Rafael Nunes - rnunes@cos.ufrj.br

"""


import re
import os
import math
import logging as log
from pprint import pprint
from nltk.stem.porter import *


f_log = __file__.split('.')[0]+'.log'
CONFIG_FILE = 'LOG_ps.CFG'
if os.path.isfile(CONFIG_FILE):
    for line in open(CONFIG_FILE, 'r'):
        if line.rstrip('\n').split('=')[0] == 'LOG_FILE':
            f_log = line.rstrip('\n').split('=')[1]
            break
        else:
            print('Invalid parameter found reading configuration.')
    else:
        print('Error reading configuration files!')
log.basicConfig(level = log.DEBUG,
                format = '%(asctime)s|%(levelname)s|%(name)s|%(funcName)s'
                         '|%(message)s',
                filename = f_log,
                filemode = 'a')
logger = log.getLogger(__file__.split('/')[-1])


def tokenizer(corpus = str('no content detected'), stop_words = list(),
                 min_word_length = int(2)):
    """ Get str() and return list() of uppercase only letters tokens.
        options:
                stop_words - list of stop words
                min_word_length - minimum lenght to became a token
    """
    # logger.info('Extracting tokens from text corpus...')
    # logger.info('Tokenizing only longer than %d words, converting to '
    #             'uppercase only letters, without punctuation nor symbols...'
    #             % min_word_length)
    # logger.info('Using this stop_words: %s' % stop_words)
    words = re.sub('[^a-zA-Z]', ' ', corpus)
    words = words.split()
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words
             if not word in stop_words
             and len(word) >= min_word_length]
    # logger.info('Finished. %d tokens extracted.' % len(words))
    return(words)


def get_corpora(file = 'corpora.csv', sep = ';', encoding = 'utf-8'):
    """ Get str('file_name') and return dict({'di': 'text content'}
        First line containing column headers
        2 columns: Corpus;Text
        options:
                file - file_name of document collection
                       default: corpora.csv
                sep - .csv field separator
                      default: ';'
                encoding - file encoding
                           default: utf-8
    """
    logger.info('Reading document collection from %s' % file)
    try:
        corpora = {}
        with open(file, 'r', encoding = encoding) as f_in:
            header = f_in.readline().split(sep)
            for line in f_in:
                corpus = line.strip('\n').split(sep)[0]
                content = line.strip('\n').split(sep)[1]
                corpora[corpus] = content
        f_in.close()
        logger.info('Corpora built with %d corpus' % len(corpora))
        return(corpora)
    except OSError:
        logger.info('Failed to read from %s' % file)


def itf_corpora(content = {'doc':'no content detected'}, stop_words = [],
              min_word_length = 2):
    """ Get a dictionary with {dj: 'text'} and returns another
        dictionary {ki : {dj : f_ij}} with number of times the term ki
        appears on document dj.
    """
    logger.info('Generating itf table (inverse term frequency)...')
    itf = {}
    for doc in content:
        words = tokenizer(content[doc])
        for word in words:
            if word not in itf:
                itf[word] = {}
            if doc not in itf[word]:
                itf[word][doc] = 1
            else:
                itf[word][doc] += 1
    logger.info('idf table generated with %d words' % len(itf))
    return(itf)


def max_freq_vector(content = {'doc' : {'word' : 0}}, ):
    """ Get a dictionary {'dj' : {'ki' : f_ij}} and returns another
    dictionary {'dj' : """
    logger.info('Calculating column vector max_freq, the maximum frequency '
                'of any given term that appears on the document dj...')
    max_freq = {}
    for doc, words in content.items():
        for word in words:
            if doc not in max_freq:
                max_freq[doc] = content[doc][word]
            else:
                if content[doc][word] > max_freq[doc]:
                    max_freq[doc] = content[doc][word]
    logger.info('Column vector max_freq calculated for %d documents'
                % len(max_freq))
    return(max_freq)


def tf_corpora(content = {'doc':'no content detected'}, stop_words = [],
                 min_word_length = 2):
    """ Get a dictionary {dj : 'text'} and returns another
        dictionary {dj : {ki : freq_ij}} with term (ki) frequency by
        document (dj).
    """
    logger.info('Generating tf table...')
    tf = {}
    for doc in content:
        words = tokenizer(content[doc])
        for word in words:
            if doc not in tf:
                tf[doc] = {}
            if word not in tf[doc]:
                tf[doc][word] = 0
            tf[doc][word] += 1
    logger.info('tf table generated %d document vectors' % len(tf))
    return(tf)


def tfn_corpora(content = {'doc':'no content detected'}, stop_words = [],
                 min_word_length = 2, weight1 = 0, weight2 = 1):
    """ Get a dictionary with {dj: 'text'} and returns another
        dictionary {dj : {ki : f_ij}} with normalized terms frequency (f_ij)
        by dj. Normalization is calculated based on the maximum frequency of
        term ki in a given document dj.

        weight1 & weight2: adjusts for normalization.
        for queries: weight1 = 0.5 and weight2 = 0.5
        for corpora: weight1 = 0 and weight2 = 1
    """
    tf = tf_corpora(content)
    max_freq = max_freq_vector(tf)
    logger.info('Generating tfn table...')
    tfn = {}
    for doc, words in tf.items():
        for word in words:
            if doc not in tfn:
                tfn[doc] = {}
            if word not in tfn[doc]:
                tfn[doc][word] = {}
            tfn[doc][word] = weight1 + (weight2*(tf[doc][word]/max_freq[doc]))
    logger.info('tfn table generated with %d document vectors' % len(tfn))
    return(tfn)


def idf_corpora(content = {'doc':'no content detected'}, stop_words = [],
                 min_word_length = 2):
    """ Get a dictionary with {dj : 'text'} and returns another dictionary
        {ki : log(N/ni)} with inverse document frequency by word.
    """
    logger.info('Generating idf vector (inverse document frequency')
    itf = itf_corpora(content)
    logger.info('Calculating line vector ni, number of documents where term '
                'ki appears')
    ni = {}
    for word in itf:
        ni[word] = len(itf[word])
    logger.info('Line vector ni calculated for %d words' % len(ni))

    idf = {}
    N = len(tfn_corpora(content))
    for word in ni:
        idf[word] = math.log(N/ni[word])
        # idf[word] = 1 + math.log(N/ni[word])
    logger.info('idf line vector generated with %d words' % len(idf))
    return(idf)


def idf_tokenized(tokenized_corpora):
    """ Implements idf_values using formula 2.2 of chapter 2 of book Modern
        Information Retrieval: idf_j = log(N/nj), where:
        N - is the total number of corpus in the corpora
        nj - number of corpus where kj (term) appears
        log - is the natural logarithm
    """
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_corpora for item in
                          sublist])
    # print(all_tokens_set)
    # print(len(all_tokens_set))
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_corpora)
        idf_values[tkn] = math.log(len(tokenized_corpora)/(sum(
                contains_token)))
        # idf_values[tkn] = 1 + math.log(len(tokenized_corpora)/(sum(
        #         contains_token)))
    return idf_values


def read_inverse_index_to_tf(file, sep = ';', min = 2):
    """ Get .csv with an inverse index and
        return tf_table in a dictionary {'di' : {'kj' : freq_ij}}
    """
    logger.info('Reading inverted list from %s...' % file)
    inverted_list = {}
    for line in open(file, 'r'):
        if line.lower() != 'word;documents\n':
            word = line.rstrip('\n').split(sep)[0]
            word = word
            # word = re.sub('[^A-Z]', '', word)
            # [as palavras não estão mais em maiúsculas por causa do
            # PorterStemmer que considera o radical apenas em minúsculas.]
            docs = line.rstrip('\n').split(sep)[1].lstrip('[').rstrip(']')\
                   .replace(' ', '').split(',')
            if len(word) >= min:
                inverted_list[word] = docs
    logger.info('%d words read into inverted list.' % len(inverted_list))
    logger.info('Building tf (term frequency) from inverted list...')
    tf = {}
    for word, docs in inverted_list.items():
        for doc in docs:
            if doc not in tf:
                tf[doc] = {}
            if word not in tf[doc]:
                tf[doc][word] = 1
            else:
                tf[doc][word] += 1
    logger.info('inverse TF built with %d words imported from inverted '
                'index into it.' % len(tf))
    return(tf)


def tf_to_itf(tf = {'doc' : {'word' : 0}}):
    """ Get {'di' : {'kj' : freq_ij}} and returns {'kj' : {'di' : freq_ij}}
    """
    itf = {}
    logger.info('Transposing tf to itf...')
    for doc, words in tf.items():
        for word in words:
            if word not in itf:
                itf[word] = {}
            if doc not in itf[word]:
                itf[word][doc] = {}
            itf[word][doc] = tf[doc][word]
    logger.info('Transposed.')
    return(itf)


def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)


def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)


# def augmented_term_frequency(term, tokenized_document):
#     max_count = max([term_frequency(t, tokenized_document)
#                      for t in tokenized_document])
#     return (0.5 + ((0.5 * term_frequency(term,
#                     tokenized_document))/max_count))


def weighted_term_frequency(term, tokenized_document, weight = 0.5):
    max_count = max([term_frequency(t, tokenized_document)
                     for t in tokenized_document])
    return (weight + ((weight * term_frequency(term,
                       tokenized_document))/max_count))


def maximum_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document)
                     for t in tokenized_document])
    return (term_frequency(term, tokenized_document)/max_count)


def normalize_tf(tf = {}, norm = 'max', weight = 0.5):
    """ Normilize tf into tfn using function norm
        norm: normalization function used to evaluate term frequency:
            . sublinear
            . augmented
            . weighted - when weight = 0.5 (augmented)
            . max (default)
        weight: weight used by normalization function, default = 0.5
    """
    max_freq = max_freq_vector(tf)
    logger.info('Normalizing tf to tfn...')
    tfn = {}
    for doc, words in tf.items():
        for word in words:
            if doc not in tfn:
                tfn[doc] = {}
            if word not in tfn[doc]:
                tfn[doc][word] = {}
            if norm == 'sublinear':
                tfn[doc][word] = 1 + math.log(tf[doc][word])
            elif norm == 'augmented':
                tfn[doc][word] = (0.5 + ((0.5 * tf[doc][word])/max_freq[doc]))
            elif norm == 'weighted':
                tfn[doc][word] = (weight + (weight * tf[doc][word])
                                  /max_freq[doc])
            else:
                tfn[doc][word] = (tf[doc][word] / max_freq[doc])
    logger.info('Transformation finished.')
    return(tfn)


def tf_idf(corpora, mode = 'dense', norm = 'max', weight1 = 0, weight2 = 1):
    """ TF_IDF (Term Frequency x Inverse Document Frequency)
        corpora: {'di' : 'text content'}
        mode:
            . dense (default) = use dict() to store values, no zero values
                                stored
            . sparse (optional) = use list() to store all values, including
                                  zeros
        norm: normalization function used to evaluate term frequency:
            . sublinear
            . augmented
            . weighted - when weight = 0.5 (augmented)
            . max (default)
        * norm is implemented only for sparse mode
        weight: weight used by normalization function, default = 0.5

    """
    if mode == 'dense':
        logger.info('Dense mode selected')
        logger.info('Generating idf of the corpora using idf_corpora()...')
        idf = idf_corpora(corpora)
        logger.info(('idf_vector calculated for %d corpus.' % len(idf)))
        # pprint(idf)

        logger.info('Generating tnf of the corpora using tnf_corpora()...')
        tfn = tfn_corpora(corpora, weight1 = weight1, weight2 = weight2)
        logger.info(('tfn_table calculated for %d corpus.' % len(tfn)))
        # pprint(tfn)

        logger.info('Building tfidf...')
        tfidf = {}
        for corpus, terms in tfn.items():
            for term in terms:
                if corpus not in tfidf:
                    tfidf[corpus] = {}
                if term not in tfidf[corpus]:
                    tfidf[corpus][term] = 0
                tfidf[corpus][term] = tfn[corpus][term] * idf[term]
        logger.info('tfidf built.')
        return (tfidf)

    elif mode == 'sparse':
        logger.info('Sparse mode selected')
        logger.info('Tokenizing corpora...')
        tokenized_corpora = [tokenizer(corpora[corpus]) for corpus in corpora]
        logger.info('%d corpus tokenized in corpora' % len(tokenized_corpora))

        logger.info('Generating idf of the corpora using idf_tokenized()...')
        idf = idf_tokenized(tokenized_corpora)
        logger.info(('idf_table calculated for %d corpus.' % len(idf)))
        # pprint(idf)
        # pprint(len(idf))

        logger.info('Building tfidf...')
        tfidf_documents = []
        for document in tokenized_corpora:
            doc_tfidf = []
            for term in idf.keys():
                if norm == 'sublinear':
                    tf = sublinear_term_frequency(term, document)
                elif norm == 'augmented':
                    tf = weighted_term_frequency(term, tokenized_corpora,
                                                 0.5)
                elif norm == 'weighted':
                    tf = weighted_term_frequency(term, tokenized_corpora,
                                                 weight)
                else:
                    tf = maximum_term_frequency(term, tokenized_corpora)
                doc_tfidf.append(tf * idf[term])
            tfidf_documents.append(doc_tfidf)
        logger.info('tfidf built.')
        return tfidf_documents

    else:
        logger.info('Invalid mode indicated... Nothing done.')
        return(1)

def cosine_similarity(vector1, vector2):
        dot_product = sum(p * q for p, q in zip(vector1, vector2))
        magnitude = math.sqrt(sum([val ** 2 for val in vector1])) * math.sqrt(
            sum([val ** 2 for val in vector2]))
        if not magnitude:
            return(0)
        return(dot_product / magnitude)


def cos_similarity(dict1, dict2):
    dot_product = 0
    for t1 in dict1:
        for t2 in dict2:
            if t1 == t2:
                dot_product += dict1[t1]*dict2[t2]
    magnitude = math.sqrt(sum([dict1[term]**2 for term in dict1]))\
                * math.sqrt(sum([dict2[term]**2 for term in dict2]))
    if not magnitude:
        return (0)
    return(dot_product/magnitude)