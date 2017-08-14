#!/Library/Frameworks/Python.framework/Versions/3.6/bin/Python3.6
# -*- coding: utf-8 -*-
"""
Created: 2017-07-05
@title:  IMIR-VSM (In Memory Information Retrieval - Vector Space Model)
@module: indexer.py
@author: Rafael Nunes - rnunes@cos.ufrj.br
"""
# Indexador - A função desse módulo é criar o modelo vetorial, dadas as listas
#             invertidas simples.
# DONE: 1) O indexador será configurado por um arquivo INDEX.CFG
# a. O arquivo conterá apenas uma linha LEIA, que terá o formato
#   i. LEIA=<nome de arquivo>
# b. O arquivo conterá apenas uma linha ESCREVA, que terá o formato
#   i. ESCREVA=<nome de arquivo>

# DONE: 2) O Indexador deverá implementar um indexador segundo o Modelo
#          Vetorial [A MAIOR PARTE FOI IMPLEMENTADA NO inverted.py]
# a. O Indexador deverá utilizar o tf/idf padrão
#   i. O tf pode ser normalizado como proposto na equação 2.1 do Cap. 2 do
#      Modern Information Retrieval
# b. O indexador deverá permitir a alteração dessa medida de maneira simples
# c. O Indexador deverá possuir uma estrutura de memória deve de alguma forma
#    representar a matriz termo documento
# d. O Indexador deverá classificar toda uma base transformando as palavras
#    apenas da seguinte forma:
#   i. Apenas palavras de 2 letras ou mais
#  ii. Apenas palavras com apenas letras
# iii. Todas as letras convertidas para os caracteres ASCII de A até Z,
#      ou seja, só letras maiúsculas e nenhum outro símbolo
# e. A base a ser indexada estará na instrução LEIA do arquivo de configuração

# DONE: 3) O sistema deverá salvar toda essa estrutura do Modelo Vetorial para
# utilização posterior

# DONE: 4) Todos os módulos devem possuir um LOG que permita pelo menos a um
# programador posterior, usando o módulo logging de Python:
# 1. Identificar quando iniciaram suas operações
# 2. Identificar quando iniciam cada parte de seu processamento
# a. Ler arquivo de configuração
# b. Ler arquivo de dados
# 3. Identificar quantos dados foram lidos
# 4. Identificar quando terminaram os processamentos
# 5. Calcular os tempos médios de processamento de consultas, documento e
#    palavras, de acordocom o programa sendo usado
# 6. Identificar erros no processamento, caso aconteçam.

import vsm_noStemmer as vsm
import os
import math
import logging as log
import pickle
from pprint import pprint


logger = log.getLogger(__file__.split('/')[-1])
CORPORA_FILE = './out/corpora_noStemmer.csv'
CONFIG_FILE = 'indexer_noStemmer.cfg'
SEP = ';'
ENCOD = 'utf-8'
MIN_WORD_LENGTH = 2


logger.info('Started %s' % __file__.split('/')[-1])
if os.path.isfile(CONFIG_FILE):
    logger.info('Reading configuration from ' + CONFIG_FILE + '...')
    for line in open(CONFIG_FILE, 'r'):
        if line.rstrip('\n').split('=')[0] == 'LEIA':
            f_leia = line.rstrip('\n').split('=')[1]
        elif line.rstrip('\n').split('=')[0] == 'ESCREVA':
            f_escreva = line.rstrip('\n').split('=')[1]
            logger.info('Gracefully stopped reading configuration file ' +
                        CONFIG_FILE + ', ESCREVA parameter found.')
            break
        else:
            logger.error('Invalid parameter found reading configuration. ')
    if f_leia and f_escreva:
        logger.info('All set! Configuration successfully read!')
    else:
        logger.error('Error reading configuration files!')

    logger.info('Creating Vector Space Model...')
    tf = vsm.read_inverse_index_to_tf(f_leia, SEP, MIN_WORD_LENGTH)
    # print(tf)
    # print(len(tf))
    tf_norm = vsm.normalize_tf(tf, norm = 'max', weight = 0.6)
    # pprint(tf_norm)
    # pprint(len(tf_norm))
    logger.info('Evaluating ni...')
    inv_tf = vsm.tf_to_itf(tf)
    ni = {}
    for word in inv_tf:
        ni[word] = len(inv_tf[word])
    # print(ni)
    # print(len(ni))
    logger.info('Evaluated ni for %d terms' % len(ni))
    logger.info('Building idf...')
    idf = {}
    N = len(tf_norm)
    for word in ni:
        idf[word] = math.log(N/ni[word])
    # print(idf)
    # print(len(idf))
    logger.info('idf built for %d terms' % len(idf))
    logger.info('Creating VSM (w_ij) from inverse index at %s...' % f_leia)
    w_ij = {}
    for doc, words in tf_norm.items():
        for word in words:
            if doc not in w_ij:
                w_ij[doc] = {}
            if word not in w_ij[doc]:
                w_ij[doc][word] = {}
            w_ij[doc][word] = tf_norm[doc][word]*idf[word]
    # print(w_ij)
    # print(len(w_ij))
    logger.info('w_ij built with %d corpus.' % len(w_ij))
    logger.info('Creating VSM (tf_idf) from corpora at %s...' % CORPORA_FILE)
    corpora = vsm.get_corpora(CORPORA_FILE, SEP, ENCOD)
    tf_idf = vsm.tf_idf(corpora, mode = 'dense', norm = 'max')
    # print(tf_idf)
    # print(len(tf_idf))
    logger.info('tf_idf built with %d corpus.' % len(tf_idf))
    logger.info('Vector Space Model created!')
    logger.info('Saving VSM (w_ij & tf_idf)')
    pickle_out = open(f_escreva,'wb')
    pickle.dump([w_ij, tf_idf], pickle_out)
    pickle_out.close()
    logger.info(('VSM (w_ij & tf_idf) saved at %s.' % f_escreva))


    # print(w_ij['1236']['WITH'])
    # aux = w_ij['1236']['WITH']
    # if w_ij == tf_idf:
    #     print('The dictionaries are identical')
    # else:
    #     print('The dictionaries are different!')
    # w_ij['1236']['WITH'] = aux - 0.0000000004
    # print(w_ij['1236']['WITH'])
    # if w_ij == tf_idf:
    #     print('The dictionaries are identical')
    # else:
    #     print('The dictionaries are different!')
    # w_ij['1236']['WITH'] = aux
    # print(w_ij['1236']['WITH'])
    # if w_ij == tf_idf:
    #     print('The dictionaries are identical')
    # else:
    #     print('The dictionaries are different!')

    # from sklearn.feature_extraction.text import TfidfVectorizer
    # sklearn_tfidf = TfidfVectorizer(norm = 'l2', min_df = 0, use_idf = True,
    #                                 smooth_idf = False, sublinear_tf = True,
    #                                 tokenizer = vsm.tokenizer())
    # sklearn_representation = sklearn_tfidf.fit_transform([
    #     text[corpus] corpus in corpora])


    logger.info('Finished %s' % __file__.split('/')[-1])
else:
    logger.error(CONFIG_FILE + ' not found!')
    print(CONFIG_FILE + ' not found! Execution aborted.')
    logger.error('Execution aborted.')
