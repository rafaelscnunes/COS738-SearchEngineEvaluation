#!/Library/Frameworks/Python.framework/Versions/3.6/bin/Python3.6
# -*- coding: utf-8 -*-
"""
Created on 07/Aug/2017 with PyCharm Community Edition
@title:  Search Engine Evaluation - evaluation
@author: rafaenune - Rafael Nunes - rnunes@cos.ufrj.br

"""


# Exercício 2: Avaliação de Engine de Busca
# Nesse exercício você deverá fazer uma avaliação de dois engines de busca que
# construir.
#
# O engine de busca que construiu na versão anterior (sem Porter Stemmer)
# O engine de busca da versão anterior usando o Porter Stemmer.
#
# Você deve criar gráficos e tabelas comparativos dos dois engines com as
# seguintes medidas:
#
# DONE: Gráfico de 11 pontos de RecallxPrecision
# DONE: Medida F1 - Harmonic Mean (F-score / F-measure)
# DONE: MAP - Mean Average Precision
# DONE: P@5 e P@10 - Precision at k
# TODO: Gráfico RPA/B
# TODO: MRR
# TODO: Normalized DCG
# TODO: BPREF

import matplotlib.pyplot as plt
import operator

# Loading all results (with and without PorterStemer) and relevants
f_resultados = '../IMIR_VSM/resultados00.csv'
f_resultados_ps = '../With_PorterStemer/resultados_ps00.csv'
f_esperados = '../IMIR_VSM/esperados.csv'

queries = {}  # without PorterStemer
for line in open(f_resultados, 'r'):
    if line.lower() != 'querynumber;[rank, document, similarity]\n':
        line = line.rstrip('\n')
        query = int(line.split(';')[0])
        result = line.split(';')[1].lstrip('[').rstrip(']')\
                 .replace(' ', '').split(',')
        if query not in queries:
            queries[query] = {}
        if int(result[1]) not in queries[query]:
            queries[query][int(result[1])] = float(result[2])

queries_ps = {}  # with PorterStemer
for line in open(f_resultados_ps, 'r'):
    if line.lower() != 'querynumber;[rank, document, similarity]\n':
        line = line.rstrip('\n')
        query = int(line.split(';')[0])
        result = line.split(';')[1].lstrip('[').rstrip(']')\
                 .replace(' ', '').split(',')
        if query not in queries_ps:
            queries_ps[query] = {}
        if int(result[1]) not in queries_ps[query]:
            queries_ps[query][int(result[1])] = float(result[2])

esperados = {}  # Relevants
for line in open(f_esperados, 'r'):
    if line.lower() != 'querynumber;docnumber;docvotes\n':
        line = line.rstrip('\n')
        if int(line.split(';')[0]) not in esperados:
            esperados[int(line.split(';')[0])] = {}
        if int(line.split(';')[1]) not in esperados[int(line.split(';')[0])]:
            esperados[int(line.split(';')[0])][int(line.split(';')[1])] = \
                int(line.split(';')[2])


def relevant(doc, esperados):
    if doc in esperados:
        return 1
    else:
        return 0

def true_positive(a, b):
    """ Return quantity of TP - True Positives
        What is in A and B
        being A the set of Positive prediction
        and B the set of Actual Positive """

    tp = 0
    for item in a:
        if item in b:
            tp += 1
    return tp

def false_positive(a, b):
    """ Return quantity FP - False Positives
        What is in A and not in B
        being A the set of Positive prediction
        and B the set of Actual Positive """

    fp = 0
    for item in a:
        if item not in b:
            fp += 1
    return fp

def num_docs(a):
    """ Return a dict with all the documents returned in all
        queries """

    full_prediction_set = {}
    for item in a:
        for doc in a[item]:
            if doc not in full_prediction_set:
                full_prediction_set[doc] = 0
            full_prediction_set[doc] += 1
    return full_prediction_set

def true_negative(a, b, full_set):
    """ Return quantity TN - True Negative
            What is not in A and is not in B but is in full_set
            being A the set of Positive prediction
            B the set of Actual Positive
            and full_set the set of all documents in all search results """

    tn = 0
    neg = {}
    full_prediction_set = num_docs(full_set)
    for item in full_prediction_set:
        if item not in b:
            neg[item] = 1

    for item in neg:
        if item not in a:
            tn += 1
    return tn

def false_negative(a, b):
    """ Return quantity FN - False Negative
        What is not in A and is in B
        being A the set of Positive prediction
        and B the set of Actual Positive """

    fn = 0
    for item in b:
        if item not in a:
            fn += 1
    return fn

def interpolation(queries, recall_levels, esperados):
    precision = {}
    for level in range(0, len(recall_levels)):
        recall_level = recall_levels[level]
        precision[recall_level] = {}
        for query in queries:
            max_results = int(len(queries[query]) * recall_level)
            count = 0
            results = []
            for doc in queries[query]:
                count += 1
                if count <= max_results:
                    results.append(doc)
                else:
                    break
            TP = true_positive(results, esperados[query])
            if len(results) > 0:
                precision[recall_level][query] = TP / len(results)
            else:
                precision[recall_level][query] = 0
    # print(precision)

    for query in queries:
        for level in range(0, len(recall_levels)):
            if precision[recall_levels[level]][query] == 0:
                for r in range(level, len(recall_levels)):
                    if precision[recall_levels[r]][query] > \
                       precision[recall_levels[level]][query]:
                        precision[recall_levels[level]][
                            query] = precision[recall_levels[r]][query]
    # print(precision)

    avg_precision = {}
    for level in recall_levels:
        avg_precision[level] = 0
        for query in queries:
            avg_precision[level] += precision[level][query]
        avg_precision[level] /= len(queries)
    # print(avg_precision)
    return avg_precision

def precision_at(queries, rank, esperados):
    precision = {}
    for query in queries:
        count = 0
        results = []
        for doc in queries[query]:
            count += 1
            if count <= rank:
                results.append(doc)
            else:
                break
        TP = true_positive(results, esperados[query])
        if len(results) > 0:
            precision[query] = TP / len(results)
        else:
            precision[query] = 0
    # return precision

    avg_precision = 0
    for query in queries:
        avg_precision += precision[query]
    avg_precision /= len(queries)
    return avg_precision

def metrics(queries):
    TP_resultados = {}  # True Positive based on results without PorterStemer
    recall = {}  # Recall based on results without PorterStemer
    precision = {}  # Precision based on results without PorterStemer
    F1 = {}  # F1 based on results without PorterStemer
    for query in queries:
        TP_resultados[query] = true_positive(queries[query], esperados[query])
        recall[query] = TP_resultados[query] / len(esperados[query])
        precision[query] = TP_resultados[query] / len(queries[query])
        if TP_resultados[query] > 0:
            F1[query] = 2 * (precision[query] * recall[query]) /\
                            (precision[query] + recall[query])
    return(recall, precision, F1)

def avg(list):
    mean, count = 0, 0
    for i in list:
        count += 1
        mean += list[i]
    mean /= count
    return mean


# Gráfico de precisão em 11 níveis de revocação
# recall_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#                  0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
recall_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
grafico = interpolation(queries, recall_levels, esperados)
grafico_ps = interpolation(queries_ps, recall_levels, esperados)
plt.figure(1)
plt.plot(*zip(*sorted(grafico.items())), label = 'without PorterStemer')
plt.plot(*zip(*sorted(grafico_ps.items())), label = 'with PorterStemer')
plt.xlabel('Recall level')
plt.ylabel('Average Precision')
plt.title('Recall-Precision 11 points curve (all)')
plt.legend(loc = 'best')
plt.show()


# F1 measure
# F1 = 2 * (Precision * Recall) / (Precision + Recall), with B = 1
f1 = metrics(queries)[2]
f1_ps = metrics(queries_ps)[2]
print('F1 measure without PorterStemer: ' + str(avg(f1)))
print('F1 measure with PorterStemer: ' + str(avg(f1_ps)))


# MAP - Mean Average Precision
map = avg(grafico)
map_ps = avg(grafico_ps)
print('MAP without PorterStemer: ' + str(map))
print('MAP with PorterStemer: ' + str(map_ps))


# P@5 & P@10
p_at_5 = precision_at(queries, 5, esperados)
print('Precision @ 5 without PorterStemer: ' + str(p_at_5))
p_at_10 = precision_at(queries, 10, esperados)
print('Precision @ 10 without PorterStemer: ' + str(p_at_10))
p_at_5_ps = precision_at(queries_ps, 5, esperados)
print('Precision @ 5 with PorterStemer: ' + str(p_at_5_ps))
p_at_10_ps = precision_at(queries_ps, 10, esperados)
print('Precision @ 10 with PorterStemer: ' + str(p_at_10_ps))


# Gráfico RPA/B
