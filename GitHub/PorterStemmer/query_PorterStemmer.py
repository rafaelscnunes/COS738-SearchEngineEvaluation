#!/Library/Frameworks/Python.framework/Versions/3.6/bin/Python3.6
# -*- coding: utf-8 -*-
"""
Created: 2017-08-07
@title:  IMIR-VSM (In Memory Information Retrieval - Vector Space Model)
@module: query_ps.py
@author: Rafael Nunes - rnunes@cos.ufrj.br
"""
# Processador de Consultas - O objetivo desse módulo é transformar o
#       arquivo de consultas fornecido no padrão de palavras que estamos
#       utilizando.


import vsm_PorterStemmer as vsm_ps
import os
import re
import logging as log
import xml.etree.cElementTree as ET



logger = log.getLogger(__file__.split('/')[-1])
CONFIG_FILE = 'query_PorterStemmer.cfg'
SEP = ';'
queries = []


class queryRecords:
    """ Classe para armazenar os registros lidos do .xml """

    def __init__(self):
        self.Number = 0
        self.Text = ''
        self.Results = 0
        self.Records = {}

    def __repr__(self):
        return '{}: {} {} {} {}'.format(self.__class__.__name__,
                                        self.Number,
                                        self.Text,
                                        self.Results,
                                        self.Records)


def computeVotes(votes):
    """Calcula o score de cada documento em cada consulta"""

    evaluation = 0
    for i in range(0, len(votes)):
        evaluation = evaluation + int(votes[i])
    return(str(evaluation))


logger.info('Started %s' % __file__.split('/')[-1])
if os.path.isfile(CONFIG_FILE):
    logger.info('Reading configuration from ' + CONFIG_FILE + '...')
    for line in open(CONFIG_FILE, 'r'):
        if line.rstrip('\n').split('=')[0] == 'LEIA':
            f_leia = line.rstrip('\n').split('=')[1]
        elif line.rstrip('\n').split('=')[0] == 'CONSULTAS':
            f_consultas = line.rstrip('\n').split('=')[1]
        elif line.rstrip('\n').split('=')[0] == 'ESPERADOS':
            f_esperados = line.rstrip('\n').split('=')[1]
            logger.info('Gracefully stopped reading configuration file ' +
                        CONFIG_FILE + ', ESPERADOS parameter found.')
            break
        else:
            logger.error('Invalid parameter found reading configuration.')
    if f_leia and f_consultas and f_esperados:
        logger.info('All set! Configuration successfully read!')
    else:
        logger.error('Error reading configuration files!')

    logger.info('Parsing .xml...')
    tree = ET.parse(f_leia)
    root = tree.getroot()
    if root:
        for QUERY in root.findall('QUERY'):
            query = queryRecords()
            query.Number = int(QUERY.find('QueryNumber').text)
            query.Text = ' '.join(re.sub('[^a-zA-Z]', ' ',
                                QUERY.find('QueryText').text).upper().split())
            query.Results = int(QUERY.find('Results').text)
            query.Records = {}
            for item in QUERY.iter('Item'):
                query.Records[item.text] = item.get('score')
            queries.append(query)
        logger.info('Recovered ' + str(len(queries)) + ' queries.')
    else:
        logger.error('Failed parsing .xml')

    logger.info('Exporting queries to .csv')
    f_out = open(f_consultas, 'w', encoding = 'utf-8')
    f_out.write('QueryNumber' + SEP + 'QueryText\n')
    count = 0
    for i in range(0, len(queries)):
        f_out.write(str(queries[i].Number) + SEP + queries[i].Text + '\n')
        count += 1
    f_out.close()
    logger.info('Exported ' + str(count) + ' records to ' + f_consultas)

    logger.info('Exporting document\'s votes to .csv')
    f_out = open(f_esperados, 'w', encoding = 'utf-8')
    f_out.write('QueryNumber' + SEP + 'DocNumber' + SEP + 'DocVotes\n')
    count = 0
    for i in range(0, len(queries)):
        for docs, votes in queries[i].Records.items():
            f_out.write(str(queries[i].Number) + SEP + docs + SEP +
                        computeVotes(votes) + '\n')
            count += 1
    f_out.close()
    logger.info('Exported ' + str(count) + ' records to ' + f_esperados)
    logger.info('Finished %s' % __file__.split('/')[-1])
else:
    logger.error(CONFIG_FILE + ' not found!')
    print(CONFIG_FILE + ' not found! Execution aborted.')
    logger.error('Execution aborted.')