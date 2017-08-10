#!/Library/Frameworks/Python.framework/Versions/3.6/bin/Python3.6
# -*- coding: utf-8 -*-
"""
Created: 2017-07-05
@title:  IMIR-VSM (In Memory Information Retrieval - Vector Space Model)
@module: query.py
@author: Rafael Nunes - rnunes@cos.ufrj.br
"""
# Processador de Consultas - O objetivo desse módulo é transformar o
#       arquivo de consultas fornecido no padrão de palavras que estamos
#       utilizando.
# DONE: 1) O Processador de Consultas deverá ler um arquivo de configuração
# a. O arquivo é criado por vocês
# b. O nome do arquivo é PC.CFG
# c. Ele contém dois tipos de instruções:
#   i. LEIA=<nome de arquivo>
#  ii. CONSULTAS=<nome de arquivo>
# iii. ESPERADOS=<nome de arquivo>
#  iv. As instruções são obrigatórias, aparecem uma única vez e nessa ordem.

# DONE: 2) O Processador de Consultas deverá ler um arquivo em formato XML
# a. O arquivo a ser lido será indicado pela instrução LEIA no arquivo de
#    configuração
#   i. O formato é descrito pelo arquivo “cfcquery-2.dtd”.
#  ii. O arquivo a ser lido é “cfquery.xml”.

# DONE: 3) O Processador de Consultas deverá gerar dois arquivos
# a. Os arquivos deverão ser no formato cvs
#   i. O caractere de separação será o “;”, ponto e vírgula
#       1. Todos os caracteres “;” que aparecerem no arquivo original
#          devem ser eliminados
#  ii. A primeira linha do arquivo cvs deve ser o cabeçalho com o nome dos
#      campos
#
# b. O primeiro arquivo a ser gerado será indicado na instrução CONSULTAS do
#    arquivo de configuração
#   i. Cada linha representará uma consulta
#       1. O primeiro campo de cada linha conterá o número da consulta
#           a. Campo QueryNumber
#       2. O segundo campo de cada linha conterá uma consulta processada em
#          letras maiúsculas, sem acento
#           a. A partir do campo QueryText
#       3. Cada aluno poderá escolher como criar sua consulta
#
# c. O segundo arquivo a ser gerado será indicado na instrução ESPERADOS
#   i. Cada linha representará uma consulta
#       1. O primeiro campo de cada linha conterá o número da consulta
#           a. Campo QueryNumber
#       2. O segundo campo conterá um documento
#           a. Campo DocNumber
#       3. O terceiro campo conterá o número de votos do documento
#           a. Campo DocVotes
#       4. Uma consulta poderá aparecer em várias linhas, pois podem possuir
#          vários documentos como resposta
#       5. As linhas de uma consulta devem ser consecutivas no arquivo
#       6. Essas contas devem ser feitas a partir dos campos Records, Item e
#          do atributo Score de Item
#           a. Considerar qualquer coisa diferente de zero como um voto

# DONE: 4) Todos os módulos devem possuir um LOG que permita pelo menos a um
#          programador posterior, usando o módulo logging de Python:
# 1. Identificar quando iniciaram suas operações
# 2. Identificar quando iniciam cada parte de seu processamento
# a. Ler arquivo de configuração
# b. Ler arquivo de dados
# 3. Identificar quantos dados foram lidos
# 4. Identificar quando terminaram os processamentos
# 5. Calcular os tempos médios de processamento de consultas, documento e
#    palavras, de acordocom o programa sendo usado
# 6. Identificar erros no processamento, caso aconteçam.

import os
import logging as log
import xml.etree.cElementTree as ET
import vsm

# os.chdir('/Users/rafaenune/Documents/PESC-EDC/COS738 - Busca e Recuperação '
#          'da Informação/GitHub/')
# log.basicConfig(level=log.DEBUG,
#                 format='%(asctime)s|%(levelname)s|%(name)s|%(funcName)s'
#                        '|%(message)s',
#                 filename=__file__.split('.')[0]+'.log',
#                 filemode='w')


logger = log.getLogger(__file__.split('/')[-1])
CONFIG_FILE = 'PC.CFG'
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
    evaluation = 0
    for i in range(0, len(votes)):
        evaluation = evaluation + int(votes[i])
    return(str(evaluation))


logger.info('Started %s' % __file__)
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
            query.Text = QUERY.find('QueryText').text
            words = query.Text.split()
            query.Text = ' '.join(words).upper()
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
    logger.info('Finished %s' % __file__)
else:
    logger.error(CONFIG_FILE + ' not found!')
    print(CONFIG_FILE + ' not found! Execution aborted.')
    logger.error('Execution aborted.')