#!/Library/Frameworks/Python.framework/Versions/3.6/bin/Python3.6
# -*- coding: utf-8 -*-
"""
Created: 2017-08-07
@title:  IMIR-VSM (In Memory Information Retrieval - Vector Space Model)
@module: inverted_ps.py
@author: Rafael Nunes - rnunes@cos.ufrj.br
"""
# Gerador de Lista Invertida - A função desse módulo é criar as listas
#                              invertidas simples.


import vsm_PorterStemmer as vsm_ps
import os
import re
import operator
import logging as log
import xml.etree.cElementTree as ET
from nltk.corpus import stopwords
if not stopwords: nltk.download('stopwords')
from nltk.stem.porter import *


# os.chdir('/Users/rafaenune/Documents/PESC-EDC/COS738 - Busca e Recuperação '
#          'da Informação/GitHub/')
# log.basicConfig(level=log.DEBUG,
#                 format='%(asctime)s|%(levelname)s|%(name)s|%(funcName)s'
#                        '|%(message)s',
#                 filename=__file__.split('.')[0]+'.log',
#                 filemode='w')


logger = log.getLogger(__file__.split('/')[-1])
CORPORA_FILE = './out/corpora_PorterStemmer.csv'
CONFIG_FILE = 'inverted_PorterStemmer.cfg'
SEP = ';'
MIN_WORD_LENGHT = 2
STOPWORDS = 0
# 1 - homemade stop_words list;
# 2 - nltk stop_words;
# any other value - no use of stop_words.


class paperRecords:
    """ Classe para armazenar os registros lidos do .xml """

    def __init__(self):
        self.PaperNum = ''
        self.Citations = []
        self.RecordNum = 0
        self.MedlineNum = 0
        self.Authors = []
        self.Title = ''
        self.Source = ''
        self.MajorSubJ_Topics = []
        self.MinorSubJ_Topics = []
        self.Abstract = ''
        self.References = []

    def __repr__(self):
        return '{}: {} {} {} {} {} {}' \
               '    {} {} {} {} {}'.format(self.__class__.__name__,
                                           self.PaperNum,
                                           self.Citations,
                                           self.RecordNum,
                                           self.MedlineNum,
                                           self.Authors,
                                           self.Title,
                                           self.Source,
                                           self.MajorSubJ_Topics,
                                           self.MinorSubJ_Topics,
                                           self.Abstract,
                                           self.References)


logger.info('Started %s' % __file__.split('/')[-1])
files = []
papers = []
count = 0
if os.path.isfile(CONFIG_FILE):
    logger.info('Reading configuration from ' + CONFIG_FILE + '...')
    for line in open(CONFIG_FILE, 'r'):
        if line.rstrip('\n').split('=')[0] == 'LEIA':
            files.append(line.rstrip('\n').split('=')[1])
            count += 1
        elif line.rstrip('\n').split('=')[0] == 'ESCREVA':
            file_out = line.rstrip('\n').split('=')[1]
            logger.info('Gracefully stopped reading configuration file ' +
                        CONFIG_FILE + ', ESCREVA parameter found.')
            break
        else:
            logger.error('Invalid parameter found reading configuration. ')
    if count > 0: logger.info('Found %d .xml files to parse' % count)

    if files and file_out:
        logger.info('All set! Configuration successfully read!')
    else:
        logger.error('Error reading configuration files!')

    logger.info('Parsing .xmls...')
    total_count = 0
    total_fails = 0
    for file in files:
        count = 0
        fails = 0
        logger.info('Parsing file %s' % file)
        tree = ET.parse(file)
        root = tree.getroot()
        if root:
            for RECORD in root.findall('RECORD'):
                paper = paperRecords()
                count += 1
                try:
                    paper.PaperNum = RECORD.find('PAPERNUM').text
                except TypeError:
                    logger.warning('Missing PAPERNUM attribute')
                    pass
                try:
                    for cite in RECORD.find('CITATIONS'):
                        paper.Citations.append(cite.attrib)
                except TypeError:
                    # logger.warning('Record: ' + paper.PaperNum +
                    #                ' at file: ' + file + ' has none'
                    #                ' citations.')
                    pass
                try:
                    paper.RecordNum = int(RECORD.find('RECORDNUM').text)
                except TypeError:
                    logger.error('Missing RECORDNUM attribute')
                    fails += 1
                    continue
                try:
                    paper.MedlineNum = int(RECORD.find('MEDLINENUM').text)
                except TypeError:
                    # logger.warning('Missing MEDLINENUM attribute')
                    pass
                try:
                    for author in RECORD.find('AUTHORS'):
                        paper.Authors.append(author.text)
                except TypeError:
                    # logger.warning('No authors found')
                    pass
                try:
                    paper.Title = ' '.join(re.sub('[^a-zA-Z]', ' ',
                                           RECORD.find('TITLE').text)
                                           .upper().split())
                except TypeError:
                    # logger.warning('Record has no title')
                    pass
                try:
                    paper.Source = RECORD.find('SOURCE').text
                except TypeError:
                    # logger.warning('Missing SOURCE attribute')
                    pass
                try:
                    for topic in RECORD.find('MAJORSUBJ'):
                        paper.MajorSubJ_Topics.append(topic.text)
                except TypeError:
                    # logger.warning('MIssing MAJORSUBJ attribute')
                    pass
                try:
                    for topic in RECORD.find('MINORSUBJ'):
                        paper.MinorSubJ_Topics.append((topic.text))
                except TypeError:
                    # logger.warning('Missing MINORSUBJ attribute')
                    pass
                try:
                    paper.Abstract = RECORD.find('ABSTRACT').text
                except AttributeError:
                    # logger.warning('Record: ' + paper.PaperNum +
                    #                ' at file: ' + file + ' has no'
                    #                ' ABSTRACT. Searching for EXTRACT...')
                    try:
                        paper.Abstract = RECORD.find('EXTRACT').text
                    except:
                        logger.error('There is no ABSTRACT nor EXTRACT at '
                                     'record %s of file %s, ignoring record'
                                     % (paper.PaperNum, file))
                        fails += 1
                        continue
                finally:
                    paper.Abstract = ' '.join(re.sub('[^a-zA-Z]', ' ',
                                            paper.Abstract).upper().split())
                try:
                    for cite in RECORD.find('REFERENCES'):
                        paper.References.append(cite.attrib)
                except:
                    # logger.warning('Record: ' + paper.PaperNum +
                    #                ' at file: ' + file + ' has none'
                    #                ' references.')
                    pass
                papers.append(paper)
            logger.info('%s - %d records successfully imported, '
                        '%d records ignored => total parsed: %d'
                        % (file, count-fails, fails, count))
        else:
            logger.error('Failed parsing file ' + file)
        total_count += count
        total_fails += fails
    logger.info('Parsed all .xmls - Successfully imported %d records out of'
                ' %d parsed.' % (len(papers), total_count))
    logger.info('Sorting papers array...')
    papers = sorted(papers, key = operator.attrgetter('RecordNum'))
    logger.info('Papers array sorted by RecordNum.')

    logger.info('Generating inverted index and saving to %s...' % file_out)
    index = dict()
    stemmer = PorterStemmer()
    if STOPWORDS == 1:
        stop_words = ['this','not','from','how','what','why','when','where',
                      'which', 'who', 'with']
        logger.info('Using homemade stop_words list.')
    elif STOPWORDS == 2:
        stop_words = set(stopwords.words('english'))
        logger.info('Using nltk standard stop_words.')
    else:
        stop_words = []
        logger.info('Not using stop_words.')
    for i in range(0, len(papers)):
        words = papers[i].Abstract.split()
        words = [stemmer.stem(word) for word in words
                 if not word in stop_words
                 and len(word) >= MIN_WORD_LENGHT]
        for word in words:
            if word in index:
                index[word].append(papers[i].RecordNum)
            else:
                index[word] = [papers[i].RecordNum]
    logger.info('Inverted index generated in memory'
                ' with %d words.' % len(index))

    f_out = open(file_out, 'w', encoding = 'utf-8')
    f_out.write('Word' + SEP + 'Documents\n')
    for word, docs in sorted(index.items()):
        f_out.write(str(word) + SEP + str(docs) + '\n')
    f_out.close()
    logger.info('Inverted index saved as %s' %file_out)

    logger.info('Exporting corpora {\'RecordNum\' : \'Abstract\'} to %s'
                % CORPORA_FILE)
    f_out = open(CORPORA_FILE, 'w', encoding = 'utf-8')
    f_out.write('corpus' + SEP + 'text\n')
    for i in range(0, len(papers)):
        f_out.write(str(papers[i].RecordNum) + SEP + papers[i].Abstract + '\n')
    f_out.close()
    logger.info('%s created with %d corpus.' % (CORPORA_FILE,
                                                len(papers)))
    logger.info('Finished %s' % __file__.split('/')[-1])

else:
    logger.error(CONFIG_FILE + ' not found!')
    print(CONFIG_FILE + ' not found! Execution aborted.')
    logger.error('Execution aborted.')
