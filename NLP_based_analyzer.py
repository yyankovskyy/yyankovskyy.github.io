#!/usr/bin/env python

# coding: utf-8

 

# # AI Silver Badge Final Project - COVID 19 Insights

# by Eugene Yankovsky 

 

# # User's instructions

# The user is expected to provide

# a) the search tasks;

# b) parameters of the search functions and verbosity of the output.

#

# # Required input

# 1. A file with metadata on papers to apply the search on that must include inique id  and path to the folder containing a file with the paper abatrsct

# 2. Text files including comprehensive information on the papers: its title, abstract, full text (if available), referenced papers

#

# # Search procedure outline

# The NLP procedcedure a) applies tokenization of the search task and papers' including their title, abstract, full text and referenced papers if they all are available, b) matches tasks' 2-token combinations (bigrams) with paper's corpus.

#

# # Reporting results

# Search results by paper

# The search results on each paper are aggregated in the text files containing the paper's cord_uid in CORD-19 archive (e.g., nlp_report_on_paper_ejv2xln0 for paper with cord_uid ejv2xln0) in the folder specified by user in path_to_output_data parameter.

# The report contains the unique cord_uid in CORD-19 archive, search tasks' full text and their bigrams, all matches of the task bigrams with the paper's corpus bigrams with no_context_words on the left and right of the matched bigrams in the paper's corpus.

#

# Search results by task

# The search results by task are aggregated in 2 files per task:

#

# a) the text report

# containing the task number, unique cord_uid in CORD-19 archive, all matches of the task bigrams with the paper's corpus bigrams with no_context_words on the left and right of the matched bigrams in the paper's corpus.

# This report is saved in the text files with the naming convention

# task_number_bigrams_report (e.g., task_1_bigrams_report for task 1)

# in the folder specified by user in path_to_output_data parameter.

#

# b) the word cloud exploratory analysis

# provided in the jupyter notebook below and saved in the files with the naming convention  task_number_bigram_wordcount.png (e.g,task_1_bigram_wordcount.png  for task 1)

# in the folder specified by user in path_to_output_data parameter.

#

# c) SVD transformation to generate a number of the tokenized themes (no_themes parameter) and report their eigenvalues in descending order to show what themes have been prevailing per each task 

#

# # Search procedure assumptions

# Research on the big sample of the papers resulted in the following conclusions:

# 1. TF_IDF normalization is preferred to Count vectorizer because of a) capability of text normalization scattered in multiple documents, b) seemingly better resutls

#

# 2. Porter stemming has been chosen over Lemmatizing since

# Porter stemming produces 1/7 less of tokens in a sample of 10 documents and consume less economical in processing

#

# 3. Collocations using PMI (Pointwise_mutual_information) produce verifiable combinations of 2-gram collocations (e.g., ('zip', 'code'), ('chief','complaint'), ('clinic', 'trial'). Since corpus' collocations resulted in too few matches with the tasks, the final solution used matches with the corpus simple bigrams instead of the collocations.

 

# In[1]:

 

 

# basic libraries for analysis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import fastai

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn import decomposition

import os

from os import path # for file exists check

import json

import re

from scipy import linalg

import fbpca

from collections import Counter

import time

 

# nltk suite

import nltk

from nltk import word_tokenize

from nltk import stem

from nltk.stem import PorterStemmer

from nltk.util import tokenwrap

from nltk.text import TokenSearcher

from nltk.collocations import *

import nltk.corpus  # for concordance

from nltk.text import Text  # for concordance

nltk.download('stopwords')

nltk.Text.common_contexts

 

# Stop words definition

from spacy.lang.en import English

# Load English tokenizer, tagger, parser, NER and word vectors

from spacy.lang.en.stop_words import STOP_WORDS

spacy_stopwords = set(STOP_WORDS)

# QA

# print(sorted(list(spacy_stopwords))[:20])

 

# Display options

pd.options.display.max_rows = 100

pd.options.display.float_format = '{:.3f}'.format

pd.options.display.max_colwidth = 100

 

# needed for wordlcoud analysis

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

 

 

# In[2]:

 

 

# initial stemmer

porter_stemmer=PorterStemmer()

 

 

# In[3]:

 

 

# Finalizing stop word list based on the test runs

stop_words_final = spacy_stopwords.update('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20',

    'aa', 'aaai','abov', 'afterward', 'alon', 'alreadi', 'alway', 'ani', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becaus', 'becom', 'befor', 'besid', 'doe', 'dure', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti', 'formerli', 'forti', 'ha', 'henc', 'hereaft', 'herebi', 'hi', 'howev', 'hundr', 'inde', 'latterli', 'll', 'mani', 'meanwhil', 'moreov', 'mostli', 'nobodi', 'noon', 'noth', 'nowher', 'onc', 'onli', 'otherwis', 'ourselv', 'perhap', 'pleas', 'quit', 'realli', 'regard', 'seriou', 'sever', 'sinc', 'sixti', 'someon', 'someth', 'sometim', 'somewher', 'themselv', 'thenc', 'thereaft', 'therebi', 'therefor', 'thi', 'thu', 'togeth', 'twelv', 'twenti', 'use', 'variou', 've', 'veri', 'wa', 'whatev', 'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev', 'whi', 'yourselv') 

 

 

# Assign the project hypeparameters:

 

# In[4]:

 

 

# Needs to be specific to each place where it's run

path_to_input_data = '/home/jupyter/project'

path_to_output_data = '/home/jupyter/project/NLPOutputData'

 

 

no_repeat_words_for_bigrams_in_corpus = 10

# only bigrams that appear repeats + times

 

no_context_words = 20

# number of context words to report around the 2-word collocation extracted from the search task

 

 

# Wordcloud exploration parameters:

user_max_font_size=50

user_max_words = 100

user_background_color="white"

 

# SVD parameters to extract a number of dominant topics

no_themes = 10

 

 

# In[5]:

 

 

# Task load

# Text of the task_general is taken straight from the task statement

task_0 = 'What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)?'

 

 

# In[6]:

 

 

task_1 = '1.How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).'

 

task_2 ='2.Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.'

 

task_3 = '3.Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.'

 

task_4 = '4.National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).'

 

task_5 = '5.Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.'

 

task_6 ='6.Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).'

 

task_7 ='7.Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.'

 

task_8 = '8.Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.'

 

 

# In[7]:

 

 

task_9 ='9.Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.'

 

task_10 ='10.Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.'

 

 

task_11 ='11.Policies and protocols for screening and testing.'

 

 

task_12 ='12.Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.'

 

task_13 ='13.Technology roadmap for diagnostics.'

 

task_14 ='14.Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.'

 

task_15 = '15.New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.'

 

task_16 = '16.Coupling genomics and diagnostic testing on a large scale.'

 

task_17 ='17.Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.'

 

task_18 ='18.Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.'

 

 

# In[8]:

 

 

task = list([task_0,task_1,task_2, task_3,task_4,task_5,task_6,task_7,task_8,task_9,task_10,

             task_11,task_12,task_13,task_14,task_15,task_16,task_17,task_18])

# QA

print(task)

 

 

# In[9]:

 

 

os.chdir(path_to_input_data)

 

 

# In[10]:

 

 

# metadata_df = pd.read_csv(f'metadata.csv')

 

full_metadata_df = pd.read_csv(f'metadata.csv',

                usecols=["cord_uid","source_x","title","abstract", "authors","journal","pdf_json_files"],

                dtype= {"cord_uid":np.str,"source_x":np.str,"title":np.str,"abstract":np.str, "authors":np.str,"journal":np.str,

                        "pdf_json_files":np.str}

                 )

 

 

# In[11]:

 

 

def porter_stemming(text):

    #Porter stem words

    words=re.split("\\s+",text)

    stemmed_words=[porter_stemmer.stem(word=word) for word in words]

    return ' '.join(stemmed_words)

 

 

# In[12]:

 

 

# Tokenize task details and splitting them into bi-grams

def task_preprocessor(text):

    text = text.lower()

    text= re.sub('[^\s\tsa-zA-Z ]',' ', text.lower())

    text = re.sub("\\s+(in|the|all|for|and|on|with|non|none)\\s+"," ",text)

    text = " ".join(text.split())

    pattern = re.compile(r'\b(' + r'|'.join(spacy_stopwords) + r')\b\s*')

    text = pattern.sub('', text)

    text = text.strip()

    text = porter_stemming(text).split()

    text_str= ' '.join(map(str,text)).lower()

    bigram_vectorizer = CountVectorizer(analyzer='word',ngram_range=(2,2), min_df=1, stop_words=spacy_stopwords)

    analyze = bigram_vectorizer.build_analyzer()

    return analyze(text_str)

 

 

# In[13]:

 

 

task_0_bigram = task_preprocessor(task_0)

task_1_bigram = task_preprocessor(task_1)

task_2_bigram = task_preprocessor(task_2)

task_3_bigram = task_preprocessor(task_3)

task_4_bigram = task_preprocessor(task_4)

task_5_bigram = task_preprocessor(task_5)

task_6_bigram = task_preprocessor(task_6)

task_7_bigram = task_preprocessor(task_7)

task_8_bigram = task_preprocessor(task_8)

task_9_bigram = task_preprocessor(task_9)

task_10_bigram = task_preprocessor(task_10)

task_11_bigram = task_preprocessor(task_11)

task_12_bigram = task_preprocessor(task_12)

task_13_bigram = task_preprocessor(task_13)

task_14_bigram = task_preprocessor(task_14)

task_15_bigram = task_preprocessor(task_15)

task_16_bigram = task_preprocessor(task_16)

task_17_bigram = task_preprocessor(task_17)

task_18_bigram = task_preprocessor(task_18)

 

#QA

print(task_0_bigram)

# print(task_18_bigram)

 

 

#

#               

 

# In[14]:

 

 

# REGEX transformations to remove original markers for the content

def input_preprocessor(text):

    text=text.lower()

    text = re.sub("'paper_id':", ' ', text)

    text = re.sub("'metadata':", ' ', text)

    text = re.sub("'authors':", ' ', text)

    text = re.sub("'title':", ' ', text)

    text = re.sub("'first':", ' ', text)

    text = re.sub("'middle':", ' ', text)

    text = re.sub("last':", ' ', text)

    text = re.sub("'cite_spans':", ' ', text)

    text = re.sub("'ref_spans':", ' ', text)

    text = re.sub("'cite_spans':", ' ', text)

    text = re.sub("'latex':", ' ', text)

    text = re.sub("'text:'", ' ', text)

    text = re.sub("'section':", ' ', text)

    text = re.sub("'suffix':", ' ', text)

    text = re.sub("'suffix':", ' ', text)

    text = re.sub("'affiliation'", ' ', text)

    text = re.sub("'abstract':", ' ', text)

    text = re.sub("'background:", ' ', text)

    text = re.sub("'objective:", ' ', text)

    text = re.sub("'ref_id':", ' ', text)

    text = re.sub("'start':", ' ', text)

    text = re.sub("'end':", ' ', text)

    text = re.sub("'venue':", ' ', text)

    text = re.sub("'volume':", ' ', text)

    text = re.sub("'issn':", ' ', text)

    text = re.sub("'pages':", ' ', text)

    text = re.sub("'other_ids':", ' ', text)

    text = re.sub("'ref_id':", ' ', text)

    # Some stop-words slipped through stop-word remover

    text = re.sub("bibref", ' ', text)

    text = re.sub("text", ' ', text)

    text = re.sub('ing', ' ', text)

    text = re.sub("\\s+(in|the|all|for|and|on|with|non|none)\\s+"," ",text)

    text = re.sub('[^\s\tsa-zA-Z ]',' ', text)

    text = " ".join(text.split())

    pattern = re.compile(r'\b(' + r'|'.join(spacy_stopwords) + r')\b\s*') # stop_words_final

    text = pattern.sub('', text)

    return text.strip()

 

 

# In[15]:

 

 

class LemmaTokenizer(object):

     def __init__(self):

         self.wnl = stem.WordNetLemmatizer()

     def __call__(self, doc):

         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

 

 

# In[16]:

 

 

def porter_stemming(text):

    #Porter stemming words

    words=re.split("\\s+",text)

    stemmed_words=[porter_stemmer.stem(word=word) for word in words]

    return ' '.join(stemmed_words)

 

 

# In[17]:

 

 

# Finding concordance with n_context_words words in a searh before and after bigrams

def find_bigrams_in_corpus(search_text, bigram, n_context_words):

    bigram_str= ' '.join(map(str,bigram)).lower()

    twotokens = list(bigram_str.split(' '))

    list_start = ['<']

    list_end = ['>']

    duplicate_list = ['<.*>'] * n_context_words

    list_between = ['> <']

    out_string = ' '.join(map(str,(duplicate_list + list_start + twotokens[0].split() + list_between + twotokens[1].split() + list_end  + duplicate_list))).lower()

    hits = TokenSearcher(search_text).findall(out_string)

    #hits = [' '.join(h) for h in hits]

    #tokenwrap(hits, "; ")

    return hits

   

 

 

# In[18]:

 

 

# Save search results into the ouput file containing the paper_id in the file name

def paper_reviewer(path_to_input_data, path_to_output_data, no_repeat_words_for_bigrams_in_corpus, no_context_words, cord_uid, task):

   

    # Reading relevant paper

    #metadata_df = input_df[input_df['cord_uid']==cord_uid]

       

    os.chdir(path_to_input_data)  

    metadata_df = full_metadata_df[full_metadata_df['cord_uid']==cord_uid]

    papers_dict = []

    for index, row in metadata_df.iterrows():

        json_file_name = row['pdf_json_files']

        try:

            if (path.exists(json_file_name)): # Sanity check to see

                with open(json_file_name) as f:

                    papers_dict.append(json.load(f))

        except:

            return

   

    # Forming corpus and REGEX cleaning

    corpus= ' '.join(map(str,papers_dict)).lower()

    # QA

    # print(corpus)          

    

    corpus_apld = input_preprocessor(corpus)

    # QA

    #print(corpus_apld)

   

    #Remove one- and two-letter words from corpus that are mostly lefovers of the paper id's

    corpus_apld_list = [i for i in corpus_apld.split() if len(i) > 2]

    # QA

    #print(corpus_apld_list)

   

    # Tokenize the CORD_19 text: a) Remove maximum of stopwords, b) stemming or lemming

  

    # Remove 1- and 2-letter words that are mostly left from the paper_id celaning

    x = porter_stemming(corpus_apld).split()

    corpus_apld_stemmed_list = [i for i in x if len(i) > 2]

    # QA

    #print(corpus_apld_stemmed_list)

   

 

          

    # Find concordance of all bigram combinations

    corpus_apld_stem_str= ' '.join(map(str,corpus_apld_stemmed_list)).lower()

    textlist = Text(corpus_apld_stemmed_list)

 

    # QA

    #print(corpus_apld_stem_str)

    #print(textlist)

   

    # Save search results into the ouput file containing the paper_id in the file name

    file_name = "nlp_report_on_paper_%s" % cord_uid

 

    os.chdir(path_to_output_data)

    f = open(file_name, "w+")

    L = ['This is report on search task in paper with cord_uid %s: \n' % cord_uid ]

    f.writelines(L)

    f.close()

 

    with open(file_name, "a") as f:

        for i in range(0,len(task)):

            name = "task_%d_bigram" % i

            f.write(task[i])

            f.writelines("\n")

            for colloc in globals()[name]:

                collocs = colloc.split(" ")

                f.writelines(colloc)

                f.writelines("\n")

                find_bigrams_in_corpus(textlist, colloc.split(), n_context_words = no_context_words)

                return_data = find_bigrams_in_corpus(textlist, colloc.split(), n_context_words = no_context_words)

                if len(return_data) == 0:

                    f.writelines("None")

                    f.writelines("\n")

                else:

                    tf[i].writelines("cord_uid: " + cord_uid + ", title: " + metadata_df['title'])

                    tf[i].writelines("\n")

                    tf[i].writelines("bigram: " + colloc)

                    tf[i].writelines("\n")

                    for k in range(0, len(return_data)):

                        sentence = return_data[k]

                        sentence_str = ' '.join(map(str,sentence)).lower()

                        f.writelines(sentence_str)

                        f.writelines("\n")

                        tf[i].writelines(sentence_str)

                        tf[i].writelines("\n")

                    tf[i].writelines("\n")

            f.writelines("\n")

            f.writelines("\n")

    f.close()

 

 

# In[ ]:

 

 

tf = []

for i in range(0, len(task)):

    name = "task_%d_bigram" % i

    task_filename = name + "s_report"

    os.chdir(path_to_output_data)

    tfi = open(task_filename, "w+")

    tf.append(tfi)

    task_title = ['This is the report on %s \n\n' % name ]

    tf[i].writelines(task_title)

   

metadata_df = full_metadata_df[:]

for index, row in metadata_df.iterrows():

    cord_uid = row['cord_uid']   

    #print(cord_uid)

    paper_reviewer(path_to_input_data ,

    path_to_output_data,

    no_repeat_words_for_bigrams_in_corpus,

    no_context_words,

    cord_uid = cord_uid, task = task)

 

for i in range(0, len(task)):

    tf[i].close()

 

 

# In[19]:

 

 

# Read data from the analysis output by task to summarize results in the wordcloud analysis

os.chdir(path_to_output_data)

start = time.time()

for i in range(0, len(task)):

    name = "task_%d_bigram" % i

    task_filename = name + "s_report"

    wordcloud_file = name + "_wordcloud.png"

    os.chdir(path_to_output_data)

    tfi = open(task_filename, "r")

    olines = tfi.readlines()

    line1 = tfi.readline().strip()

    lines = []

    for line in olines:

        line = line.replace('\n', '')

        line = re.sub(r'cord_uid.*,','',line)

        line = line.replace('bigram', '')

        line = line.replace('title', '')

        line = line.replace('task_', '')

        line = line.replace('report', '')

        line = line.replace('_', '')

        line = line.replace('covid-19', '')

        line = line.replace('covid-19', '')

        line = line.replace('Covid-19', '')

        line = line.replace('COVID-19', '')

        line = line.replace('covid 19', '')

        line = line.replace('Covid 19', '')

        line = line.replace('COVID 19', '')

        line = line.replace('covid', '')

        line = line.replace('Covid', '')

        line = line.replace('COVID', '')

        line = line.replace('health', '')

        line = line.replace('patient', '')

        line = line.replace('year', '')

        line = line.replace('diseas', '')

        line = line.replace('viru', '')

        for twogram in globals()[name]:

            line = line.replace(twogram, '')

        lines.append(line)

          

    if len(lines)==0:

        text = 'No words'

        print('%s word cloud chart is not feasible. It requires at least 1 word to plot a word cloud, got 0 for \n' % name) 

    else:

        # Create and generate a word cloud image:

        print("%s word cloud chart" % name)

        wordcloud = WordCloud(stopwords=stop_words_final,max_font_size= user_max_font_size, max_words= user_max_words, background_color=user_background_color)

 

        counts_all = Counter()

 

        for line in lines:  # Here you can also use the Cursor

            counts_line = wordcloud.process_text(line)

            counts_all.update(counts_line)

           

        wordcloud.generate_from_frequencies(counts_all)

 

        # Display the generated image:

        plt.figure()

        plt.imshow(wordcloud, interpolation="bilinear")

        plt.axis("off")

        plt.show()

   

    # Save the image in the path_to_output_data folder:

        wordcloud.to_file(wordcloud_file)

 

    tfi.close()

end = time.time()

print ("TIME TO RUN:")

print(end-start)

 

 

# In[19]:

 

 

def show_topics(a, num_top_words):

    top_words = lambda t: [vocab[i] for i in np.argsort(t)[:-num_top_words-1:-1]]

    topic_words = ([top_words(t) for t in a])

    return [' '.join(t) for t in topic_words]

 

 

# In[20]:

 

 

vectorizer = CountVectorizer(stop_words=spacy_stopwords)

vectorizer_tfidf = TfidfVectorizer(stop_words=spacy_stopwords)

 

 

# In[22]:

 

 

os.chdir(path_to_output_data)

start = time.time()

for i in range(0, len(task)):

    tname = "task_%d" % i

    name = tname + "_bigram"

    task_filename = name + "s_report"

    olines = []

    lines = []

    with open(task_filename, "r") as f:

        olines = f.readlines()

    for line in olines:

        line = line.replace('\n', '')

        line = re.sub(r'cord_uid.*,','',line)

        line = line.replace('bigram', '')

        line = line.replace('title', '')

        line = line.replace('task_', '')

        line = line.replace('report', '')

        line = line.replace('_', '')

        line = line.replace('covid-19', '')

        line = line.replace('covid-19', '')

        line = line.replace('Covid-19', '')

        line = line.replace('COVID-19', '')

        line = line.replace('covid 19', '')

        line = line.replace('Covid 19', '')

        line = line.replace('COVID 19', '')

        line = line.replace('covid', '')

        line = line.replace('Covid', '')

        line = line.replace('COVID', '')

        line = line.replace('health', '')

        line = line.replace('patient', '')

        line = line.replace('year', '')

        line = line.replace('diseas', '')

        line = line.replace('viru', '')

        for twogram in globals()[name]:

            line = line.replace(twogram, '')

        lines.append(line)

 

    vectors = vectorizer.fit_transform(lines).todense()

    vocab = np.array(vectorizer.get_feature_names())

    print(globals()[tname])

    get_ipython().run_line_magic('time', 'U, s, Vh = fbpca.pca(vectors, no_themes)')

    plt.plot(s)

    plt.show()

    svd_list = show_topics(Vh[:no_themes], 20)

    for svd in svd_list:

        print("[ %s ]" % svd)

    f.close()

end = time.time()

print ("TIME TO RUN:")

print(end-start)

 

 

# In[ ]:

 

 

 

 

 

Thank you,

 

Eugene Yankovsky | Data Scientist Assistant Director | EY Technology

Ernst & Young LLP

500 High St, Palo Alto, CA 94303, USA

Mobile: +1 408 649 0284, Fax: +1 844 663 7933 | Eugene.Yankovsky@ey.com

 

❤️🇺🇦🙏​

OPEN YOUR HEART, SUPPORT UKRAINE WITH DONATIONS

Ukraine: Save lives in Ukraine | Come Back Alive

Poland: #EYGDSPoland4Ukraine | zrzutka.pl

USA-EY Employee: Ernst & Young (cybergrants.com)

USA-general: Home - Razom (razomforukraine.org)

 



Any tax advice in this e-mail should be considered in the context of the tax services we are providing to you. Preliminary tax advice should not be relied upon and may be insufficient for penalty protection.
________________________________________________________________________
The information contained in this message may be privileged and confidential and protected from disclosure. If the reader of this message is not the intended recipient, or an employee or agent responsible for delivering this message to the intended recipient, you are hereby notified that any dissemination, distribution or copying of this communication is strictly prohibited. If you have received this communication in error, please notify us immediately by replying to the message and deleting it from your computer.

Notice required by law: This e-mail may constitute an advertisement or solicitation under U.S. law, if its primary purpose is to advertise or promote a commercial product or service. You may choose not to receive advertising and promotional messages from Ernst & Young LLP (except for My EY, which tracks e-mail preferences through a separate process) at this e-mail address by opting out of emails through EY’s Email Preference Center. Our principal postal address is One Manhattan West, New York, NY 10001. Thank you. Ernst & Young LLP
...

[Message clipped]  View entire message

Eugene Yankovsky <Eugene.Yankovsky@ey.com>
Tue, Feb 14, 2023, 10:58 AM
to me

Tasks/Datasets

Please submit a contribution that addresses one or more of the tasks below.Datasets are available on Kaggle.

What do we know about non-pharmaceutical interventions?

https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=587

 

What do we know about diagnostics and surveillance?

https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=570

 

What has been published about information sharing and inter-sectoral collaboration?

 

https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/tasks?taskId=583



Eugene Yankovsky <Eugene.Yankovsky@ey.com>
Tue, Feb 14, 2023, 11:29 AM
to me, Eugene

# Sentiment Classification-Naive Bayes and Logistic Regression.py

 

#!/usr/bin/env python

# coding: utf-8

 

# In[5]:

 

 

get_ipython().run_line_magic('reload_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')

 

 

# In[6]:

 

 

from fastai import *

from fastai.text import *

from fastai.utils.mem import GPUMemTrace #call with mtrace

import sklearn.feature_extraction.text as sklearn_text

import pickle

#?? URLs

 

<!-- # If You face any issues with installing torch then follow the below steps

1. Go to PyTorch website

2. Get Started -> Start locally

3. select the PyTorch version you want (e.g. Stable 1.5), select your OS, Package manager (e.g. conda, pip), select language (e.g. Python, Java), and your CUDA version if any

then the site will generate a command for you to run "Run this command", e.g.

pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html -->

# In[7]:

 

 

path = untar_data(URLs.IMDB_SAMPLE)

path

df = pd.read_csv(path/'texts.csv')

df.head()

 

 

# In[8]:

 

 

get_ipython().run_cell_magic('time', '', "# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!\n\ncount = 0\nerror = True\nwhile error:\n    try: \n        # Preprocessing steps\n        movie_reviews = (TextList.from_csv(path, 'texts.csv', cols='text')\n                         .split_from_df(col=2)\n                         .label_from_df(cols=0))\n        error = False\n        print(f'failure count is {count}\\n')    \n    except: # catch *all* exceptions\n        # accumulate failure count\n        count = count + 1\n        print(f'failure count is {count}')\n        \nmovie_reviews\n#dir(movie_reviews)")

 

 

# In[9]:

 

 

print(f'There are {len(movie_reviews.train.x)} and {len(movie_reviews.valid.x)} reviews in the training and validations sets, respectively.')

 

 

# In[10]:

 

 

print(f'\fThere are {len(movie_reviews.train.x)} movie reviews in the training set\n')

print(movie_reviews.train)

 

 

# In[11]:

 

 

print(movie_reviews.train.x[0].text)

print(f'\nThere are {len(movie_reviews.train.x[0].text)} characters in the review')

 

 

# In[12]:

 

 

print(movie_reviews.train.x[0].text.split())

print(f'\nThe review has {len(movie_reviews.train.x[0].text.split())} tokens')

 

 

# In[13]:

 

 

print(movie_reviews.train.x[0].data)

print(f'\nThe array contains {len(movie_reviews.train.x[0].data)} numericalized tokens')

 

 

# In[14]:

 

 

movie_reviews.vocab

 

 

# In[15]:

 

 

movie_reviews.vocab.stoi

 

 

# In[16]:

 

 

movie_reviews.vocab.itos

 

 

# In[17]:

 

 

print('itos ', 'length ',len(movie_reviews.vocab.itos),type(movie_reviews.vocab.itos) )

print('stoi ', 'length ',len(movie_reviews.vocab.stoi),type(movie_reviews.vocab.stoi) )

 

 

# In[18]:

 

 

rare_words = ['acrid','a_random_made_up_nonexistant_word','acrimonious','allosteric','anodyne','antikythera']

for word in rare_words:

    print(movie_reviews.vocab.stoi[word])

 

 

# In[19]:

 

 

print(movie_reviews.vocab.itos[0])

 

 

# In[20]:

 

 

print(f'len(stoi) = {len(movie_reviews.vocab.stoi)}')

print(f'len(itos) = {len(movie_reviews.vocab.itos)}')

print(f'len(stoi) - len(itos) = {len(movie_reviews.vocab.stoi) - len(movie_reviews.vocab.itos)}')

 

 

# In[21]:

 

 

unk = []

for word, num in movie_reviews.vocab.stoi.items():

    if num==0:

        unk.append(word)

 

 

# In[22]:

 

 

len(unk)

 

 

# In[23]:

 

 

unk[:25]

 

 

# In[24]:

 

 

print(f'There are {len(movie_reviews.vocab.itos)} unique tokens in the IMDb review sample vocabulary')

print(f'The numericalized token values run from {min(movie_reviews.vocab.stoi.values())} to {max(movie_reviews.vocab.stoi.values())} ')

 

 

# In[25]:

 

 

TokenCounter = lambda review_index : Counter((movie_reviews.train.x)[review_index].data)

TokenCounter(0).items()

 

 

# In[26]:

 

 

TokenCounter(0).keys()

 

 

# In[27]:

 

 

TokenCounter(0).values()

 

 

# In[28]:

 

 

n_terms = len(movie_reviews.vocab.itos)

n_docs = len(movie_reviews.train.x)

make_token_counter = lambda review_index: Counter(movie_reviews.train.x[review_index].data)

def count_vectorizer(review_index,n_terms = n_terms,make_token_counter = make_token_counter):

    # input: review index, n_terms, and tokenizer function

    # output: embedding vector for the review

    embedding_vector = np.zeros(n_terms)       

    keys = list(make_token_counter(review_index).keys())

    values = list(make_token_counter(review_index).values())

    embedding_vector[keys] = values

    return embedding_vector

 

# make the embedding vector for the first review

embedding_vector = count_vectorizer(0)

 

 

# In[29]:

 

 

print(f'The review is embedded in a {len(embedding_vector)} dimensional vector')

embedding_vector

 

 

# In[30]:

 

 

# Define a function to build the full document-term matrix

print(f'there are {n_docs} reviews, and {n_terms} unique tokens in the vocabulary')

def make_full_doc_term_matrix(count_vectorizer,n_terms=n_terms,n_docs=n_docs):

   

    # loop through the movie reviews

    for doc_index in range(n_docs):

       

        # make the embedding vector for the current review

        embedding_vector = count_vectorizer(doc_index,n_terms)   

            

        # append the embedding vector to the document-term matrix

        if(doc_index == 0):

            A = embedding_vector

        else:

            A = np.vstack((A,embedding_vector))

           

    # return the document-term matrix

    return A

 

# Build the full document term matrix for the movie_reviews training set

A = make_full_doc_term_matrix(count_vectorizer)

 

 

# In[31]:

 

 

NNZ = np.count_nonzero(A)

sparsity = (A.size-NNZ)/A.size

print(f'Only {NNZ} of the {A.size} elements in the document-term matrix are nonzero')

print(f'The sparsity of the document-term matrix is {sparsity}')

 

 

# In[32]:

 

 

fig = plt.figure()

plt.spy(A, markersize=0.10, aspect = 'auto')

fig.set_size_inches(8,6)

fig.savefig('doc_term_matrix.png', dpi=800)

 

 

# In[33]:

 

 

# construct the document-term matrix in CSR format

# i.e. return (values, column_indices, row_pointer)

def get_doc_term_matrix(text_list, n_terms):

   

    # inputs:

    #    text_list, a TextList object

    #    n_terms, the number of tokens in our IMDb vocabulary

   

    # output:

    #    the CSR format sparse representation of the document-term matrix in the form of a

    #    scipy.sparse.csr.csr_matrix object

 

   

    # initialize arrays

    values = []

    column_indices = []

    row_pointer = []

    row_pointer.append(0)

 

    # from the TextList object

    for _, doc in enumerate(text_list):

        feature_counter = Counter(doc.data)

        column_indices.extend(feature_counter.keys())

        values.extend(feature_counter.values())

        # Tack on N (number of nonzero elements in the matrix) to the end of the row_pointer array

        row_pointer.append(len(values))

       

    return scipy.sparse.csr_matrix((values, column_indices, row_pointer),

                                   shape=(len(row_pointer) - 1, n_terms),

                                   dtype=int)

 

 

# In[34]:

 

 

get_ipython().run_cell_magic('time', '', 'train_doc_term = get_doc_term_matrix(movie_reviews.train.x, len(movie_reviews.vocab.itos))')

 

 

# In[35]:

 

 

type(train_doc_term)

 

 

# In[36]:

 

 

train_doc_term.shape

 

 

# In[37]:

 

 

get_ipython().run_cell_magic('time', '', 'valid_doc_term = get_doc_term_matrix(movie_reviews.valid.x, len(movie_reviews.vocab.itos))')

 

 

# In[38]:

 

 

type(valid_doc_term)

 

 

# In[39]:

 

 

valid_doc_term.shape

 

 

# In[40]:

 

 

def CSR_to_full(values, column_indices, row_ptr, m,n):

    A = zeros(m,n)

    for row in range(n):

        if row_ptr is not null:

            A[row,column_indices[row_ptr[row]:row_ptr[row+1]]] = values[row_ptr[row]:row_ptr[row+1]]

    return A

 

 

# In[41]:

 

 

valid_doc_term

 

 

# In[42]:

 

 

valid_doc_term.todense()[:10,:10]

 

 

# In[45]:

 

 

review = movie_reviews.valid.x[1]

 

 

# In[44]:

 

 

review

 

 

# In[46]:

 

 

#Print how many times "it" occurs in the review list

print(movie_reviews.vocab.stoi["it"])

 

 

# In[49]:

 

 

valid_doc_term[1,17]

 

 

# In[50]:

 

 

valid_doc_term[1]

 

 

# In[51]:

 

 

valid_doc_term[1].sum()

 

 

# In[50]:

 

 

len(set(review.data))

 

 

# In[52]:

 

 

review.data

 

 

# In[52]:

 

 

word_list = [movie_reviews.vocab.itos[a] for a in review.data]

print(word_list)

 

 

# In[53]:

 

 

#confirm that review has 81 distinct tokens

len(set(review.data))

 

 

# In[53]:

 

 

reconstructed_text = ' '.join(word_list)

print(reconstructed_text)

 

 

# In[54]:

 

 

dir(movie_reviews)

 

 

# In[55]:

 

 

movie_reviews.y.c

 

 

# In[56]:

 

 

movie_reviews.y.classes

 

 

# In[57]:

 

 

positive = movie_reviews.y.c2i['positive']

negative = movie_reviews.y.c2i['negative']

print(f'Integer representations:  positive: {positive}, negative: {negative}')

 

 

# In[58]:

 

 

x = train_doc_term

y = movie_reviews.train.y

valid_y = movie_reviews.valid.y

v = movie_reviews.vocab

 

 

# In[59]:

 

 

x.shape

 

 

# In[60]:

 

 

C1 = np.squeeze(np.asarray(x[y.items==positive].sum(0)))

C0 = np.squeeze(np.asarray(x[y.items==negative].sum(0)))

 

 

# In[61]:

 

 

print(C1[:10])

print(C0[:10])

 

 

# In[62]:

 

 

# Exercise: How often does the word "love" appear in neg vs. pos reviews?

ind = v.stoi['love']

pos_counts = C1[ind]

neg_counts = C0[ind]

print(f'The word "love" appears {pos_counts} and {neg_counts} times in positive and negative documents, respectively')

 

 

# In[63]:

 

 

# Exercise: How often does the word "hate" appear in neg vs. pos reviews?

ind = v.stoi['hate']

pos_counts = C1[ind]

neg_counts = C0[ind]

print(f'The word "hate" appears {pos_counts} and {neg_counts} times in positive and negative documents, respectively')

 

 

# In[64]:

 

 

index = v.stoi['hated']

a = np.argwhere((x[:,index] > 0))[:,0]

print(a)

b = np.argwhere(y.items==positive)[:,0]

print(b)

c = list(set(a).intersection(set(b)))[0]

review = movie_reviews.train.x[c]

review.text

 

 

# In[65]:

 

 

index = v.stoi['loved']

a = np.argwhere((x[:,index] > 0))[:,0]

print(a)

b = np.argwhere(y.items==negative)[:,0]

print(b)

c = list(set(a).intersection(set(b)))[0]

review = movie_reviews.train.x[c]

review.text

 

 

# In[66]:

 

 

L1 = (C1+1) / ((y.items==positive).sum() + 1)

L0 = (C0+1) / ((y.items==negative).sum() + 1)

 

 

# In[67]:

 

 

R = np.log(L1/L0)

print(R)

 

 

# In[68]:

 

 

n_tokens = 10

highest_R = np.argpartition(R, -n_tokens)[-n_tokens:]

lowest_R = np.argpartition(R, n_tokens)[:n_tokens]

 

 

# In[69]:

 

 

print(f'Highest {n_tokens} log-count ratios: {R[list(highest_R)]}\n')

print(f'Lowest {n_tokens} log-count ratios: {R[list(lowest_R)]}')

 

 

# In[70]:

 

 

highest_R

 

 

# In[71]:

 

 

[v.itos[k] for k in highest_R]

 

 

# In[72]:

 

 

token = 'biko'

train_doc_term[:,v.stoi[token]]

 

 

# In[73]:

 

 

 

index = np.argmax(train_doc_term[:,v.stoi[token]])

n_times = train_doc_term[index,v.stoi[token]]

print(f'review # {index} has {n_times} occurrences of "{token}"\n')

print(movie_reviews.train.x[index].text)

 

 

# In[74]:

 

 

lowest_R

 

 

# In[75]:

 

 

[v.itos[k] for k in lowest_R]

 

 

# In[76]:

 

 

token = 'soderbergh'

train_doc_term[:,v.stoi[token]]

 

 

# In[77]:

 

 

index = np.argmax(train_doc_term[:,v.stoi[token]])

n_times = train_doc_term[index,v.stoi[token]]

print(f'review # {index} has {n_times} occurrences of "{token}"\n')

print(movie_reviews.train.x[index].text)

 

 

# In[78]:

 

 

train_doc_term[:,v.stoi[token]]

 

 

# In[79]:

 

 

p = (y.items==positive).mean()

q = (y.items==negative).mean()

print(f'The prior probabilities for positive and negative classes are {p} annd {q}')

 

 

# In[80]:

 

 

b = np.log((y.items==positive).mean() / (y.items==negative).mean())

print(f'The log probability ratio is L = {b}')

 

 

# In[81]:

 

 

W = train_doc_term.sign()

preds_train = (W @ R + b) > 0

train_accuracy = (preds_train == y.items).mean()

print(f'The prediction accuracy for the training set is {train_accuracy}')

 

 

# In[82]:

 

 

W = valid_doc_term.sign()

preds_valid = (W @ R + b) > 0

valid_accuracy = (preds_valid == valid_y.items).mean()

print(f'The prediction accuracy for the validation set is {valid_accuracy}')

 

 

# In[83]:

 

 

path = untar_data(URLs.IMDB)

path.ls()

 

 

# In[84]:

 

 

(path/'train').ls()

 

 

# In[85]:

 

 

get_ipython().run_cell_magic('time', '', "# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!\ncount = 0\nerror = True\nwhile error:\n    try: \n        # Preprocessing steps\n        reviews_full = (TextList.from_folder(path)\n             #  Make a `TextList` object that is a list of `WindowsPath` objects, \n             #     each of which contains the full path to one of the data files.\n             .split_by_folder(valid='test')\n             # Generate a `LabelLists` object that splits files by training and validation folders\n             # Note: .label_from_folder in next line causes the `BrokenProcessPool` error\n             .label_from_folder(classes=['neg', 'pos']))\n             # Create a `CategoryLists` object which contains the data and\n             #   its labels that are derived from folder names\n        error = False\n        print(f'failure count is {count}\\n')    \n    except: # catch *all* exceptions\n        # accumulate failure count\n        count = count + 1\n        print(f'failure count is {count}')")

 

 

# In[86]:

 

 

get_ipython().run_cell_magic('time', '', 'valid_doc_term = get_doc_term_matrix(reviews_full.valid.x, len(reviews_full.vocab.itos))')

 

 

# In[87]:

 

 

get_ipython().run_cell_magic('time', '', 'train_doc_term = get_doc_term_matrix(reviews_full.train.x, len(reviews_full.vocab.itos))')

 

 

# In[88]:

 

 

scipy.sparse.save_npz("train_doc_term.npz", train_doc_term)

 

 

# In[89]:

 

 

scipy.sparse.save_npz("valid_doc_term.npz", valid_doc_term)

 

 

# In[90]:

 

 

with open('reviews_full.pickle', 'wb') as handle:

    pickle.dump(reviews_full, handle, protocol=pickle.HIGHEST_PROTOCOL)

 

 

# In[91]:

 

 

train_doc_term = scipy.sparse.load_npz("train_doc_term.npz")

valid_doc_term = scipy.sparse.load_npz("valid_doc_term.npz")

 

 

# In[92]:

 

 

with open('reviews_full.pickle', 'rb') as handle:

    pickle.load(handle)

 

 

# In[93]:

 

 

type(reviews_full)

 

 

# In[94]:

 

 

type(reviews_full.valid)

 

 

# In[95]:

 

 

print(reviews_full.vocab)

 

 

# In[96]:

 

 

full_vocab = reviews_full.vocab

 

 

# In[97]:

 

 

full_vocab.itos[100:110]

 

 

# In[98]:

 

 

reviews_full.valid

 

 

# In[99]:

 

 

type(reviews_full.valid.x[0])

 

 

# In[100]:

 

 

reviews_full.valid.x[0].text

 

 

# In[101]:

 

 

reviews_full.valid.x[0].data

 

 

# In[102]:

 

 

reviews_full.valid.x.items

 

 

# In[103]:

 

 

type(reviews_full.valid.y)

 

 

# In[104]:

 

 

type(reviews_full.valid.y[0])

 

 

# In[105]:

 

 

reviews_full.valid.y[0]

 

 

# In[106]:

 

 

reviews_full.valid.y.items

 

 

# In[107]:

 

 

reviews_full.valid.y[0]

 

 

# In[108]:

 

 

reviews_full.valid.y.classes

 

 

# In[109]:

 

 

reviews_full.valid.y.c

 

 

# In[110]:

 

 

reviews_full.valid.y.c2i

 

 

# In[111]:

 

 

reviews_full.valid.y[0].data

 

 

# In[112]:

 

 

reviews_full.valid.y[0].obj

 

 

# In[113]:

 

 

len(reviews_full.train), len(reviews_full.valid)

 

 

# In[114]:

 

 

x=train_doc_term

y=reviews_full.train.y

valid_y = reviews_full.valid.y.items

 

 

# In[115]:

 

 

x

 

 

# In[116]:

 

 

positive = y.c2i['pos']

negative = y.c2i['neg']

 

 

# In[117]:

 

 

C0 = np.squeeze(np.asarray(x[y.items==negative].sum(0)))

C1 = np.squeeze(np.asarray(x[y.items==positive].sum(0)))

 

 

# In[118]:

 

 

C0

 

 

# In[119]:

 

 

C1

 

 

# In[120]:

 

 

L1 = (C1+1) / ((y.items==positive).sum() + 1)

L0 = (C0+1) / ((y.items==negative).sum() + 1)

 

 

# In[121]:

 

 

R = np.log(L1/L0)

 

 

# In[122]:

 

 

R[full_vocab.stoi['hated']]

 

 

# In[123]:

 

 

R[full_vocab.stoi['loved']]

 

 

# In[124]:

 

 

R[full_vocab.stoi['liked']]

 

 

# In[125]:

 

 

R[full_vocab.stoi['worst']]

 

 

# In[126]:

 

 

R[full_vocab.stoi['best']]

 

 

# In[127]:

 

 

b = np.log((y.items==positive).mean() / (y.items==negative).mean())

print(f'The bias term b is {b}')

 

 

# In[128]:

 

 

# predict labels for the validation data

W = valid_doc_term.sign()

preds = (W @ R + b) > 0

valid_accuracy = (preds == valid_y).mean()

print(f'Validation accuracy is {valid_accuracy} for the full data set')

 

 

# In[129]:

 

 

from sklearn.linear_model import LogisticRegression

 

 

# In[130]:

 

 

m = LogisticRegression(C=0.1, dual=False,solver = 'liblinear')

# 'liblinear' and 'newton-cg' solvers both get 0.88328 accuracy

# 'sag', 'saga', and 'lbfgs' don't converge

m.fit(train_doc_term, y.items.astype(int))

preds = m.predict(valid_doc_term)

valid_accuracy = (preds==valid_y).mean()

print(f'Validation accuracy is {valid_accuracy} using the full doc-term matrix')

 

 

# In[131]:

 

 

m = LogisticRegression(C=0.1, dual=False,solver = 'liblinear')

m.fit(train_doc_term.sign(), y.items.astype(int))

preds = m.predict(valid_doc_term.sign())

valid_accuracy = (preds==valid_y).mean()

print(f'Validation accuracy is {valid_accuracy} using the binarized doc-term matrix')

 

 

# In[132]:

 

 

path = untar_data(URLs.IMDB_SAMPLE)

 

 

# In[133]:

 

 

get_ipython().run_cell_magic('time', '', "# throws `BrokenProcessPool' Error sometimes. Keep trying `till it works!\n\ncount = 0\nerror = True\nwhile error:\n    try: \n        # Preprocessing steps\n        movie_reviews = (TextList.from_csv(path, 'texts.csv', cols='text')\n                .split_from_df(col=2)\n                .label_from_df(cols=0))\n\n        error = False\n        print(f'failure count is {count}\\n')    \n    except: # catch *all* exceptions\n        # accumulate failure count\n        count = count + 1\n        print(f'failure count is {count}')")

 

 

# In[134]:

 

 

vocab_sample = movie_reviews.vocab.itos

vocab_len = len(vocab_sample)

print(f'IMDb_sample vocabulary has {vocab_len} tokens')

 

 

# In[135]:

 

 

min_n=1

max_n=3

 

j_indices = []

indptr = []

values = []

indptr.append(0)

num_tokens = vocab_len

 

itongram = dict()

ngramtoi = dict()

 

 

# In[136]:

 

 

get_ipython().run_cell_magic('time', '', 'for i, doc in enumerate(movie_reviews.train.x):\n    feature_counter = Counter(doc.data)\n    j_indices.extend(feature_counter.keys())\n    values.extend(feature_counter.values())\n    this_doc_ngrams = list()\n\n    m = 0\n    for n in range(min_n, max_n + 1):\n        for k in range(vocab_len - n + 1):\n            ngram = doc.data[k: k + n]\n            if str(ngram) not in ngramtoi:\n                if len(ngram)==1:\n                    num = ngram[0]\n                    ngramtoi[str(ngram)] = num\n                    itongram[num] = ngram\n                else:\n                    ngramtoi[str(ngram)] = num_tokens\n                    itongram[num_tokens] = ngram\n                    num_tokens += 1\n            this_doc_ngrams.append(ngramtoi[str(ngram)])\n            m += 1\n\n    ngram_counter = Counter(this_doc_ngrams)\n    j_indices.extend(ngram_counter.keys())\n    values.extend(ngram_counter.values())\n    indptr.append(len(j_indices))')

 

 

# In[137]:

 

 

get_ipython().run_cell_magic('time', '', 'train_ngram_doc_matrix = scipy.sparse.csr_matrix((values, j_indices, indptr),\n                                   shape=(len(indptr) - 1, len(ngramtoi)),\n                                   dtype=int)')

 

 

# In[138]:

 

 

train_ngram_doc_matrix

 

 

# In[139]:

 

 

len(ngramtoi), len(itongram)

 

 

# In[140]:

 

 

itongram[20005]

 

 

# In[141]:

 

 

ngramtoi[str(itongram[20005])]

 

 

# In[142]:

 

 

vocab_sample[125],vocab_sample[340],vocab_sample[10],

 

 

# In[143]:

 

 

itongram[100000]