# -*- coding: utf-8 -*-
# @Time    : 2018-11-26 21:12:23
# @Author  : Liu Huan (liuhuan@mail.las.ac.cn)

import itertools
from nltk.tag.stanford import StanfordNERTagger
from nltk.tokenize import word_tokenize
import spotlight
import en_core_web_sm
import time
from multiprocessing import Process
import multiprocessing
import pymssql
import pandas as pd


def GateResults(conn, text, recordID):
    sql = 'select obj1.objectValue,obj2.typeEn from extr_FileObjects obj1,dict_senstiveWordType obj2 where CAST(obj1.objectType AS NUMERIC) = obj2.id  and recordId = {recordID}'.format(
        recordID=recordID)
    df = pd.read_sql(sql, conn)
    gate_dict = {}
    for i in range(len(df)):
        gate_dict[df.ix[i]['objectValue']] = df.ix[i]['typeEn'].upper()
    return gate_dict


def StanfordMerge(a_list):
    index_list = []
    i = 0
    for key, group in itertools.groupby([a[1] for a in a_list]):
        in_list = []
        in_list.append(key)
        i_list = []
        for j in range(len(list(group))):
            i_list.append(i)
            i += 1
        in_list.append(i_list)
        index_list.append(in_list)
    a_dict = {}
    for in_list in index_list:
        name = ''
        for i in in_list[1]:
            name = name + a_list[i][0] + ' '
        a_dict[name] = in_list[0]
    return a_dict


def StanfordResults(txt):
    # print('st start')
    # t1 = time.time()
    st_dict = {}
    tokenized_text = word_tokenize(txt)
    # Stanford ner
    st = StanfordNERTagger('english.muc.7class.distsim.crf.ser.gz')
    ano_st = st.tag(tokenized_text)
    for key, value in StanfordMerge(ano_st).items():
        if value != 'O':
            st_dict[key] = value
    return st_dict
    # print('st ', time.time()-t1)
    # print('st end')


def DbpediaResults(txt):
    dp_dict = {}
    # print('dp start')
    # t1 = time.time()
    # dp = 1
    try:
        ano_dp = spotlight.annotate('http://159.226.125.180:8080/rest/annotate', txt, confidence=0.4, support=20,spotter='Default')
        # ano_dp = spotlight.annotate('http://api.dbpedia-spotlight.org/en/annotate', txt, confidence=0.4, support=20,spotter='Default')
        for a in ano_dp:
            if a['types'] != '':
                # dp_dict[a['surfaceForm']]=a['types'].split(',')[-1].split(':')[1]
                cla = a['types'].split(',')
                for c in cla:
                    if c.startswith('DBpedia'):
                        dp_dict[a['surfaceForm']] = c.split(':')[1].upper()
                        break
    except:
        print('no dbpedia results')
    # print('dp ', time.time() - t1)
    # print('dp end')
    return dp_dict


def SpacyResults(txt):
    spa_dict = {}
    # print('spa start')
    # t1 = time.time()
    # nlp = spacy.load("en")
    nlp = en_core_web_sm.load()
    ano_spa = nlp(txt)
    for i in ano_spa.ents:
        spa_dict[str(i)] = i.label_
    # print('spa ', time.time() - t1)
    # print('spa end')
    return spa_dict


def anotation_color(txt, ano_dict):
    ano_dict_sup = {}
    tokenized_text = word_tokenize(txt)
    for key, value in ano_dict.items():
        ano_dict_sup[key] = value

    nums = []
    for key, value in ano_dict_sup.items():
        nums.append(len(key.split(' ')))
        if key[-1] == ' ':
            ano_dict[key[:-1]] = value
        # print(key)
    # exit()
    word_list = tokenized_text
    # print(word_list)
    # exit()
    k = max(nums)
    # print('max:', k)
    new_word_list = []

    N = len(word_list)
    # print(N)
    i = 0
    while i < N:
        if i <= N - k:
            token_tmp = []
            j = k
            while j > 0:
                token_tmp = ' '.join(word_list[i:i + j])
                # print(token_tmp)
                if token_tmp in ano_dict.keys():
                    # new_word_list += ['<' + ano_dict[token_tmp] + '>'] + word_list[i:i + j] + [
                    #     '</font>']
                    new_word_list.append('<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">'+ token_tmp + '<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">' + ano_dict[token_tmp]+'</mark>')
                    # print('!!!', token_tmp)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                # print(word_list[i])
                new_word_list += [word_list[i]]
                i += 1
        else:
            # print('-----------------------------------------------------')
            token_tmp = []
            j = N - i
            while j > 0:
                token_tmp = ' '.join(word_list[i:i + j])
                # print(token_tmp)
                if token_tmp in ano_dict.keys():
                    new_word_list.append('<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">'+ token_tmp + '<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">' + ano_dict[token_tmp]+'</mark>')
                    # print('!!!', token_tmp)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                # print(word_list[i])
                new_word_list.append(word_list[i])
                i += 1
        # print(new_word_list)
        # if i>20:
        #     exit()
        # print(i)
    return new_word_list


def anotation_ner(text):
    ano_dict = {}
    st_dict = StanfordResults(text)
    dp_dict = DbpediaResults(text)
    spa_dict = SpacyResults(text)
    # print(spa_dict)
    # print(dp_dict)
    # print(st_dict)


    # spacy标注结果
    for key, value in spa_dict.items():
        value = value.replace('ORG', 'ORGENIZATION').replace('GPE', 'REGION').replace('FAC', 'FACILITIES').replace(
            'LOC',
            'LOCATION')
        ano_dict[key] = value

    # stanford标注结果
    for key, value in st_dict.items():
        flag = 1
        for name in ano_dict.keys():
            if name in key:
                flag = 0
        if flag:
            if key not in ano_dict:
                ano_dict[key] = value

    # dbpedia 标注结果
    for key, value in dp_dict.items():
        ano_dict[key] = value

    # # gate标注结果
    # for key, value in gate_dict.items():
    #     ano_dict[key] = value
    print(ano_dict)
    # html ='<p style="text-align:justify;"><div class="entities" style="line-height: 2.5;text-align:justify;">' + ' '.join(anotation_color(text,ano_dict)) + '</div></p>'
    # for key, value in ano_dict.items():
    #     if key != '\n':
    #         html = html + key + ': ' + value + '</br>'
    return ano_dict


if __name__ == '__main__':
    while True:
        text = input()
        anotation_color(text)
    pass