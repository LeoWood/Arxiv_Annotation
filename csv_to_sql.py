#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/4/9 10:19

import sys
import os
import pymssql
import pandas as pd
import numpy as np
import json
import time

from tqdm import tqdm
import spacy
nlp = spacy.load("en_core_sci_sm")

from move_masked_labels_model import predict_move_masked_labels_model
from itertools import combinations
from ner import annotation_ner
from ner_physics.ner_physics import predict_second_category
from predict_def import predict_def
from spo_relation import spo_relation
from dep_relation import dep_relation

from pySql import pySql

os.chdir(sys.path[0])  # 设置工作路径为当前文件夹


def max_match(txt, ano_dict, max_num):
    word_list = [str(t) for t in nlp(txt)]
    new_word_list = []
    N = len(word_list)
    k = max_num
    i = 0
    while i < N:
        if i <= N - k:
            j = k
            while j > 0:
                token_tmp = ' '.join(word_list[i:i + j])
                # print(token_tmp)
                if token_tmp in ano_dict.keys() or token_tmp.lower() in ano_dict.keys():
                    # print(token_tmp,'！!！!!!！!!!！!！!！!')
                    new_word_list.append(token_tmp)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                i += 1
        else:
            j = N - i
            while j > 0:
                token_tmp = ' '.join(word_list[i:i + j])
                # print(token_tmp)
                if token_tmp in ano_dict.keys() or token_tmp.lower() in ano_dict.keys():
                    # print(token_tmp, '！!！!!!！!!!！!！!！!')
                    new_word_list.append(token_tmp)
                    i += j
                    break
                else:
                    j -= 1
            if j == 0:
                i += 1
    return new_word_list


def csv_to_sql(data,db_server,db_server_sw):

    # 获取数据库当前最大id
    sql = "select paperId from paper order by paperId desc"
    df = db_server.read_sql(sql)
    if not len(df):
        paperId = 0
    else:
        paperId = df.iloc[0]['paperId']

    sql = "select sentenceId from sentence order by sentenceId desc"
    df = db_server.read_sql(sql)
    if not len(df):
        sentenceId = 0
    else:
        sentenceId = df.iloc[0]['sentenceId']

    sql = "select termId from term order by termId desc"
    df = db_server.read_sql(sql)
    if not len(df):
        termId = 0
    else:
        termId = df.iloc[0]['termId']

    sql = "select relationId from relation order by relationId desc"
    df = db_server.read_sql(sql)
    if not len(df):
        relationId = 0
    else:
        relationId = df.iloc[0]['relationId']

    sql = "select entityId from entity order by entityId desc"
    df = db_server.read_sql(sql)
    if not len(df):
        entityId = 0
    else:
        entityId = df.iloc[0]['entityId']

    # 获取termType映射
    termType = {}
    sql = "SELECT * FROM termType"
    df = db_server.read_sql(sql)
    for i in range(len(df)):
        termType[str(df.iloc[i]['topCategory']) + ',' + str(df.iloc[i]['secondCategory'])] = df.iloc[i]['termTypeId']

    # 获取relaitionType映射
    relationType = {}
    sql = "SELECT * FROM relationType"
    df = db_server.read_sql(sql)
    for i in range(len(df)):
        relationType[str(df.iloc[i]['typeName'])] = df.iloc[i]['relationTypeId']

    # 语法关系映射
    dep_dict = {'compound': '组合关系', 'amod': '修饰关系', 'conj': '连接关系', 'nmod': '修饰关系', 'appos': '修饰关系', 'poss': '修饰关系',
                'acl': '修饰关系'}

    # 获取本地术语表
    a = []
    term_dict = {}
    with open('keywords_all(new)(modified).txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            a.append(line)
            term_dict[line] = 1

    max_num = max([len(lines.split(' ')) for lines in a])

    # 获取实体识别类型映射
    entityType = {}
    sql = "SELECT * FROM entityType"
    df = db_server.read_sql(sql)
    for k in range(len(df)):
        if df.iloc[k]['deleteTag'] != 1:
            entityType[str(df.iloc[k]['typeName'])] = df.iloc[k]['entityTypeId']
    # print(entityType)
    # print(len(entityType))
    # exit()



    for i in tqdm(range(len(data))):
        this_abst = data.iloc[i]['abstracts']
        ## 插入paper表
        paperId += 1
        row = []
        row.append(paperId)
        row.append(data.iloc[i]['title'].replace("'", "''"))
        row.append(this_abst.replace("'", "''"))
        row.append(data.iloc[i]['authors'].replace("'", "''"))
        row.append(str(data.iloc[i]['csoaid']))
        row.append(data.iloc[i]['createtime'])
        row.append(data.iloc[i]['subjects'])
        if len(this_abst) > 1000:
            row.append(1)
        else:
            row.append(0)
        row.append(data.iloc[i]['createtime'])
        row = tuple(row)
        sql = "INSERT INTO paper VALUES (%d, '%s', '%s', '%s', '%s', '%s', '%s', %d, '%s')" % row
        # print(sql)
        db_server.write_sql(sql)

        ## scienceWise插入term表
        ner_phy_results = predict_second_category(this_abst)
        if ner_phy_results:
            for key,value in ner_phy_results.items():
                term = {}
                termId += 1
                term['termId'] = termId
                term['sentenceId'] = 0
                term['paperId'] = paperId
                term['name'] = key.replace("'", "''")
                ## 如果数据库中没有当前的范畴
                if value not in termType.keys():
                    sql = "SELECT * FROM termType"
                    df = db_server.read_sql(sql)
                    max_termTypeId = max(df['termTypeId'].tolist())
                ## 当前范畴写入数据库
                    row = []
                    row.append(max_termTypeId + 1)
                    row.append(value.split(',')[0])
                    row.append(value.split(',')[1])
                    row = tuple(row)
                    sql = "INSERT INTO termType VALUES (%d, '%s', '%s')" % row
                    print(sql)
                    db_server.write_sql(sql)
                    termType[value] = max_termTypeId + 1
                ## 写入term表
                term['termTypeId'] = termType[value]
                term['isScienceWise'] = 1
                row = []
                for field in ['termId', 'sentenceId', 'paperId', 'name', 'termTypeId', 'isScienceWise']:
                    row.append(term[field])
                row = tuple(row)
                sql = "INSERT INTO term VALUES (%d, %d, %d, '%s', %d, %d)" % row
                # print(sql)
                db_server.write_sql(sql)

                # scienceWise插入relation表

                sql1 = "select incoming_concept,incoming_relation,category from sw_incoming_relation where incoming_concept='%s'" % key.replace("'", "''")
                # print(key)
                df1 = db_server_sw.read_sql(sql1)
                sql2 = "select category,outgoing_relation,outgoing_concept from sw_outgoing_relation where outgoing_concept='%s'" % key.replace("'", "''")
                df2 = db_server_sw.read_sql(sql2)
                if len(df1):
                    for j in range(len(df1)):
                        relation = {}
                        relationId += 1
                        relation['relationId'] = relationId
                        relation['sentenceId'] = 0
                        relation['paperId'] = paperId
                        relation['conj'] = ''
                        relation['isSemantic'] = 1
                        relation['termPre'] = df1.iloc[j]['incoming_concept'].replace("'", "''")
                        relation['termSuf'] = df1.iloc[j]['category'].replace("'", "''")
                        relation['relationTypeId'] = relationType[df1.iloc[j]['incoming_relation']]
                        row = []
                        for field in ['relationId', 'sentenceId', 'paperId', 'termPre', 'conj', 'termSuf',
                                      'relationTypeId', 'isSemantic']:
                            row.append(relation[field])
                        row = tuple(row)
                        sql = "INSERT INTO relation VALUES (%d, %d, %d, '%s','%s','%s', %d, %d)" % row
                        # print(sql)
                        db_server.write_sql(sql)
                if len(df2):
                    for j in range(len(df2)):
                        relation = {}
                        relationId += 1
                        relation['relationId'] = relationId
                        relation['sentenceId'] = 0
                        relation['paperId'] = paperId
                        relation['conj'] = ''
                        relation['isSemantic'] = 1
                        relation['termPre'] = df2.iloc[j]['category'].replace("'", "''")
                        relation['termSuf'] = df2.iloc[j]['outgoing_concept'].replace("'", "''")
                        relation['relationTypeId'] = relationType[df2.iloc[j]['outgoing_relation']]
                        row = []
                        for field in ['relationId', 'sentenceId', 'paperId', 'termPre', 'conj', 'termSuf',
                                      'relationTypeId', 'isSemantic']:
                            row.append(relation[field])
                        row = tuple(row)
                        sql = "INSERT INTO relation VALUES (%d, %d, %d, '%s','%s','%s', %d, %d)" % row
                        # print(sql)
                        db_server.write_sql(sql)

        ## 当前paper插入sentences表
        # move
        move_labels, sentences = predict_move_masked_labels_model(data.iloc[i]['abstracts'])
        if not move_labels:
            exit()
        # def
        def_labels = predict_def(sentences)
        if not def_labels:
            exit()

        j = 0
        for sen in sentences:
            # sentence
            sentence = {}
            sentenceId += 1
            sentence['sentenceId'] = sentenceId
            sentence['paperId'] = paperId
            sentence['moveTypeId'] = move_labels[j]
            sentence['sentenceOrder'] = j + 1
            sentence['detail'] = sen.replace("'", "''")
            sentence['subject'], sentence['predicate'], sentence['object'] = spo_relation(sen,nlp)
            sentence['subject'] = sentence['subject'].replace("'", "''")
            sentence['predicate'] = sentence['predicate'].replace("'", "''")
            sentence['object'] = sentence['object'].replace("'", "''")
            sentence['isDefinition'] = def_labels[j]

            row = []
            for field in ['sentenceId', 'paperId', 'moveTypeId', 'sentenceOrder', 'detail', 'subject', 'object','predicate', 'isDefinition']:
                row.append(sentence[field])
            row = tuple(row)

            sql = "INSERT INTO sentence VALUES (%d, %d, %d, %d, '%s', '%s', '%s', '%s', %d)" % row
            # print(sql)
            db_server.write_sql(sql)

            j += 1

            # 每一句的term抽取
            terms = set(max_match(sen, term_dict, max_num))
            for t in terms:
                term = {}
                termId += 1
                term['termId'] = termId
                term['sentenceId'] = sentenceId
                term['paperId'] = paperId
                term['name'] = t.replace("'", "''")
                term['termTypeId'] = 0
                term['isScienceWise'] = 0

                row = []
                for field in ['termId', 'sentenceId', 'paperId', 'name', 'termTypeId', 'isScienceWise']:
                    row.append(term[field])
                row = tuple(row)
                sql = "INSERT INTO term VALUES (%d, %d, %d, '%s', %d, %d)" % row
                # print(sql)
                db_server.write_sql(sql)

            # 每一句relation抽取
            # 依存关系
            deps = dep_relation(terms, sen, nlp)
            for dep in deps:
                if dep[2] in dep_dict.keys():
                    relation = {}
                    relationId += 1
                    relation['relationId'] = relationId
                    relation['sentenceId'] = sentenceId
                    relation['paperId'] = paperId
                    if dep[2] == 'conj':
                        relation['termPre'] = dep[1].replace("'", "''")
                        relation['conj'] = dep[3].replace("'", "''")
                        relation['termSuf'] = dep[0].replace("'", "''")
                        relation['relationTypeId'] = relationType[dep_dict[dep[2]]]
                        relation['isSemantic'] = 0
                    else:
                        relation['termPre'] = dep[1].replace("'", "''")
                        relation['conj'] = ''
                        relation['termSuf'] = dep[0].replace("'", "''")
                        relation['relationTypeId'] = relationType[dep_dict[dep[2]]]
                        relation['isSemantic'] = 0

                    row = []
                    for field in ['relationId', 'sentenceId', 'paperId', 'termPre', 'conj', 'termSuf',
                                  'relationTypeId', 'isSemantic']:
                        row.append(relation[field])
                    row = tuple(row)
                    # print(row)
                    sql = "INSERT INTO relation VALUES (%d, %d, %d, '%s','%s','%s', %d, %d)" % row
                    # print(sql)
                    db_server.write_sql(sql)

            # 共现关系
            co_ccur = set(list(combinations(terms, 2)))
            for co in co_ccur:
                relation = {}
                relationId += 1
                relation['relationId'] = relationId
                relation['sentenceId'] = sentenceId
                relation['paperId'] = paperId
                relation['termPre'] = co[0].replace("'", "''")
                relation['conj'] = ''
                relation['termSuf'] = co[1].replace("'", "''")
                relation['relationTypeId'] = relationType['共现关系']
                relation['isSemantic'] = 0

                row = []
                for field in ['relationId', 'sentenceId', 'paperId', 'termPre', 'conj', 'termSuf', 'relationTypeId',
                              'isSemantic']:
                    row.append(relation[field])
                row = tuple(row)
                # print(row)
                sql = "INSERT INTO relation VALUES (%d, %d, %d, '%s','%s','%s', %d, %d)" % row
                # print(sql)
                db_server.write_sql(sql)

            # 每一句实体识别
            sen = sen.replace('\\', ' ').replace('$', ' ').replace('~', ' ')
            ano_dict = annotation_ner(sen)

            for key, value in ano_dict.items():

                entity = {}
                entityId += 1
                entity['entityId'] = entityId
                entity['sentenceId'] = sentenceId
                entity['paperId'] = paperId
                entity['name'] = key.replace("'", "''")

                if value in entityType.keys():
                    entity['entityTypeId'] = entityType[value]

                    row = []
                    for field in ['entityId', 'sentenceId', 'paperId', 'name', 'entityTypeId']:
                        row.append(entity[field])
                    row = tuple(row)
                    # print(row)
                    sql = "INSERT INTO entity VALUES (%d, %d, %d, '%s', %d)" % row
                    # print(sql)
                    db_server.write_sql(sql)

    db_server.close()


if __name__ == '__main__':
    ## 读取数据库信息
    with open('db_info.json', 'r', encoding='utf-8') as f:
        db = json.load(f)
    db_info = db['ArxivSearch_bak_1203']
    # db_info = db['ArxivSearch_New']
    db_server = pySql(ip=db_info['ip'], user=db_info['user'], pwd=db_info['pwd'], db=db_info['db'])

    db_info_sw = db['arxiv_physics_article']
    db_server_sw = pySql(ip=db_info_sw['ip'], user=db_info_sw['user'], pwd=db_info_sw['pwd'], db=db_info_sw['db'])
    ## 读取csv
    data = pd.read_csv(r'arxiv_2019.csv',float_precision='round_trip')

    # data = data[13601:]
    # print(data.iloc[1]['csoaid'])

    t0 = time.time()
    csv_to_sql(data,db_server,db_server_sw)
    print(time.time()-t0)



