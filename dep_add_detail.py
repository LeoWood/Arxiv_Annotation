#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   dep_add_detail.py
@Time    :   2020/05/27 18:10:45
@Author  :   Leo Wood 
@Contact :   leowood@foxmail.com
@Desc    :   None
'''

from tqdm import tqdm
from pySql import pySql
import json
from dep_relation import dep_relation
import spacy
nlp = spacy.load("en_core_sci_sm")

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


if __name__ == '__main__':
    # 语法关系映射
    dep_dict = {'compound': '组合关系', 'amod': '修饰关系', 'conj': '连接关系', 'nmod': '修饰关系', 'appos': '修饰关系', 'poss': '修饰关系',
                'acl': '修饰关系'}

    # 获取本地术语表
    a = []
    term_dict = {}
    with open('../keywords_all(new)(modified).txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            a.append(line)
            term_dict[line] = 1

    max_num = max([len(lines.split(' ')) for lines in a])

    ## 读取数据库信息
    with open('db_info.json', 'r', encoding='utf-8') as f:
        db = json.load(f)
    db_info = db['ArxivSearch_bak_1203']
    # db_info = db['ArxivSearch_New']
    db_server = pySql(ip=db_info['ip'], user=db_info['user'], pwd=db_info['pwd'], db=db_info['db'])

    # 获取relaitionType映射
    relationType = {}
    sql = "SELECT * FROM relationType"
    df = db_server.read_sql(sql)
    for i in range(len(df)):
        relationType[str(df.iloc[i]['typeName'])] = df.iloc[i]['relationTypeId']
    print(relationType)

    # # 语法关系映射
    # dep_dict = {'compound': '组合关系', 'amod': '修饰关系', 'conj': '连接关系', 'nmod': '修饰关系', 'appos': '修饰关系', 'poss': '修饰关系',
    #             'acl': '修饰关系'}
    
    # dep_detail = {}
    

    ## 读取当前全部的paperID
    sql = "select paperId from paper order by paperId desc"
    df = db_server.read_sql(sql)
    paperId = df.iloc[0]['paperId']

    ## 读取当前全部的relaitionID
    sql = "select relationId from relation order by relationId desc"
    df = db_server.read_sql(sql)
    relationId = df.iloc[0]['relationId']

    for i in tqdm(range(2,76444)):
        ## 删掉17,18,20三种关系
        # sql = "delete from relation where paperId={} and relationTypeId in (17,18,20)".format(i)
        # db_server.write_sql(sql)

        ## 获取当前paper的所有sentence
        sql = "select detail,sentenceId from sentence where paperId={}".format(i)
        df = db_server.read_sql(sql)

        for j in range(len(df)):
            sen = df.iloc[j]['detail']
            sen_id = df.iloc[j]['sentenceId']
            terms = set(max_match(sen, term_dict, max_num))
            deps = dep_relation(terms, sen, nlp)
            ## 对当前句子的dep进行处理
            for dep in deps:
                if dep[2] in relationType.keys():
                    relation = {}
                    relationId += 1
                    relation['relationId'] = relationId
                    relation['sentenceId'] = sen_id
                    relation['paperId'] = i
                    if dep[2] == 'conj':
                        relation['termPre'] = dep[1].replace("'", "''")
                        relation['conj'] = dep[3].replace("'", "''")
                        relation['termSuf'] = dep[0].replace("'", "''")
                        relation['relationTypeId'] = relationType[dep[2]]
                        relation['isSemantic'] = 0
                    else:
                        relation['termPre'] = dep[1].replace("'", "''")
                        relation['conj'] = ''
                        relation['termSuf'] = dep[0].replace("'", "''")
                        relation['relationTypeId'] = relationType[dep[2]]
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

    db_server.close()
