#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/4/24 17:19


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

    while True:
        text = input()
        for sen in nlp(text).sents:
            print('Sentence:')
            print(sen)
            sen = str(sen)
            terms = set(max_match(sen, term_dict, max_num))
            deps = dep_relation(terms, sen, nlp)
            print('Terms:')
            print(terms)
            print('Relations:')
            print(deps)
            print()