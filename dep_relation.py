#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/4/9 10:44

from itertools import combinations

def dep_relation(terms, txt, nlp):
    doc = nlp(txt)
    deps = []
    for co in list(combinations(terms, 2)):
        # 取短语最后一个词
        if ' ' in co[0]:
            a = co[0].split(' ')[-1]
        else:
            a = co[0]
        if ' ' in co[1]:
            b = co[1].split(' ')[-1]
        else:
            b = co[1]
        conj = ''
        for w in doc:
            # deps.append([w.text,w.dep_,w.head.text])
            if w.dep_ == 'cc':
                conj = w.text
            if a in w.text:
                if b in w.head.text:
                    if w.dep_ == 'conj':
                        deps.append([co[0], co[1], w.dep_, conj])
                    else:
                        deps.append([co[0], co[1], w.dep_])
            if b in w.text:
                if a in w.head.text:
                    if w.dep_ == 'conj':
                        deps.append([co[1], co[0], w.dep_, conj])
                    else:
                        deps.append([co[1], co[0], w.dep_])
    return deps

if __name__ == '__main__':
    pass