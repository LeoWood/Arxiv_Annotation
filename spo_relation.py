#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/4/9 10:27



def spo_relation(txt, nlp):
    doc = nlp(txt)
    # 找谓语
    root = ''
    root_lemma = ''
    for w in doc:
        if w.dep_ == "ROOT":
            root = w.text
            root_lemma = w.lemma_
    # 找主语、宾语
    subject = ''
    object = ''
    for chunk in doc.noun_chunks:
        if chunk.root.head.text == root:
            if chunk.root.dep_ == 'nsubj' or chunk.root.dep_ == 'csubj':
                subject = chunk.text
            if chunk.root.dep_ == 'attr':
                object = chunk.text
            if chunk.root.dep_ == 'nsubjpass' or chunk.root.dep_ == 'csubjpass':
                object = chunk.text
            if chunk.root.dep_ == 'dobj' or chunk.root.dep_ == 'pobj':
                object = chunk.text

    if subject == '-PRON-':
        subject = ''
    if object == '-PRON-':
        object = ''

    return subject, root_lemma, object

if __name__ == '__main__':
    pass