#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2019/9/6 10:00
from bert_base.client import BertClient
import time
import spacy
nlp = spacy.load("en_core_sci_sm")
import numpy as np
import socket


def predict_sentences(sentences):
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.settimeout(1)
    flag = 0
    try:
        sk.connect(('159.226.125.191', 5730))
        flag = 1
    except:
        pass
    sk.close()

    if not flag:
        print('server not available')
        return 0, 0
    else:

        predict_examples = []
        i = 0
        for sen in sentences:
            masked_abs = ''
            mask = ' AAAA ' + 'AAAA ' * 23 + 'AAAA. '
            if i == 0:
                masked_abs = mask + ' '.join(sentences[1:])
            elif i == len(sentences) - 1:
                masked_abs = ' '.join(sentences[:-1]) + mask
            else:
                masked_abs = ' '.join(sentences[:i]) + mask + ' '.join(sentences[i + 1:])

            predict_examples.append(masked_abs)
            i += 1
        for sen in sentences:
            predict_examples.append(sen)
            i += 1

        with BertClient(port=5730, port_out=5731, show_server_config=False, check_version=False, check_length=False,
                        mode='CLASS') as bc:
            result = bc.encode(predict_examples)

        result1 = []
        result2 = []
        j = 1
        for re in result:
            for r in re['score']:
                if j <= i / 2:
                    result1.append(r)
                else:
                    result2.append(r)
                j += 1

        results = np.array(result1) + np.array(result2)
        # print('results: ',results)
        results = results.tolist()
        # print('results: ',results)


    return results, sentences

def predict_move_masked_sentence_model(text):
    sentences = [str(sen) for sen in nlp(text).sents]
    results, sentences = predict_sentences(sentences)
    if not results:
        return 0,0
    else:
        cla = {'0': 'Objective', '1': 'Methods', '2': 'Results', '3': 'Conclusions', '4': 'Background'}
        i = 0
        for a in results:
            print('<<' + cla[str(a.index(max(a)))] + '>> ' + sentences[i])
            i += 1
        zhuanyi_sens = []
        for sen in sentences:
            sen = sen.replace('&', '&amp;').replace('"', '&quot;').replace("'", "&apos;").replace('<', '&lt;').replace(
                '>',
                '&gt;')
            zhuanyi_sens.append(sen)
        return results, zhuanyi_sens


def predict_move_masked_sentence_model_multi(absts):
    nums = []
    sentences = []
    for abst in absts:
        sens = [str(sen) for sen in nlp(abst).sents]
        nums.append(len(sens))
        sentences += sens
    results, sentences = predict_sentences(sentences)
    if not results:
        return {'results':'抱歉，当前服务暂未开放！'}
    else:
        cla = {'0': 'Objective', '1': 'Methods', '2': 'Results', '3': 'Conclusions', '4': 'Background'}
        i = 0
        results_return = {'results':[]}
        for num in nums:
            scores_temp = results[i:i+num] # 当前摘要预测分值
            abst_temp = sentences[i:i+num] # 当前摘要内容
            results_temp = {}
            for j in range(len(abst_temp)):
                a = scores_temp[j] # 当前句子分值
                results_temp[abst_temp[j]] = cla[str(a.index(max(a)))]
            results_return['results'].append(results_temp)
        return results_return # 返回所有摘要标注结果


if __name__ == '__main__':

    while True:
        text = input()
        t1 = time.time()
        predict_move_masked_sentence_model(text)
        print('用时: ', time.time() - t1)
    pass