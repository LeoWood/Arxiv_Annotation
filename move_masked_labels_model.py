#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2019/9/6 10:00
from bert_base.client import BertClient
import time
import spacy
import socket
from move_masked_sentence_model import predict_move_masked_sentence_model


nlp = spacy.load("en_core_sci_sm")

def msm_results(text):
    t1 = time.time()
    # 调用mask_abs_sen预测初步得分
    results, zhuanyi_sens = predict_move_masked_sentence_model(text)
    if not results:
        return 0,0
    sentences = [str(sen) for sen in nlp(text).sents]
    ##　小于４句话，直接返回结果
    if len(sentences) < 4:
        return results,sentences
    # print(results)
    # print('msm用时：', time.time() - t1)
    # t1 = time.time()

    cla = {'0': 'Purpose', '1': 'Methods', '2': 'Results', '3': 'Conclusions', '4': 'Background'}

    # 获取最高两个得分
    max_list = []
    label_dict = {}
    k = 0
    for a in results:
        m = max(a)
        max_list.append(m)
        label_dict[k] = cla[str(a.index(max(a)))]
        k += 1
    sor_max_list = sorted(max_list, reverse=True)
    i_0 = max_list.index(sor_max_list[0])
    i_1 = max_list.index(sor_max_list[1])
    i_2 = max_list.index(sor_max_list[2])

    rs, ls = [i_0, i_1, i_2], [label_dict[i_0], label_dict[i_1], label_dict[i_2]]  # 获取得分最高三个
    # print(rs,ls)

    # print('获取得分用时：', time.time() - t1)

    predict_examples = []
    # 利用得分构造新的输入
    k = 0
    for sen in sentences:
        new_sens = {}
        for j in range(len(sentences)):
            new_sens[j] = '[...]. '  # 初始化一个全是[...].的数组
        new_sens[rs[0]] = '[' + ls[0] + ']. '  # 选出来的两个位置用label替代
        new_sens[rs[1]] = '[' + ls[1] + ']. '
        new_sens[rs[2]] = '[' + ls[2] + ']. '

        new_sens[k] = sentences[k] + ' '  # 句子本身保留
        masked_label_abs = ''
        for j in range(len(sentences)):
            masked_label_abs += new_sens[j]  # 得到每一句的mask_label表示
        # print(masked_label_abs)
        # guid = 'test-' + str(k)
        predict_examples.append(masked_label_abs)
        k += 1
    return predict_examples,sentences


def predict_move_masked_labels_model(text):
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.settimeout(1)
    flag = 0
    try:
        sk.connect(('159.226.125.191', 5720))
        flag = 1
    except:
        pass
    sk.close()

    if not flag:
        print('mask labels server not available')
        return 0, 0

    predict_examples,sentences = msm_results(text)
    if not sentences:
        return 0,0

    ## 小于4句话，直接用msm结果
    if len(sentences)<4:
        i = 0
        labels = []
        for a in predict_examples:
            labels.append(a.index(max(a)))
            i += 1
        return labels,sentences


    with BertClient(port=5720,port_out=5721,show_server_config=False, check_version=False, check_length=False, mode='CLASS') as bc:
        results = bc.encode(predict_examples)

    i = 0
    labels = []
    for re in results:
        for a in re['score']:
            labels.append(a.index(max(a)))
            i += 1

    return labels,sentences



if __name__ == '__main__':

    while True:
        text = input()
        t1 = time.time()
        predict_move_masked_labels_model(text)
        print('用时: ', time.time() - t1)
    pass