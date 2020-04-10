# -*- coding: utf-8 -*-
# @Time    : 2020/1/10 12:41
# @Author  : 丁良萍
# @FileName: tst.py
# @Software: PyCharm
# @Reference:
# @Introduction:
import re
import jieba
import nltk

import pandas as pd


def Remove_chars(word_str):
    # 利用正则表达式去掉一些一些标点符号之类的符号。
    word_str = re.sub(r'\s+', ' ', word_str)  # trans 多空格 to空格
    word_str = re.sub(r'\n', '', word_str)  # trans 换行 to空格
    word_str = re.sub(r'\t', '', word_str)  # trans Tab to空格
    # word_str=re.sub(r'\d+','',word_str)#去掉数字
    # word_str = re.sub("[\.\!\/_,$%^*(+\"\']+|[+——！，。？;；、~@#￥%～……&*（）±《》]+", "", word_str)

    # word_str = re.sub('\W+', '', word_str).replace("_", '');##不能去掉连字符
    # word_str=re.sub('[a-zA-Z]+','',word_str)##去掉英文字母


    return word_str


##python分离中文和英文的混合字符串



def is_zh(c):
    x = ord(c)
    # Punct & Radicals
    if x >= 0x2e80 and x <= 0x33ff:
        return True

    # Fullwidth Latin Characters
    elif x >= 0xff00 and x <= 0xffef:
        return True

    # CJK Unified Ideographs &
    # CJK Unified Ideographs Extension A
    elif x >= 0x4e00 and x <= 0x9fbb:
        return True
    # CJK Compatibility Ideographs
    elif x >= 0xf900 and x <= 0xfad9:
        return True

    # CJK Unified Ideographs Extension B
    elif x >= 0x20000 and x <= 0x2a6d6:
        return True

    # CJK Compatibility Supplement
    elif x >= 0x2f800 and x <= 0x2fa1d:
        return True

    else:
        return False

def split_zh_en(zh_en_str):
    mark = {"en": 1, "zh": 2}
    zh_en_group = []
    zh_gather = ""###准备一个中文收集器
    en_gather = ""
    zh_status = False

    for c in zh_en_str:
        if not zh_status and is_zh(c):
            zh_status = True
            if en_gather != "":
                zh_en_group.append([mark["en"], en_gather])
                en_gather = ""
        elif not is_zh(c) and zh_status:
            zh_status = False
            if zh_gather != "":
                zh_en_group.append([mark["zh"], zh_gather])
        if zh_status:
            zh_gather += c
        else:
            en_gather += c
            zh_gather = ""

    if en_gather != "":
        zh_en_group.append([mark["en"], en_gather])
    elif zh_gather != "":
        zh_en_group.append([mark["zh"], zh_gather])

    return zh_en_group


##############将英问标点转换为中文标点
def E_trans_to_C(string):
    E_pun = u',!?[]()<>"\''
    C_pun = u'，！？【】（）《》“‘'
    table= {ord(f):ord(t) for f,t in zip(E_pun,C_pun)}
    return string.translate(table)


def text_to_word_list(abs):
    whole_txt = []
    abs = Remove_chars(abs)
    abs = E_trans_to_C(abs)
    abs = abs.lower()
    whole_txt = []
    result = split_zh_en(abs)
    for j, tx in enumerate(result):
        if tx[0] == 1:
            whole_txt.append(result[j][1])
        else:
            whole_txt.extend([tt for tt in tx[1]])

    return whole_txt


if __name__ == '__main__':

    abs='目的 探讨经皮内镜椎间孔入路微创治疗对腰椎间盘突出症的临床疗效。' \
        '方法 抽取至我院就诊的腰椎间盘突出症患者92例,收治的时间为2013年3月15日-2017年3月15日,' \
        '依照计算机随机分组模式,其中一组采取综合治疗方案,另外一组采取经皮内镜微创治疗,分析两组的治疗效果。' \
        '结果 治疗后实验组的腰疼痛与腿疼痛VAS评分均明显低于常规组,P<0.05;实验组与常规组的治疗优良率分别为89.13%与52.17%,P<0.05。' \
        '结论 对腰椎间盘突出症患者采取经皮内镜椎间孔入路微创治疗的效果显著,便于有效缓解疼痛感,降低疾病复发率,值得采纳。 '
    abs = Remove_chars(abs)
    abs = E_trans_to_C(abs)
    abs = abs.lower()
    whole_txt = []
    result = split_zh_en(abs)
    for j, tx in enumerate(result):
        if tx[0] == 1:
            whole_txt.append(result[j][1])
        else:
            whole_txt.extend([tt for tt in tx[1]])

    print(whole_txt)