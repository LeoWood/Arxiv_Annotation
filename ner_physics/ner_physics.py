#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/1/10 10:49

import torch
from nltk.tokenize import word_tokenize
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import BertForTokenClassification, BertTokenizer
from tqdm import tqdm
import numpy as np
from .utils_ner import convert_examples_to_features, get_first_category, read_examples_from_list,get_second_category
# from tst import text_to_word_list, split_zh_en
import pymssql
import logging
logging.basicConfig(level=logging.INFO)
conn = pymssql.connect('159.226.125.180', 'monitor', 'Monitor@6320', 'arxiv_physics_article')
cursor = conn.cursor()

sql1="select ch_second_category,ch_top_category from  first_second_catagory where second_category='%s'"
ch_top_category=['模型、方法理论','计量、仪器和数据分析','现象和现象规律','数学']
top_category=['Formalism (models, methods)','Metrology, Instrumentation, Data analysis','Phenomena, phenomenological laws','Mathematics']
### Some Parameters
#一级范畴的模型位置
output_dir1 = r'E:\LiuHuan\Projects\Sci_Engine\resource\science_wise_models\ouput_9w_first_category'
#二级范畴的模型位置
output_dir2 = r'E:\LiuHuan\Projects\Sci_Engine\resource\science_wise_models\ouput_9w_second_category'
MAX_SEQ_LENGTH = 512
MODEL_TYPE = 'bert'
EVAL_BATCH_SIZE = 8
DEVICE = torch.device("cuda")

### Pre-Load model
tokenizer = BertTokenizer.from_pretrained(output_dir1)
model1 = BertForTokenClassification.from_pretrained(output_dir1)
model1.to('cuda')
model2= BertForTokenClassification.from_pretrained(output_dir2)
model2.to('cuda')
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = CrossEntropyLoss().ignore_index

second_category_label = get_second_category(path=None)
first_category_label=get_first_category(path=None)



word_list = []

def load_and_cache_examples_first(text):
    global word_list
    word_list = text.split()
    print(word_list)
    print(len(word_list))
    examples = read_examples_from_list(word_list)
    features = convert_examples_to_features(examples, first_category_label, MAX_SEQ_LENGTH, tokenizer,
                                            cls_token_at_end=bool(MODEL_TYPE in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if MODEL_TYPE in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(MODEL_TYPE in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(MODEL_TYPE in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if MODEL_TYPE in ["xlnet"] else 0,
                                            pad_token_label_id=pad_token_label_id
                                            )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset

def load_and_cache_examples_second(text):
    global word_list
    word_list = text.split()
    print(word_list)
    print(len(word_list))
    examples = read_examples_from_list(word_list)
    features = convert_examples_to_features(examples, second_category_label, MAX_SEQ_LENGTH, tokenizer,
                                            cls_token_at_end=bool(MODEL_TYPE in ["xlnet"]),
                                            # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            cls_token_segment_id=2 if MODEL_TYPE in ["xlnet"] else 0,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(MODEL_TYPE in ["roberta"]),
                                            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                            pad_on_left=bool(MODEL_TYPE in ["xlnet"]),
                                            # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if MODEL_TYPE in ["xlnet"] else 0,
                                            pad_token_label_id=pad_token_label_id
                                            )

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def predict_first_catagory(text,need_to_find_first):
    eval_dataset = load_and_cache_examples_first(text)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model1.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if MODEL_TYPE != "distilbert":
                inputs["token_type_ids"] = batch[2] if MODEL_TYPE in ["bert",
                                                                           "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model1(**inputs)
            tmp_eval_loss, logits = outputs[:2]


            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(first_category_label)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    # print(out_label_list)
    # print(len(out_label_list[0]))
    print(preds_list)
    print(len(preds_list[0]))


    # word_list = ['目', '的', ' ', '探', '讨', '经', '皮', '内', '镜', '椎', '间', '孔', '入', '路', '微', '创', '治', '疗']
    # label_list = ['O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'B', 'I', 'I', 'I', 'I', 'B', 'I', 'O', 'O']

    label_list = preds_list[0]
    str_ = ' '
    all_keys = []

    for j, token in enumerate(label_list):
        if token.split('-')[0]=='B':
            keys = [word_list[j]]
            for z in range(j + 1, len(label_list)):  ############如果最后一个是B，range()两个相同的值，函数自动不执行
                if label_list[z].split('-')[0]=='I':
                    keys.append(word_list[z])
                else:
                    break
            all_keys.append([str_.join(keys), token.split('-')[-1]])
    print(all_keys)
    #####去除重复的
    del_elements_index = []
    for j, kk in enumerate(all_keys):
        for z in range(j + 1, len(all_keys)):
            if kk == all_keys[z]:
                del_elements_index.append(z)  ####如果重复，就把后面的元素添加进来
        # 倒序删除
    del_elements_index = set(del_elements_index)  ############一定要去除重复的，否则的话会删去很多无关的元素
    for i in range(len(all_keys) - 1, -1, -1):
        if i in del_elements_index:
            all_keys.pop(i)

    first_category=[]
    for ele in need_to_find_first:
        for kk,key in enumerate(all_keys):
            if ele == key[0]:
                first_category.append([ele, key[1]])
    return first_category






def predict_second_category(text):
    eval_dataset = load_and_cache_examples_second(text)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=EVAL_BATCH_SIZE)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model2.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "labels": batch[3]}
            if MODEL_TYPE != "distilbert":
                inputs["token_type_ids"] = batch[2] if MODEL_TYPE in ["bert",
                                                                           "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
            outputs = model2(**inputs)
            tmp_eval_loss, logits = outputs[:2]


            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(second_category_label)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    # print(out_label_list)
    # print(len(out_label_list[0]))
    print(preds_list)
    print(len(preds_list[0]))


    # word_list = ['目', '的', ' ', '探', '讨', '经', '皮', '内', '镜', '椎', '间', '孔', '入', '路', '微', '创', '治', '疗']
    # label_list = ['O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'B', 'I', 'I', 'I', 'I', 'B', 'I', 'O', 'O']
    global word_list
    word_list = word_list[:len(preds_list[0])]
    label_list = preds_list[0]
    str_ = ' '
    all_keys = []

    for j, token in enumerate(label_list):
        if token.split('-')[0]=='B':
            keys = [word_list[j]]
            for z in range(j + 1, len(label_list)):  ############如果最后一个是B，range()两个相同的值，函数自动不执行
                if label_list[z].split('-')[0]=='I':
                    keys.append(word_list[z])
                else:
                    break
            all_keys.append([str_.join(keys), token.split('-')[-1]])
    print(all_keys)
    #####去除重复的
    del_elements_index = []
    for j, kk in enumerate(all_keys):
        for z in range(j + 1, len(all_keys)):
            if kk == all_keys[z]:
                del_elements_index.append(z)  ####如果重复，就把后面的元素添加进来
        # 倒序删除
    del_elements_index = set(del_elements_index)  ############一定要去除重复的，否则的话会删去很多无关的元素
    for i in range(len(all_keys) - 1, -1, -1):
        if i in del_elements_index:
            all_keys.pop(i)
#####################################################
    ##查找二级范畴的上一级范畴，分两种情况，如果二级范畴为None,则要调用一级范畴模型去识别范畴
    need_to_find_first=[]
    for zk,key in enumerate(all_keys):
        if key[1]=='None':
            need_to_find_first.append(key[0])

    if len(need_to_find_first)!=0:
        first_category=predict_first_catagory(text,need_to_find_first)
        term_have_first=[i[0] for i in first_category]

    for zk, key in enumerate(all_keys):
        if key[1]=='None':
            ##对于找到了一级范畴的术语
            if key[0] in term_have_first:
                first = first_category[term_have_first.index(key[0])][1]
                all_keys[zk].extend(['', ch_top_category[top_category.index(first)]])
            else:
            ###对于没有找到一级范畴的术语
                all_keys[zk].extend(['', ''])
        else:
            cursor.execute(sql1%all_keys[zk][1])
            result=cursor.fetchone()
            all_keys[zk].extend(result)###注意此处也可能为空，因为有新术语

    all_keys=[[i[0],i[2],i[3]] for i in all_keys]

    print(all_keys)
    anno = {}
    for i in all_keys:
        anno[i[0]] = i[2] + ',' + i[1]


    return anno##########all_keys应该是[term,ch_second_categoyr,



def anotation_ner_physics(txt):
    print(txt)
    ano_dict = predict_second_category(txt)
    print(ano_dict)

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
                    # new_word_list.append(
                    #     '<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">' + token_tmp + '<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">[' +
                    #     ano_dict[token_tmp] + ']</mark>')
                    new_word_list.append('<strong><a href="#" class="tooltip-test" data-toggle="tooltip" title="' + ano_dict[
                        token_tmp] + '">' + token_tmp + '</a></strong>')
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
                    # new_word_list.append(
                    #     '<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em; box-decoration-break: clone; -webkit-box-decoration-break: clone">' + token_tmp + '<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">[' +
                    #     ano_dict[token_tmp] + ']</mark>')
                    new_word_list.append('<strong><a href="#" class="tooltip-test" data-toggle="tooltip" title="' + ano_dict[
                        token_tmp] + '">' + token_tmp + '</a></strong>')
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
    return ' '.join(new_word_list)


if __name__ == '__main__':
    while True:
        text = input()
        # text='We report new high-quality galaxy scale strong lens candidates found in the Kilo Degree Survey data release 4 using Machine Learning. We have developed a new Convolutional Neural Network (CNN) classifier to search for gravitational arcs, following the prescription by Petrillo et al. 2019 and using only r−band images. We have applied the CNN to two "predictive samples": a Luminous red galaxy (LRG) and a "bright galaxy" (BG) sample (r<21).'
        anotation_ner_physics(text)
