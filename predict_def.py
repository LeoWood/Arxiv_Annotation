#!/usr/bin/env python
#-*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2020/4/9 12:32

from bert_base.client import BertClient
import socket

def predict_def(sentences):
    sk = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sk.settimeout(1)
    flag = 0
    try:
        sk.connect(('159.226.125.191', 5610))
        flag = 1
    except:
        pass
    sk.close()

    if not flag:
        print('def server not available')
        return 0
    else:

        with BertClient(port=5610,port_out=5611,show_server_config=False, check_version=False, check_length=False, mode='CLASS') as bc:
            result = bc.encode(sentences)

        k = 0
        labels = []
        for re in result:
            for a in re['score']:
                if a[1] > 0.9:
                    labels.append(1)
                else:
                    labels.append(0)
                k += 1

        return labels

if __name__ == '__main__':
    pass