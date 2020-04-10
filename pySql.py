#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: LiuHuan
# Datetime: 2019/11/15 16:25


import pymssql
import pandas as pd


class pySql:
    def __init__(self, ip, user, pwd, db):
        self.ip = ip
        self.user = user
        self.pwd = pwd
        self.db = db
        self.conn = pymssql.connect(server=ip, user=user, password=pwd, database=db)

    def read_sql(self, sql):
        df = pd.read_sql(sql, self.conn)
        return df

    def write_sql(self, sql):
        cursor = self.conn.cursor()
        cursor.execute(sql)
        self.conn.commit()

    def close(self):
        self.conn.close()

