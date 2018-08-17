# -*- coding: utf-8 -*-

"""
注: 原始数据的 列名不允许是 中文
"""

import os
import sys
import glob

import numpy as np
import scipy.stats as ss
import pandas as pd
import logging


from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from BinChar import *
from BinNum import *

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s]  %(asctime)s %(message)s',
                    filename='train.log',
                    filemode='a')


INPUT_PATH = u'C:/Users/liuzhihui/Desktop/电话邦_v1_20180323/返回结果/'
OUTPUT_PATH = u'C:/Users/liuzhihui/Desktop/电话邦_v1_20180323/返回结果/'
OUTPUT_PATH_DETAIL = OUTPUT_PATH + "detail/"

Y_NAME = "FST_BILL_A20_OVRD"
PROFILE_NUM_NAME = "profile_num.csv"
IV_TRAIN_NUM_NAME = "iv_train_num.csv"
IV_TEST_NUM_NAME = "iv_test_num.csv"
PSI_NUM_NAME = "psi_num.csv"
PROFILE_CHAR_NAME = "profile_char.csv"
IV_TRAIN_CHAR_NAME = "iv_train_char.csv"
IV_TEST_CHAR_NAME = "iv_test_char.csv"
PSI_CHAR_NAME = "psi_char.csv"
NUM_BIN = 10
CHAR_BIN = 30
drop_lst = ["fst_bill_a20_ovrd", "fst_3_zero_pay",# "fst_bill_a7_ovrd",
            "everm2", "h_max_od_day", "fst_bill_a3_ovrd", "fst_bill_ovrd",
            "pi_id", "is_id", "cp_id", "is_person_infromation_id",
            "cp_person_infromation_id", "bid", "cid", "mob", "is_advance", "loannum",
            "loan_amt", "samekind_loan_num", "cid", "mob", "is_blk_post", "prd",
            "is_blk_pre", "samekind_apply_num"]


df_D = pd.read_csv(INPUT_PATH + 'callbang-180402.csv', sep='\t', encoding='utf-8')
df_T = pd.read_csv(INPUT_PATH + 'callbang-180402.csv', sep='\t', encoding='utf-8', nrows=10)


df_D = df_D[np.isnan(df_D[Y_NAME]) == False]
df_T = df_T[np.isnan(df_T[Y_NAME]) == False]


try:
    os.mkdir(OUTPUT_PATH_DETAIL)
except:
    pass
try:
    os.mkdir(OUTPUT_PATH)
except:
    pass


var_lst = list(df_D.columns)
for x in drop_lst:
    if x in var_lst:
        var_lst.remove(x)
df_D = df_D[var_lst]
df_T = df_T[var_lst]

# TYPE
dtypes = df_D.dtypes


# char_lst = list(dtypes[dtypes == "object"].index)
char_lst = list(
    dtypes[dtypes.isin([np.dtype("O"), np.dtype("<M8[ns]")])].index)

# num_lst = list(dtypes[dtypes != "object"].index)
num_lst = list(
    dtypes[~dtypes.isin([np.dtype("O"), np.dtype("<M8[ns]")])].index)


if Y_NAME in num_lst:
    num_lst.remove(Y_NAME)


for x in char_lst:
    df_T[x] = df_T[x].astype('unicode')
    df_D[x] = df_D[x].astype('unicode')


# NUM PLACE
bn = BIN_NUM(num_lst, Y_NAME, NUM_BIN, output=OUTPUT_PATH_DETAIL,)

bn.fit(df_D[num_lst], df_D[Y_NAME])
logging.info("NUM drop list: %s" % bn.drop_list_)
logging.info("NUM keep list: %s" % bn.keep_list_)


bn.profile(df_D[num_lst], df_D[Y_NAME],
           OUTPUT_PATH + PROFILE_NUM_NAME, label=True)

print(bn.iv(df_D[num_lst], df_D[Y_NAME], OUTPUT_PATH + IV_TRAIN_NUM_NAME))
print(bn.iv(df_T[num_lst], df_T[Y_NAME], OUTPUT_PATH + IV_TEST_NUM_NAME))

# bn._profile_js()

print(bn.psi(df_T[num_lst], output=OUTPUT_PATH + PSI_NUM_NAME))
logging.info("NUM variable psi: %s" % bn.psi_dict_)
for k, v in bn.psi_group_dict_.items():
    v.to_csv(OUTPUT_PATH_DETAIL + "psi_%s.csv" % k, index=False)


# CHAR PLACE
bc = BIN_CHAR(char_lst, Y_NAME, CHAR_BIN, output=OUTPUT_PATH_DETAIL,)
bc.fit(df_D[char_lst], df_D[Y_NAME])
logging.info("CHAR drop list: %s" % bc.drop_list_)
logging.info("CHAR keep list: %s" % bc.keep_list_)


bc.profile(df_D[char_lst], df_D[Y_NAME],
           OUTPUT_PATH + PROFILE_CHAR_NAME, label=True)

print(bc.iv(df_D[char_lst], df_D[Y_NAME], OUTPUT_PATH + IV_TRAIN_CHAR_NAME))
print(bc.iv(df_T[char_lst], df_T[Y_NAME], OUTPUT_PATH + IV_TEST_CHAR_NAME))


# bc._profile_js()

print(bc.psi(df_T[char_lst], output=OUTPUT_PATH + PSI_CHAR_NAME))
logging.info("CHAR variable psi: %s" % bc.psi_dict_)
for k, v in bc.psi_group_dict_.iteritems():
    v.to_csv(OUTPUT_PATH_DETAIL + "psi_%s.csv" % k, index=False, encoding="gbk")


# 汇总:
iv_train_num = pd.read_csv(OUTPUT_PATH + IV_TRAIN_NUM_NAME)
iv_test_num = pd.read_csv(OUTPUT_PATH + IV_TEST_NUM_NAME)
iv_train_char = pd.read_csv(OUTPUT_PATH + IV_TRAIN_CHAR_NAME)
iv_test_char = pd.read_csv(OUTPUT_PATH + IV_TEST_CHAR_NAME)


iv_train_num["time"] = "train"
iv_train_num["type"] = "num"
iv_test_num["time"] = "test"
iv_test_num["type"] = "num"
iv_train_char["time"] = "train"
iv_train_char["type"] = "char"
iv_test_char["time"] = "test"
iv_test_char["type"] = "char"
df = pd.concat([iv_train_num, iv_train_char,
                iv_test_num, iv_test_char], axis=0)
df.to_csv(OUTPUT_PATH + "iv_all.csv")
