#-*- coding: utf-8 -*-

import os
import sys
import logging

import numpy as np
import scipy.stats as ss
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from IV2 import *

logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s]  %(asctime)s %(message)s',
                    filename='train.log',
                    filemode='a')


class BIN_CHAR(BaseEstimator, TransformerMixin):
    """
        :param lst: 需要计算的自变量列表
        :type lst: list

        :param var_y: Y变量名
        :type var_y: str

        :param max_bin: 离散变量最多划分数
        :type max_bin: int

        :param output: 如果不为None, 则作为文件路径，输出分段文件
        :type output: str

        :param label: 是所有 lst 对应的中文字。作用一是在js 中显示中文而不是英文字段名。
        :type label: dict

        :attr lst: 直接从 lst 中过来，不会变
        :type lst: list

        :attr var_y: 直接从 var_y 过来, 不会变
        :type var_y: str

        :attr max_bin_: 直接从 max_bin 过来, 不会变
        :type max_bin_: int

        :attr output_: 输出目录
        :type output_: str

        :attr drop_list_: 根据训练结果, 记录需要删除的字段。
        :type drop_list_: list

        :attr drop_dict_: 记录 drop_list_ 对应的 分类数。 keys() 对应于 drop_list_
        :type drop_dict_: dict

        :attr keep_list_: 根据训练结果, 记录需要保留的字段
        :type keep_list_: list

        :attr transform_dict_: 记录每个有效字段的 NumVarBin
        :type transform_dict_: dict

        :attr transform_group_dict_: 记录每个有效字段的 分段 DF
        :type transform_group_dict_: dict

        :attr profile_dict_: profile dict 由 profile 计算得到
        :type profile_dict_: dict

        :attr iv_dict_: iv 字典, key 是 所有有效变量, value 是 iv值, 由 iv() 计算得到
        :type iv_dict_: dict 

    """
    def __init__(self, lst, var_y, max_bin=30, output=None, label=None):
        self.lst = lst
        self.var_y = var_y
        self.max_bin_ = max_bin
        self.output_ = output
        self.label_ = label
        logging.info("now init BIN_CHAR: X_lst: %s \r\n Y: %s" % 
                    (str(self.lst), str(self.var_y)))

    def fit(self, X, y):
        self.drop_list_ = []
        self.drop_dict_ = {}
        self.transform_dict_ = {}
        self.transform_group_dict_ = {}

        X.index = range(0, len(X))
        y.index = range(0, len(y))

        df_D = pd.concat([X, y], axis=1)
        for x in self.lst:
            print(x)
            if len(set(df_D[x])) > self.max_bin_:
                self.drop_list_.append(x)
                self.drop_dict_[x] = len(set(df_D[x]))
                continue

            df_X = df_D[x]
            df_Y = df_D[self.var_y]

            cvb = CharVarBin(x, self.var_y, max_bin=self.max_bin_)
            print(df_X)
            print(df_Y)
            cvb.fit(df_X, df_Y)
            # print cvb.
            self.transform_dict_[x] = cvb
            self.transform_group_dict_[x] = cvb.transform_group(df_X, df_Y)

            if self.output_:
                print(self.transform_group_dict_[x])
                self.transform_group_dict_[x].to_csv(self.output_ + x + ".csv", encoding='gbk')
        self.keep_list_ = self.transform_dict_.keys()
        return self

    def transform(self, X, y=None):
        test_df = pd.DataFrame()
        for x in self.keep_list_:
            cvb = self.transform_dict_[x]
            print(cvb)
            test_df[x] = cvb.transform(X[x], y)
        return test_df

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)


    def profile(self, X, y=None, output=None, label=True):
        """
            :param output: 如果不为None, 那么就是输出路径
            :type output: str

            :param label: 如果 label 是 True 并且 self.label_ 不为空的话, 
                        那么就会将label信息输出到csv
            :type label: Boolean

        """
        self.profile_dict_ = {}
        if y is None:
            y = [1] * len(X)

        X.index = range(0, len(X))
        y.index = range(0, len(y))

        df_D = pd.concat([X, y], axis=1)

        for x in self.keep_list_:
            cvb = self.transform_dict_[x]
            transform_group = cvb.transform_group(df_D[x], y)
            self.profile_dict_[x] = self._get_profile(transform_group)

        if output is not None:
            result = pd.DataFrame()
            for k, v in self.profile_dict_.items():
                result = pd.concat([result, v], axis=0)
            if label is True and self.label_ is not None:
                result["label"] = result["var"].apply(lambda x: self.label_[x])
                result.to_csv(output, index=False, encoding="gbk")
            else:
                result.to_csv(output, index=False, encoding="gbk")
        return self.profile_dict_

    def iv(self, X, y, output=None):
        """
            :param output: 如果不为None, 那么就是输出路径
            :type output: str

        """
        self.iv_dict_ = {}
        
        X.index = range(0, len(X))
        y.index = range(0, len(y))

        df_D = pd.concat([X, y], axis=1)

        for x in self.keep_list_:
            cvb = self.transform_dict_[x]
            transform_group = cvb.transform_group(df_D[x], y)
            _, self.iv_dict_[x] = cal_iv(transform_group)

        if output is not None:
            result = pd.DataFrame()
            for k, v in self.iv_dict_.items():
                result.ix[k, "iv"] = v
            result.to_csv(output, index=True)

        return self.iv_dict_

    def _profile_js(self, js_file="iv.js", html_file="iv.html", label=True):
        """
            :param js_file: 违约和样本量的代码输出文件
            :type js_file: str

            :param html_file: HTML 调用 js 代码
            :type html_file: str

            :param label: 是否将变量输出为标签
            :type label: Boolean
        """
        print (>> open("iv.js", "wb"), "")
        print (>> open("iv.html", "wb"), "")
        for x in self.keep_list_:
            feature_name = x
            gd = self.transform_group_dict_[x]
            print(gd)
            gd = gd[gd["count"] != 0]
            gd["mean"] = gd["sum"] * 100.0 / gd["count"]
            gd.sort("mean", inplace=True)
            bin_lst = list(gd["bin"])
            cnt_lst = [str(int(y)) for y in list(gd["count"])]
            pct_lst = [str(round(y, 2)) for y in list(gd["mean"])]
            if label is True and self.label_ is not None:
                print (>> open(js_file, "ab"), "$(function(){$('#feature_%(feature)s').highcharts({chart:{zoomType:'xy'},title:{text:'%(feature_label)s'},xAxis:[{categories:[%(bin)s],crosshair:true}],yAxis:[{labels:{format:'{value}%%',style:{color:Highcharts.getOptions().colors[1]}},title:{text:'违约率',style:{color:Highcharts.getOptions().colors[1]}}},{title:{text:'样本数',style:{color:Highcharts.getOptions().colors[0]}},labels:{format:'{value}',style:{color:Highcharts.getOptions().colors[0]}},opposite:true}],tooltip:{shared:true},legend:{layout:'vertical',align:'center',x:50,verticalAlign:'top',y:30,floating:true,backgroundColor:(Highcharts.theme&&Highcharts.theme.legendBackgroundColor)||'#FFFFFF'},credits:{enabled:false},series:[{name:'样本数',type:'column',yAxis:1,data:[%(cnt_lst)s],tooltip:{valueSuffix:'个'}},{name:'违约率',type:'spline',data:[%(dv_pct)s],tooltip:{valueSuffix:'%%'}}]});});" % {
                                "feature": feature_name, 
                                "feature_label": self.label_[feature_name].encode("utf-8"), 
                                "bin": "'" + "', '".join(bin_lst) + "'",
                                "cnt_lst": ", ".join(cnt_lst),
                                "dv_pct": ", ".join(pct_lst)
                                })
            else:
                print(feature_name)
                print("'" + "', '".join(bin_lst) + "'")
                print(", ".join(cnt_lst))
                print(", ".join(pct_lst))
                print (>> open(js_file, "ab"), "$(function(){$('#feature_%(feature)s').highcharts({chart:{zoomType:'xy'},title:{text:'%(feature)s'},xAxis:[{categories:[%(bin)s],crosshair:true}],yAxis:[{labels:{format:'{value}%%',style:{color:Highcharts.getOptions().colors[1]}},title:{text:'违约率',style:{color:Highcharts.getOptions().colors[1]}}},{title:{text:'样本数',style:{color:Highcharts.getOptions().colors[0]}},labels:{format:'{value}',style:{color:Highcharts.getOptions().colors[0]}},opposite:true}],tooltip:{shared:true},legend:{layout:'vertical',align:'center',x:50,verticalAlign:'top',y:30,floating:true,backgroundColor:(Highcharts.theme&&Highcharts.theme.legendBackgroundColor)||'#FFFFFF'},credits:{enabled:false},series:[{name:'样本数',type:'column',yAxis:1,data:[%(cnt_lst)s],tooltip:{valueSuffix:'个'}},{name:'违约率',type:'spline',data:[%(dv_pct)s],tooltip:{valueSuffix:'%%'}}]});});" % {
                                "feature": feature_name.encode("utf-8"), 
                                "bin": ("'" + "', '".join(bin_lst) + "'").encode("utf-8"),
                                "cnt_lst": ", ".join(cnt_lst),
                                "dv_pct": ", ".join(pct_lst)
                                })
            print (>> open(html_file, "ab"), '<div class="col-xs-12 col-sm-6 placeholder"><div id="feature_%s" style="min-width:300px;height:300px"></div></div>' % feature_name)


    def psi(self, X, y=None, output=None, psi_group_print=True):
        """
            需要调完

            :param X: 需要至少传入 self.keep_list_ 这么多字段
            :type X: pd.DataFrame

            :param y: 默认None, 传入也没用
            :type y: 

            :param output: 如果不为None, 那么就是输出路径
            :type output: str

        """
        self.psi_dict_ = {}
        self.psi_group_dict_ = {}
        for x in self.keep_list_:
            cvb = self.transform_dict_[x]
            transform_group = cvb.transform_group(X[x], None)
            train_transform_group = self.transform_group_dict_[x]

            psi, psi_group = cal_psi(train_transform_group, transform_group)
            self.psi_dict_[x] = psi
            self.psi_group_dict_[x] = psi_group
        if output is not None:
            result = pd.DataFrame()
            for k, v in self.psi_dict_.items():
                result.ix[k, "psi"] = v
            result.to_csv(output, index=True)
        return self.psi_dict_


    def _get_profile(self, X):
        """
            被 profile 调用
        """
        ccnt = X["count"].sum()
        ssum = X["sum"].sum()
        vvar = X.ix[0, "var"]
        print(vvar)
        mmea = ssum * 1.0 / ccnt
        X.loc[-200] = {"bin": "total", 
                        "count": ccnt, 
                        "sum": ssum,
                        "var": vvar
                        }
        print(X)
        X.fillna(0, inplace=True)
        print(X)
        X["percent"] = X["count"] * 1.0 / ccnt
        X["average_dv"] = X["sum"] / X["count"]
        X["average_dv"].fillna(mmea, inplace=True)
        X["index"] = (X["average_dv"] * 100.0 / mmea ).astype('int')
        return X



if __name__ == "__main__":
    INPUT_PATH = u'E:/维信工作/待完成/手机迭代/model/0917/data/'
    OUTPUT_PATH = u'KKD_0923/'
    # OUTPUT_PATH = u'DDQ_0923/'

    df_D = pd.read_csv(INPUT_PATH + 'train_DOUDOUQIAN.tsv', sep=',', encoding='utf-8')
    df_T = pd.read_csv(INPUT_PATH + 'test_DOUDOUQIAN.tsv', sep=',', encoding='utf-8')
    # df_D = pd.read_csv(INPUT_PATH + 'train_KAKADAI.tsv', sep=',', encoding='utf-8')
    # df_T = pd.read_csv(INPUT_PATH + 'test_KAKADAI.tsv', sep=',', encoding='utf-8')
    df_D.dtypes.to_csv("dtypes.csv")

    # DDQ
    x_lst = ["id_no", "apply_date", "loan_date", "fst_bill_date", "pyof_date", "agent", "id_addr", "id_vld_time", "gender", "star", "zodiac", "loankind", "is_advance", "is_local", "education", "marriage", "bas_dt", "bas_id_card", "bas_name", "bas_reg_time", "bas_update_time", "bas_createtime", "bas_process_dt", "bas_sendtime", "bas_receivefilepath", "bas_bustype", "bas_busid", "c_dt", "c2_dt", "ct_dt", "thrm_dt", "n_dt", "s_dt", "b_dt",]
    # KKD
    x_lst = ["id_no", "apply_date", "loan_date", "fst_bill_date", "pyof_date", "id_vld_time", "gender", "star", "zodiac", "loankind", "is_advance", "education", "marriage", "bas_dt", "bas_id_card", "bas_name", "bas_reg_time", "bas_update_time", "bas_createtime", "bas_process_dt", "bas_sendtime", "bas_receivefilepath", "bas_bustype", "bas_busid", "c_dt", "c2_dt", "ct_dt", "thrm_dt", "n_dt", "s_dt", "b_dt", ]

    bc = BIN_CHAR(x_lst, "forth_m2", 10, output=OUTPUT_PATH,)

    # 1. 先进行 fit 学习操作, 
    # 学习过程中就可以得到属性 drop_list_ - 不需要的变量列表, 
    # keep_list_ 需要的变量列表,
    # transform_dict_ 就是 keep_list_ 对应的离散化转换的实例
    # transform_group_dict_ 记录的是 keep_list_ 中训练数据的转换的 dataframe, 以供将来计算一些值使用
    bc.fit(df_D[x_lst], df_D["forth_m2"])
    print (bc.drop_list_)
    print (bc.drop_dict_)
    print (bc.keep_list_)
    print (bc.transform_dict_)
    print (bc.transform_group_dict_)

    # 2. 对于新的数据就可以进行转换
    print (bc.transform(df_D[x_lst]))

    # 3. 可以对 新的数据 进行 profile, 
    # profile 输出就是 一个 excel - profile.csv, 以及一个合并的 dataframe
    # 以及属性 profile_dict_ - 画像字典
    bc.profile(df_D[x_lst], df_D["forth_m2"], OUTPUT_PATH + "profile.csv", label=True)
    print (bc.profile_dict_)

    # 4. 可以 计算新数据的IV
    # 调用 IV 功能，就可以得到 
    # iv_dict_ - keep_list_ 的 iv 字典
    print (bc.iv(df_D[x_lst], df_D["forth_m2"], OUTPUT_PATH + "iv.csv"))
    print (bc.iv_dict_)

    # 5. 可以输出 JS 文件
    # 默认生成 iv.js 和 iv.html 文件
    # 可以快速作为 可视化展示
    bc._profile_js()

    # 5. 可以计算 训练集 和 新数据之间的 PSI
    # 调用 PSI 功能后，就可以得到
    # psi_dict_ - keep_list_ 的 psi 字典
    # psi_group_dict_ - keep_list_ 的 psi 计算的 df 的字典
    print (bc.psi(df_T[x_lst], output=OUTPUT_PATH + "psi.csv"))
    print (bc.psi_dict_)
    print (bc.psi_group_dict_)
    for k, v in bc.psi_group_dict_.iteritems():
        v.to_csv(OUTPUT_PATH+"psi_%s.csv" % k, index=False)