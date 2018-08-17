# -*- coding: utf-8 -*-

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


def generate_case_when_sql(variable, bin, right=True, as_var=True):
    """
        :param variable:
        :type variable:

        :param bin:
        :type bin:
        
    """
    result_sql_lst = []
    if len(bin) <= 1:
        raise Exception
    for ii, x_last in enumerate(bin):
        if ii == len(bin) - 1:
            sql = " else 'NULL or abnormal' end"
            result_sql_lst.append(sql)
        else:
            x = bin[ii + 1]
            if right:
                sql = """ when %(var)s > %(x_last)s and %(var)s <= %(x)s then '(%(x_last)s, %(x)s]'""" % {
                    "var": variable, "x": x, "x_last": x_last}
            else:
                sql = """ when %(var)s >= %(x_last)s and %(var)s < %(x)s then '[%(x_last)s, %(x)s)'""" % {
                    "var": variable, "x": x, "x_last": x_last}
            result_sql_lst.append(sql)

    if as_var:
        return "\r\n".join(result_sql_lst) + " as %s," % variable
    else:
        return "\r\n".join(result_sql_lst)


class BIN_NUM(BaseEstimator, TransformerMixin):
    """
        :param lst: 需要计算的自变量列表
        :type lst: list

        :param var_y: Y变量名
        :type var_y: str

        :param cut_bin: 数值变量分段数
        :type cut_bin: int

        :param output: 如果不为None, 则作为文件路径，输出分段文件
        :type output: str

        :param label: 是所有 lst 对应的中文字。作用一是在js 中显示中文而不是英文字段名。
        :type label: dict

        :attr lst:
        :type lst:

        :attr var_y:
        :type var_y:

        :attr cut_bin_:
        :type cut_bin_:

        :attr output_:
        :type output_:

        :attr drop_list_: 根据训练结果, 记录需要删除的字段。删除规则 todo
        :type drop_list_: list

        :attr keep_list_: 根据训练结果, 记录需要保留的字段
        :type keep_list_: list

        :attr transform_dict_: 记录每个有效字段的 NumVarBin
        :type transform_dict_: dict

        :attr transform_group_dict_: 记录每个有效字段的 分段 DF
        :type transform_group_dict_: dict

        :attr iv_dict_: IV 的字典
        :type iv_dict_: dict

        :attr psi_dict_: PSI 的字典
        :type psi_dict_: dict

        :attr psi_group_dict_: 每个保留变量PSI计算的df的字典
        :type psi_group_dict_: dict

        :attr sql_lst_: 变量case when 的代码
        :type sql_lst_: list

    """

    def __init__(self, lst, var_y, cut_bin=10, output=None, label=None):
        self.lst = lst
        self.var_y = var_y
        self.cut_bin_ = cut_bin
        self.output_ = output
        self.label_ = label

    def fit(self, X, y):
        self.drop_list_ = []
        self.transform_dict_ = {}
        self.transform_group_dict_ = {}

        X.index = range(0, len(X))
        y.index = range(0, len(y))

        df_D = pd.concat([X, y], axis=1)
        for x in self.lst:
            print (x)
            if len(df_D[np.isnan(df_D[x]) == False]) == 0:
                self.drop_list_.append(x)
                continue

            df_X = df_D[x]
            df_Y = df_D[self.var_y]

            nvb = NumVarBin(x, self.var_y, cut_bin=self.cut_bin_)
            print (df_X)
            print (df_Y)
            nvb.fit(df_X, df_Y)
            self.transform_dict_[x] = nvb
            self.transform_group_dict_[x] = nvb.transform_group(df_X, df_Y)

            if self.output_:
                print (self.transform_group_dict_[x])
                self.transform_group_dict_[x].to_csv(self.output_ + x + ".csv")
        self.keep_list_ = self.transform_dict_.keys()
        return self

    def transform(self, X, y=None):
        test_df = pd.DataFrame()
        for x in self.keep_list_:
            nvb = self.transform_dict_[x]
            test_df[x] = nvb.transform(X[x], y)
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
            nvb = self.transform_dict_[x]
            transform_group = nvb.transform_group(df_D[x], y)
            self.profile_dict_[x] = self._get_profile(transform_group)

        if output is not None:
            result = pd.DataFrame()
            for k, v in self.profile_dict_.items():
                result = pd.concat([result, v], axis=0)
            if label is True and self.label_ is not None:
                result["label"] = result["var"].apply(lambda x: self.label_[x])
                result.to_csv(output, index=False, encoding="gbk")
            else:
                result.to_csv(output, index=False)
        return self.profile_dict_

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
            print (gd)
            gd = gd[gd["count"] != 0]
            gd["mean"] = gd["sum"] * 100.0 / gd["count"]
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
                print (>> open(js_file, "ab"), "$(function(){$('#feature_%(feature)s').highcharts({chart:{zoomType:'xy'},title:{text:'%(feature)s'},xAxis:[{categories:[%(bin)s],crosshair:true}],yAxis:[{labels:{format:'{value}%%',style:{color:Highcharts.getOptions().colors[1]}},title:{text:'违约率',style:{color:Highcharts.getOptions().colors[1]}}},{title:{text:'样本数',style:{color:Highcharts.getOptions().colors[0]}},labels:{format:'{value}',style:{color:Highcharts.getOptions().colors[0]}},opposite:true}],tooltip:{shared:true},legend:{layout:'vertical',align:'center',x:50,verticalAlign:'top',y:30,floating:true,backgroundColor:(Highcharts.theme&&Highcharts.theme.legendBackgroundColor)||'#FFFFFF'},credits:{enabled:false},series:[{name:'样本数',type:'column',yAxis:1,data:[%(cnt_lst)s],tooltip:{valueSuffix:'个'}},{name:'违约率',type:'spline',data:[%(dv_pct)s],tooltip:{valueSuffix:'%%'}}]});});" % {
                    "feature": feature_name,
                    "bin": "'" + "', '".join(bin_lst) + "'",
                    "cnt_lst": ", ".join(cnt_lst),
                    "dv_pct": ", ".join(pct_lst)
                })
            print(>> open(
                html_file, "ab"), '<div class="col-xs-12 col-sm-6 placeholder"><div id="feature_%s" style="min-width:300px;height:300px"></div></div>' % feature_name)

    def _generate_sql(self):
        self.sql_lst_ = []
        for x in self.keep_list_:
            gd = self.transform_dict_[x]
            lst = gd.cut_lst_
            sql_code = generate_case_when_sql(x, lst, as_var=True)
            self.sql_lst_.append(sql_code)
        return "\r\n".join(self.sql_lst_)

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
            nvb = self.transform_dict_[x]
            transform_group = nvb.transform_group(df_D[x], y)
            _, self.iv_dict_[x] = cal_iv(transform_group)

        if output is not None:
            result = pd.DataFrame()
            for k, v in self.iv_dict_.items():
                result.ix[k, "iv"] = v
            result.to_csv(output, index=True)

        return self.iv_dict_

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
            nvb = self.transform_dict_[x]
            transform_group = nvb.transform_group(X[x], None)
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
        vvar = X.ix[-100, "var"]
        print (vvar)
        mmea = ssum * 1.0 / ccnt
        X.loc[-200] = {"bin": "total",
                       "count": ccnt,
                       "sum": ssum,
                       "var": vvar
                       }
        print (X)
        X.fillna(0, inplace=True)
        print (X)
        X["percent"] = X["count"] * 1.0 / ccnt
        X["average_dv"] = X["sum"] / X["count"]
        X["average_dv"].fillna(mmea, inplace=True)
        X["index"] = (X["average_dv"] * 100.0 / mmea).astype(int)
        return(X)


##############################################################################
def app0923():
    INPUT_PATH = u'DDH_1019/'
    OUTPUT_PATH = u'DDH_1019/'

    # df_D = pd.read_csv(INPUT_PATH + 'train_DOUDOUQIAN.tsv', sep=',', encoding='utf-8')
    # df_T = pd.read_csv(INPUT_PATH + 'test_DOUDOUQIAN.tsv', sep=',', encoding='utf-8')
    df_D = pd.read_csv(INPUT_PATH + 'train.tsv', sep=',', encoding='utf-8')
    df_T = pd.read_csv(INPUT_PATH + 'test.tsv', sep=',', encoding='utf-8')
    # df_D = pd.read_csv(INPUT_PATH + 'train_KAKADAI.tsv', sep=',', encoding='utf-8')
    # df_T = pd.read_csv(INPUT_PATH + 'test_KAKADAI.tsv', sep=',', encoding='utf-8')
    # df_D.dtypes.to_csv("dtypes.csv")

    df_D["forth_m2"] = df_D["fst_bill_a5_ovrd"]
    df_T["forth_m2"] = df_T["fst_bill_a5_ovrd"]
    x_lst = ["bid", "cid", "fst_bill_ovrd", ]
    # # 卡卡贷变量删选 V1
    # x_lst = ["c_call_len_nit", "c_phone_nbr_nit", "c_call_cnt_nit", "s_avg_night_sms_num_rate", "s_avg_night_sms_num", "s_avg_night_sms_cnt", "c_call_days_rate", "c_uncall_ge3_days", "ct_onem_ln_fin_call_acnt", "ct_thrm_loane_call_cnt", "ct_thrm_loan_call_pcnt", "c2_same_cnt", "c2_bj_days_rt_3m", "ct_onem_loane_call_acnt", "ct_onem_ln_fin_call_cnt", "c_uncall_lt3_days", "c_phone_nbr_zj_rate", "c2_callst_len_rat", "c2_callst_cnt_rat_1m", "c2_zjst_day_rat_1m", "ct_thrm_carpt_call_cnt", "n_nigth_days_rate", "c2_bj_days_rt_1m", "ct_thrm_carpt_call_wcnt", "c_phone_nbr_bj", "ct_thrm_stor_call_times", "c_call_place_hd", "ct_thrm_stor_call_cnt", "c2_zjst_nur_rat", "c_phone_nbr_am", "ct_onem_bank_call_cnt", "c_call_len_my_rate", "s_phone_sms_num_1mth", "c_bj_phone_nbr_loc", "c2_zj_days_rt_3m", "n_avg_mth_usetime", "ct_thrm_recrut_call_cnt", "c_phone_nbr_avg", "n_flow_amt_1mth", "s_avg_phone_sms_num", "b_max_pay_amt", "b_max_plan_amt", "ct_thrm_crdt_call_pcnt", "c_call_cnt_bd_rate", "c_phone_nbr_zjrx_avg", "s_ph_sms_cnt", "n_avg_net_cnt", "b_pay_month", "c_call_len_bj_rate", "c2_len_bj_rt_1m", "n_avg_othr_tm_rate", "c_call_len_avg_zj", "thrm_ln_fin_biz_sms_cnt", "c2_calllen_mor_rt_3m", "c_tel_len", "thrm_estat_sms_cnt", "ct_onem_biz_call_acnt", "c2_zj_range_len", "ct_onem_biz_call_cnt", "b_min_plan_amt", "c_call_cnt_noon", "c_wm_place", "thrm_loane_sms_cnt", "c_bd_phn_crg", "s_sv_sms_cnt_rate", "c_tel_nbr", "ct_thrm_biz_call_avgtime", "s_sub_total_1mth", "c2_bj_range_cnt",]

    # # 豆豆钱变量删选 V3
    # x_lst = ["c_phone_nbr_nit", "c_call_cnt_nit", "c_call_len_nit", "s_avg_night_sms_num", "c_call_days_rate",  "c_uncall_ge3_days", "c_uncall_lt3_days", "ct_onem_carpt_call_acnt", "ct_thrm_carpt_call_avgtime", "c2_zjst_day_rat_1m", "c_call_cnt_am", "n_nigth_days_rate", "c_call_cnt_zjrx_avg", "n_flow_amt_1mth", "c2_st_day_rat_1m", "c2_bj_days_rt_3m", "c_phone_nbr_bj", "ct_thrm_loan_call_pcnt", "n_avg_mth_flow_amt", "c_phone_nbr_zj_rate", "c2_same_cnt", "ct_thrm_loane_call_wcnt", "ct_thrm_stor_call_cnt", "n_avg_mth_flo", "b_min_plan_amt", "c_call_place_hd", "c_bj_phone_nbr_loc", "c_zj_phone_nbr_loc", "ct_onem_loan_call_cnt", "n_aavg_wktm_cnt_rate", "c2_zj_days_rt_1m", "s_phone_sms_num_1mth", "c_ff_place", "s_avg_work_sms_num_sd", "ct_thrm_family_call_cnt", "c_call_len_ct", "c2_len_zjsj_avg_1m",  "c2_cnt_zjsj_avg_3m", "c_call_len_mor", "c_call_cnt_noon", "c_bj_phn_crg_avg", "c2_bj_range_cnt", "thrm_insurc_sms_cnt", "c_call_cnt_gjct",]

    # x_lst = ["bas_bid"]
    # x_lst = ["ln_his_exist_bad_debt", "ln_his_exist_tfr", "ln_his_exist_ovrd"]

    bn = BIN_NUM(x_lst, "forth_m2", 10, output=OUTPUT_PATH,)

    # 1. 先进行 fit 学习操作,
    # 学习过程中就可以得到属性 drop_list_ - 不需要的变量列表,
    # keep_list_ 需要的变量列表,
    # transform_dict_ 就是 keep_list_ 对应的离散化转换的实例
    # transform_group_dict_ 记录的是 keep_list_ 中训练数据的转换的 dataframe, 以供将来计算一些值使用
    bn.fit(df_D[x_lst], df_D["forth_m2"])
    print (bn.drop_list_)
    print (bn.keep_list_)
    print (bn.transform_dict_)
    print (bn.transform_group_dict_)

    # 2. 对于新的数据就可以进行离散化的转换
    print (bn.transform(df_D[x_lst]))

    # 3. 可以对 新的数据 进行 profile,
    # profile 输出就是 一个 excel - profile.csv, 以及一个合并的 dataframe
    # 以及属性 profile_dict_ - 画像字典
    bn.profile(df_D[x_lst], df_D["forth_m2"],
               OUTPUT_PATH + "profile.csv", label=True)
    print (bn.profile_dict_)

    # 4. 可以 计算新数据的IV
    # 调用 IV 功能，就可以得到
    # iv_dict_ - keep_list_ 的 iv 字典
    print (bn.iv(df_D[x_lst], df_D["forth_m2"], OUTPUT_PATH + "iv.csv"))
    print (bn.iv_dict_)

    print (bn.iv(df_T[x_lst], df_T["forth_m2"], OUTPUT_PATH + "iv_test.csv"))
    print (bn.iv_dict_)

    # 5. 可以输出 JS 文件
    # 默认生成 iv.js 和 iv.html 文件
    # 可以快速作为 可视化展示
    bn._profile_js()

    # 6. 可以生成 SQL 语句
    print (bn._generate_sql())

    # 6. 可以计算 训练集 和 新数据之间的 PSI
    # 调用 PSI 功能后，就可以得到
    # psi_dict_ - keep_list_ 的 psi 字典
    # psi_group_dict_ - keep_list_ 的 psi 计算的 df 的字典
    print (bn.psi(df_T[x_lst], output=OUTPUT_PATH + "psi.csv"))
    print (bn.psi_dict_)
    print (bn.psi_group_dict_)
    for k, v in bn.psi_group_dict_.iteritems():
        v.to_csv(OUTPUT_PATH + "psi_%s.csv" % k, index=False)


def ex_generate_case_when_sql():
    print (generate_case_when_sql("crd_amt", [3000, 5000, 8000, 10000, 20000, ], right=True, as_var=True))




if __name__ == "__main__":
    ex_generate_case_when_sql()