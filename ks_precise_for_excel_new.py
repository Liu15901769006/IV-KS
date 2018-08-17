#-*- coding: utf-8 -*-

'''
'''
import pandas as pd



def observe_and_score_write_ks(inputfile, outputfile, ks_print=True, trend_print=True):
    '''

        :param inputfile: csv 文件路径, 必须包含表头, 必须有 observe 和 score
        :type inputfile: str

        :param outputfile: excel 文件路径
        :type outputfile: str

        :param ks_print: 是否计算KS表
        :type ks_print: Boolean

        :param trend_print: 是否计算趋势图
        :type trend_print: Boolean

        :returns:
        :rtype:

    '''
    def write_df_to_excel(xlsname, df_dict):
        with pd.ExcelWriter(xlsname) as writer:
            for key, value in df_dict.items():
                value.to_excel(writer, sheet_name=key, index=False)


    df = pd.read_csv(inputfile)

    df = df[["observe", "score"]]
    df.sort_values(by=['score'], inplace=True)

    df_len = len(df)
    df.index = range(1, len(df)+1)

    cut_lst = [df.loc[int(ii*df_len/10)]["score"] for ii in range(1, 11)]

    cut_df = pd.DataFrame(cut_lst, columns=["cut"])
    cut_df["cut_bin"] = range(0, 10)

    df['bin'] = df['score'].apply(lambda x: cut_df[cut_df['cut'] >=x]['cut_bin'].min())
    df_ks_base = {}

    ## ks 图
    if ks_print is True:
        df_ks_base = df.groupby("bin").agg({
                    "score": ["max", "min"], 
                    "observe": ["count", "sum"]
                    })

        df_ks_base.columns = df_ks_base.columns.droplevel()
        df_ks_base.rename(columns={
                                "count": u"总样本", 
                                "max": u"最高分",
                                "min": u"最低分",
                                "sum": u"违约样本",
                                }, inplace=True)

        df_ks_base[u"违约率"] = df_ks_base[u"违约样本"] / df_ks_base[u"总样本"]
        df_ks_base[u"未违约样本"] = df_ks_base[u"总样本"] - df_ks_base[u"违约样本"]
        df_ks_base[u"违约累计"] = df_ks_base[u"违约样本"].cumsum() / df_ks_base[u"违约样本"].sum()
        df_ks_base[u"未违约累计"] = df_ks_base[u"未违约样本"].cumsum() / df_ks_base[u"未违约样本"].sum()

        df_ks_base[u"KS"] = df_ks_base[u"违约累计"] - df_ks_base[u"未违约累计"]

        df_ks_base_wyl = 1.0 * df_ks_base[u"违约样本"].sum() / df_ks_base[u"总样本"].sum()
        df_ks_base[u"提升"] = df_ks_base[u"违约率"] / df_ks_base_wyl
        df_ks_base[u"累计提升"] = df_ks_base[u"违约样本"].cumsum() / df_ks_base[u"总样本"].cumsum() / df_ks_base_wyl

        df_ks_base = df_ks_base[[u"最低分", u"最高分", u"总样本", u"违约样本", u"违约率"
                            , u"未违约样本", u"违约累计", u"未违约累计", u"KS", u"提升", u"累计提升"]]
        # print df_ks_base
    df_sd_20 = {}

    if trend_print is True:
        # zipped = list(zip([260, 280, 300, 320, 340, 360, 380,
        #                             400, 420, 440, 460, 480, 500, 520, 540,
        #                             560, 580, 600, 620, 640, 660, 680, 700, 720, 1000],
        #     range(0, 25)))
        zipped = list(zip([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100],
            range(0, 21)))

        cut_df20 = pd.DataFrame(zipped, columns=["cut", "cut_bin"])

        df['bin_20'] = df['score'].apply(lambda x: cut_df20[cut_df20['cut'] >x]['cut_bin'].min())
        df_sd_20 = df.groupby("bin_20").agg({"observe": ["count", "sum"]})
        df_sd_20.columns = df_sd_20.columns.droplevel()

        df_sd_20.rename(columns={"count": u"总体", "sum": u"违约"}, inplace=True)
        index_lst = list(df_sd_20.index)
        # delta_lst = list(set(range(0, 25)) - set(index_lst))
        # for x in delta_lst:
        #     df_sd_20.ix[x, u"总体"] = 0
        #     df_sd_20.ix[x, u"违约"] = 0
        # df_sd_20.sort_index(inplace=True)
        # df_sd_20 = df_sd_20.reindex(range(0, 25), fill_value=0)
        df_sd_20 = df_sd_20.reindex(range(0, 20), fill_value=0)

        df_sd_20[u"违约率"] = df_sd_20[u"违约"] / df_sd_20[u"总体"]
        df_sd_20[u"分布"] = df_sd_20[u"总体"] / df_sd_20[u"总体"].sum()
        df_sd_20[u"累计分布"] = df_sd_20[u"分布"].cumsum()
        # df_sd_20[u"划分"] = [u"260 以下", u"[260,280)", u"[280,300)", u"[300,320)",
        #                     u"[320,340)", u"[340,360)", u"[360,380)", u"[380,400)",
        #                     u"[400,420)", u"[420,440)", u"[440,460)", u"[460,480)",
        #                     u"[480,500)", u"[500,520)", u"[520,540)", u"[540,560)",
        #                     u"[560,580)", u"[580,600)", u"[600,620)", u"[620,640)",
        #                     u"[640,660)", u"[660,680)", u"[680,700)", u"[700,720)", u"720 以上"]
        df_sd_20[u"划分"] = [u"[0,5)", u"[5,10)", u"[10,15)", u"[15,20)", u"[20,25)", u"[25,30)", u"[30,35)",
                            u"[35,40)", u"[40,45)", u"[45,50)", u"[50,55)",
                            u"[55,60)", u"[60,65)", u"[65,70)", u"[70,75)",
                            u"[75,80)", u"[80,85)", u"[85,90)", u"[90,95)", u"[95,100)"]
        df_sd_20 = df_sd_20[[
                            u"划分", 
                            u"总体", u"违约", u"违约率", u"分布", u"累计分布"]]

    df_dict = {}
    if ks_print is True:
        df_dict.update({u'KS': df_ks_base})
    if trend_print is True:
        df_dict.update({
            # u'70': df_sd_70,
            # u'50': df_sd_50,
            u'20': df_sd_20})

    write_df_to_excel(outputfile, df_dict)

    if ks_print is True:
        return df_ks_base["KS"].max()


def run():
    inputfile = u'E:/haoan/data/4月.csv'
    outputfile = u'E:/haoan/result/4月.xlsx'
    observe_and_score_write_ks(inputfile, outputfile, ks_print=True, trend_print=True)


if __name__ == "__main__":
    run()





