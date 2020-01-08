# -*-coding:utf-8-*-
import numbers
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession

from pipline.selector.chiSquareFunc import *
###################################
#基于卡方分箱法对变量进行分箱#
###################################
'''
对不同类型的变量，分箱的处理是不同的：
（1）数值型变量可直接分箱
（2）取值个数较多的类别型变量，需要用bad rate做编码转换成数值型变量，再分箱
（3）取值个数较少的类别型变量不需要分箱，但是要检查是否每个类别都有好坏样本。如果有类别只有好或坏，需要合并
'''

class chiSquareMerge:

    def __init__(self):
        self.categorical_var=[]
        self.numerical_var=[]
        self.deleted_features = []  # 将处理过的变量删除，防止对后面建模的干扰
        self.encoded_features = {}  # 将bad rate编码方式保存下来，在以后的测试和生产环境中需要使用
        self.merged_features = {}  # 将类别型变量合并方案保留下来
        self.var_IV = {}  # save the IV values for binned features       #将IV值和WOE值保留
        self.var_WOE = {}



    def get_num_and_cat(self,df):
        '''
        get numerical_column and categary_column
        :return:
        '''
        allFeatures = list(df.columns)
        allFeatures.remove('CUST_NO')
        allFeatures.remove('IS_TARGET')
        numerical_var = []

        # 剔除常量
        def del_constant(df,allFeatures):
            for col in allFeatures:
                if len(set(df[col])) == 1:
                    print('delete {} from the dataset because it is a constant'.format(col))
                    del df[col]
                    allFeatures.remove(col)
        for i in range(3):
            # 不循环删除会有部分字段遗留
            del_constant(df,allFeatures)

        # 类别区分
        for col in allFeatures:
            uniq_vals = list(set(df[col]))
            if np.nan in uniq_vals:
                uniq_vals.remove(np.nan)
            if len(uniq_vals) >= 10 and isinstance(uniq_vals[0], numbers.Real):
                numerical_var.append(col)

        categorical_var = [i for i in allFeatures if i not in numerical_var]

        self.categorical_var = categorical_var
        self.numerical_var = numerical_var
        print("categorical_var:{0},\nlen:{1}".format(categorical_var,len(categorical_var)))
        print("numerical_var:{0},\nlen:{1}".format(numerical_var,len(numerical_var)))

        return categorical_var,numerical_var

    def cat_col_process(self,df):
        '''
        主函数1
        对于类别型变量，按照以下方式处理
        1，如果变量的取值个数超过5，计算bad rate进行编码
        2，除此之外，其他任何类别型变量如果有某个取值中，对应的样本全部是坏样本或者是好样本，进行合并。
        :return:
        '''

        # deleted_features = []  # 将处理过的变量删除，防止对后面建模的干扰
        # encoded_features = {}  # 将bad rate编码方式保存下来，在以后的测试和生产环境中需要使用
        # merged_features = {}  # 将类别型变量合并方案保留下来
        # var_IV = {}  # save the IV values for binned features       #将IV值保留和WOE值
        # var_WOE = {}
        for col in self.categorical_var:
            print('we are processing {}'.format(col))
            if len(set(df[col])) > 5:
                print('{} is encoded with bad rate'.format(col))
                col0 = str(col) + '_encoding'

                # (1), 计算坏样本率并进行编码
                encoding_result = BadRateEncoding(df, col, 'IS_TARGET')
                df[col0], br_encoding = encoding_result['encoding'], encoding_result['bad_rate']

                # (2), 将（1）中的编码后的变量也加入数值型变量列表中，为后面的卡方分箱做准备
                self.numerical_var.append(col0)

                # (3), 保存编码结果
                self.encoded_features[col] = [col0, br_encoding]

                # (4), 删除原始值
                self.deleted_features.append(col)
            else:
                bad_bin = df.groupby([col])['IS_TARGET'].sum()
                # 对于类别数少于5个，但是出现0坏样本的组的特征需要做处理
                '''举例（只是改动了上面结果，数字是不正确的，方便理解而已）
                CUST_LOYALTY  total  bad  bad_rate
                     1         635   75   0.1918110
                     2         1909  176  0.092195
                     3          229   0    0.0
                     '''
                if min(bad_bin) == 0:
                    print('{} has 0 bad sample!'.format(col))
                    col1 = str(col) + '_mergeByBadRate'
                    # (1), 找出最优合并方式，使得每一箱同时包含好坏样本
                    mergeBin = MergeBad0(df, col, 'IS_TARGET')
                    # (2), 依照（1）的结果对值进行合并
                    df[col1] = df[col].map(mergeBin)
                    maxPcnt = MaximumBinPcnt(df, col1)
                    # 如果合并后导致有箱占比超过90%，就删除。
                    if maxPcnt > 0.9:
                        print('{} is deleted because of large percentage of single bin'.format(col))
                        self.deleted_features.append(col)
                        self.categorical_var.remove(col)
                        del df[col]
                        continue
                    # (3) 如果合并后的新的变量满足要求，就保留下来
                    self.merged_features[col] = [col1, mergeBin]
                    WOE_IV = CalcWOE(df, col1, 'IS_TARGET')
                    self.var_WOE[col1] = WOE_IV['WOE']
                    self.var_IV[col1] = WOE_IV['IV']
                    # del trainData[col]
                    self.deleted_features.append(col)
                else:
                    WOE_IV = CalcWOE(df, col, 'IS_TARGET')
                    self.var_WOE[col] = WOE_IV['WOE']
                    self.var_IV[col] = WOE_IV['IV']
        print(self.var_WOE)
        print(self.var_IV)


    def num_col_process(self, df):
        '''
        主函数2
        对于连续型变量，处理方式如下：
        1，利用卡方分箱法将变量分成5个箱
        2，检查坏样本率的单带性，如果发现单调性不满足，就进行合并，直到满足单调性
        :return:
        '''
        var_cutoff = {}
        for col in self.numerical_var:
            print("{} is in processing".format(col))
            col1 = str(col) + '_Bin'

            # (1),用卡方分箱法进行分箱，并且保存每一个分割的端点。例如端点=[10,20,30]表示将变量分为x<10,10<x<20,20<x<30和x>30.
            # 特别地，缺失值-1不参与分箱,补缺时标为-1
            if -1 in set(df[col]):
                special_attribute = [-1]
            else:
                special_attribute = []
            cutOffPoints = ChiMerge(df, col, 'IS_TARGET', special_attribute=special_attribute)
            var_cutoff[col] = cutOffPoints
            df[col1] = df[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))

            # (2),检查坏样本率的单带性，如果发现单调性不满足，就进行合并，直到满足单调性
            BRM = BadRateMonotone(df, col1, 'IS_TARGET', special_attribute=special_attribute)
            if not BRM:
                if special_attribute == []:
                    bin_merged = Monotone_Merge(df, 'IS_TARGET', col1)  # 传入是确定非单调的bin列
                    print('bin_merged', bin_merged)
                    # 非单调的列，合并后的列需要删除之前的列的一半
                    removed_index = []
                    for bin in bin_merged:
                        if len(bin) > 1:
                            indices = [int(b.replace('Bin ', '')) for b in bin]
                            removed_index = removed_index + indices[0:-1]
                    removed_point = [cutOffPoints[k] for k in removed_index]
                    for p in removed_point:
                        cutOffPoints.remove(p)
                    var_cutoff[col] = cutOffPoints
                    df[col1] = df[col].map(
                        lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
                else:
                    cutOffPoints2 = [i for i in cutOffPoints if i not in special_attribute]
                    temp = df.loc[~df[col].isin(special_attribute)]
                    bin_merged = Monotone_Merge(temp, 'IS_TARGET', col1)
                    removed_index = []
                    for bin in bin_merged:
                        if len(bin) > 1:
                            indices = [int(b.replace('Bin ', '')) for b in bin]
                            removed_index = removed_index + indices[0:-1]
                    removed_point = [cutOffPoints2[k] for k in removed_index]
                    for p in removed_point:
                        cutOffPoints2.remove(p)
                    cutOffPoints2 = cutOffPoints2 + special_attribute
                    var_cutoff[col] = cutOffPoints2
                    df[col1] = df[col].map(
                        lambda x: AssignBin(x, cutOffPoints2, special_attribute=special_attribute))

            # (3), 分箱后再次检查是否有单一的值占比超过90%。如果有，删除该变量
            maxPcnt = MaximumBinPcnt(df, col1)
            if maxPcnt > 0.9:
                # del trainData[col1]
                self.deleted_features.append(col)
                self.numerical_var.remove(col)
                print('we delete {} because the maximum bin occupies more than 90%'.format(col))
                continue

            WOE_IV = CalcWOE(df, col1, 'IS_TARGET')
            self.var_IV[col] = WOE_IV['IV']
            self.var_WOE[col] = WOE_IV['WOE']
        print(self.var_IV)
        print(self.var_WOE)

        # 利用计算的WOE新增df的列,新列为使用WOE值进行编码的列
        for col in self.var_WOE.keys():
            col2 = str(col) + "_WOE"
            if col in var_cutoff.keys():
                cutOffPoints = var_cutoff[col]
                special_attribute = []
                if - 1 in cutOffPoints:
                    special_attribute = [-1]
                binValue = df[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=special_attribute))
                df[col2] = binValue.map(lambda x: self.var_WOE[col][x])
            else:
                df[col2] = df[col].map(lambda x: self.var_WOE[col][x])


    def select_imp_col(self):

        # WOE编码后的单变量分析与多变量分析
        # 选择IV高于阈值的变量
        all_IV = list(self.var_IV.values())
        all_IV = sorted(all_IV, reverse=True)
        # plt.bar(x=range(len(all_IV)), height=all_IV)
        iv_threshould = 0.02
        varByIV = [k for k, v in self.var_IV.items() if v > iv_threshould]







if __name__=="__main__":
    spark = SparkSession.builder \
        .config("spark.sql.warehouse.dir", "../spark-warehouse") \
        .appName('word_count_app') \
        .master('local[2]') \
        .getOrCreate()
    df = spark.read.csv("file:///home/hadoop/work/pycharmProject/data/train_data1.txt", header=True, inferSchema=True)
    df=df.toPandas()
    chimerge=chiSquareMerge()
    categorical_var,numerical_var=chimerge.get_num_and_cat(df)
    # dict=chimerge.BadRateEncoding(df,'CUST_LOYALTY','IS_TARGET')
    chimerge.cat_col_process(df)
    chimerge.num_col_process(df)