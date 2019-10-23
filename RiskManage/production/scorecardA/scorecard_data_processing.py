import pandas as pd
import datetime
import collections
import numpy as np
import numbers
import random
import sys
import pickle
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from importlib import reload
from matplotlib import pyplot as plt
reload(sys)
#sys.setdefaultencoding( "utf-8")
#from scorecard_functions import *
from sklearn.linear_model import LogisticRegressionCV
# -*- coding: utf-8 -*-

'''
脚本目标：
数据分析的工作, 包括读取数据文件、检查用户Id的一致性、特征工程、填充等
'''


folderOfData='/Users/jacquelin/Documents/python/riskManegeproject/riskManageProcessing/'
data1 = pd.read_csv(folderOfData+'PPD_LogInfo_3_1_Training_Set.csv', header = 0)
data2 = pd.read_csv(folderOfData+'PPD_Training_Master_GBK_3_1_Training_Set.csv', header = 0,encoding = 'gbk')
data3 = pd.read_csv(folderOfData+'PPD_Userupdate_Info_3_1_Training_Set.csv', header = 0)

'''
PPD_LogInfo_3_1_Training_Set
Idx	    Listinginfo1	LogInfo1	LogInfo2	LogInfo3
10001	2014/3/5	      107	      6	     2014/2/20
10001	2014/3/5	      107	      6	     2014/2/23
Log_Info（借款人的登陆信息）
idx：每一笔贷款的unique key
ListingInfo：借款成交时间
LogInfo1：操作代码
LogInfo2：操作类别
LogInfo3：登陆时间


PPD_Training_Master_GBK_3_1_Training_Set
Idx	UserInfo_1	UserInfo_2	UserInfo_3	UserInfo_4	WeblogInfo_1	WeblogInfo_2	WeblogInfo_3	WeblogInfo_4	WeblogInfo_5	WeblogInfo_6	WeblogInfo_7	WeblogInfo_8	WeblogInfo_9	WeblogInfo_10	WeblogInfo_11	WeblogInfo_12	WeblogInfo_13	WeblogInfo_14	WeblogInfo_15	WeblogInfo_16	WeblogInfo_17	WeblogInfo_18	UserInfo_5	UserInfo_6	UserInfo_7	UserInfo_8	UserInfo_9	UserInfo_10	UserInfo_11	UserInfo_12	UserInfo_13	UserInfo_14	UserInfo_15	UserInfo_16	UserInfo_17	UserInfo_18	UserInfo_19	UserInfo_20	UserInfo_21	UserInfo_22	UserInfo_23	UserInfo_24	Education_Info1	Education_Info2	Education_Info3	Education_Info4	Education_Info5	Education_Info6	Education_Info7	Education_Info8	WeblogInfo_19	WeblogInfo_20	WeblogInfo_21	WeblogInfo_23	WeblogInfo_24	WeblogInfo_25	WeblogInfo_26	WeblogInfo_27	WeblogInfo_28	WeblogInfo_29	WeblogInfo_30	WeblogInfo_31	WeblogInfo_32	WeblogInfo_33	WeblogInfo_34	WeblogInfo_35	WeblogInfo_36	WeblogInfo_37	WeblogInfo_38	WeblogInfo_39	WeblogInfo_40	WeblogInfo_41	WeblogInfo_42	WeblogInfo_43	WeblogInfo_44	WeblogInfo_45	WeblogInfo_46	WeblogInfo_47	WeblogInfo_48	WeblogInfo_49	WeblogInfo_50	WeblogInfo_51	WeblogInfo_52	WeblogInfo_53	WeblogInfo_54	WeblogInfo_55	WeblogInfo_56	WeblogInfo_57	WeblogInfo_58	ThirdParty_Info_Period1_1	ThirdParty_Info_Period1_2	ThirdParty_Info_Period1_3	ThirdParty_Info_Period1_4	ThirdParty_Info_Period1_5	ThirdParty_Info_Period1_6	ThirdParty_Info_Period1_7	ThirdParty_Info_Period1_8	ThirdParty_Info_Period1_9	ThirdParty_Info_Period1_10	ThirdParty_Info_Period1_11	ThirdParty_Info_Period1_12	ThirdParty_Info_Period1_13	ThirdParty_Info_Period1_14	ThirdParty_Info_Period1_15	ThirdParty_Info_Period1_16	ThirdParty_Info_Period1_17	ThirdParty_Info_Period2_1	ThirdParty_Info_Period2_2	ThirdParty_Info_Period2_3	ThirdParty_Info_Period2_4	ThirdParty_Info_Period2_5	ThirdParty_Info_Period2_6	ThirdParty_Info_Period2_7	ThirdParty_Info_Period2_8	ThirdParty_Info_Period2_9	ThirdParty_Info_Period2_10	ThirdParty_Info_Period2_11	ThirdParty_Info_Period2_12	ThirdParty_Info_Period2_13	ThirdParty_Info_Period2_14	ThirdParty_Info_Period2_15	ThirdParty_Info_Period2_16	ThirdParty_Info_Period2_17	ThirdParty_Info_Period3_1	ThirdParty_Info_Period3_2	ThirdParty_Info_Period3_3	ThirdParty_Info_Period3_4	ThirdParty_Info_Period3_5	ThirdParty_Info_Period3_6	ThirdParty_Info_Period3_7	ThirdParty_Info_Period3_8	ThirdParty_Info_Period3_9	ThirdParty_Info_Period3_10	ThirdParty_Info_Period3_11	ThirdParty_Info_Period3_12	ThirdParty_Info_Period3_13	ThirdParty_Info_Period3_14	ThirdParty_Info_Period3_15	ThirdParty_Info_Period3_16	ThirdParty_Info_Period3_17	ThirdParty_Info_Period4_1	ThirdParty_Info_Period4_2	ThirdParty_Info_Period4_3	ThirdParty_Info_Period4_4	ThirdParty_Info_Period4_5	ThirdParty_Info_Period4_6	ThirdParty_Info_Period4_7	ThirdParty_Info_Period4_8	ThirdParty_Info_Period4_9	ThirdParty_Info_Period4_10	ThirdParty_Info_Period4_11	ThirdParty_Info_Period4_12	ThirdParty_Info_Period4_13	ThirdParty_Info_Period4_14	ThirdParty_Info_Period4_15	ThirdParty_Info_Period4_16	ThirdParty_Info_Period4_17	ThirdParty_Info_Period5_1	ThirdParty_Info_Period5_2	ThirdParty_Info_Period5_3	ThirdParty_Info_Period5_4	ThirdParty_Info_Period5_5	ThirdParty_Info_Period5_6	ThirdParty_Info_Period5_7	ThirdParty_Info_Period5_8	ThirdParty_Info_Period5_9	ThirdParty_Info_Period5_10	ThirdParty_Info_Period5_11	ThirdParty_Info_Period5_12	ThirdParty_Info_Period5_13	ThirdParty_Info_Period5_14	ThirdParty_Info_Period5_15	ThirdParty_Info_Period5_16	ThirdParty_Info_Period5_17	ThirdParty_Info_Period6_1	ThirdParty_Info_Period6_2	ThirdParty_Info_Period6_3	ThirdParty_Info_Period6_4	ThirdParty_Info_Period6_5	ThirdParty_Info_Period6_6	ThirdParty_Info_Period6_7	ThirdParty_Info_Period6_8	ThirdParty_Info_Period6_9	ThirdParty_Info_Period6_10	ThirdParty_Info_Period6_11	ThirdParty_Info_Period6_12	ThirdParty_Info_Period6_13	ThirdParty_Info_Period6_14	ThirdParty_Info_Period6_15	ThirdParty_Info_Period6_16	ThirdParty_Info_Period6_17	ThirdParty_Info_Period7_1	ThirdParty_Info_Period7_2	ThirdParty_Info_Period7_3	ThirdParty_Info_Period7_4	ThirdParty_Info_Period7_5	ThirdParty_Info_Period7_6	ThirdParty_Info_Period7_7	ThirdParty_Info_Period7_8	ThirdParty_Info_Period7_9	ThirdParty_Info_Period7_10	ThirdParty_Info_Period7_11	ThirdParty_Info_Period7_12	ThirdParty_Info_Period7_13	ThirdParty_Info_Period7_14	ThirdParty_Info_Period7_15	ThirdParty_Info_Period7_16	ThirdParty_Info_Period7_17	SocialNetwork_1	SocialNetwork_2	SocialNetwork_3	SocialNetwork_4	SocialNetwork_5	SocialNetwork_6	SocialNetwork_7	SocialNetwork_8	SocialNetwork_9	SocialNetwork_10	SocialNetwork_11	SocialNetwork_12	SocialNetwork_13	SocialNetwork_14	SocialNetwork_15	SocialNetwork_16	SocialNetwork_17	target	ListingInfo
10001	1	深圳	4	深圳		1		1	1	1	14	0	0	0	0	0	0	6	6	0	6	2	2	2	广东	深圳	中国移动 	0				2	2	1	1	29	四川省	南充市	0	D	D	D	0	E	E	E	0	E	E	E	I	I5	D	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	10	47	167	0	25	65	68	34	2	1	10	63	27489	0	18900	6083	2506	25	78	312	0	46	14	186	112	1	1	16	0	33477	0	642	19267	13568	36	79	394	0	55	6	244	144	6	3	12	1	53440	0	184	30284	22972	4	56	176	0	47	1	134	41	0	0	5	1	13379	0	0	9297	4082	0	98	277	0	45	10	193	74	1	1	10	0	23516	0	1571	16882	5063	0	45	81	0	20	3	53	25	1	1	4	0	10180	0	199	6969	3012	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	0	0	-1	-1	-1	-1	-1	126	234	222	-1	0	0	0	0	0	1	0	2014/3/5
10002	1	温州	4	温州		0		1	1	1	14	0	0	0	0	0	0	0	0	7	7	0	2	2	浙江	温州	中国移动 	1	0	0	1	3	3	2	2	28	福建省	不详	0	D	D	D	0	E	E	E	0	E	E	E	I	I5	D	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	1	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	0	68	105	40	2	132	39	2	1	8	0	3839	9558	49	9571	3777	7	1	118	152	56	4	219	47	8	5	13	0	15870	16325	186	26656	5353	1	0	205	247	74	0	387	65	7	5	19	0	20784	30202	0	42483	8503	0	0	302	295	81	1	487	109	11	5	22	1	29389	39214	29	54055	14519	0	0	299	267	75	0	474	92	11	5	11	0	24692	30781	0	46372	9101	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	-1	0	0	-1	-1	-1	-1	-1	33	110	1	-1	0	0	0	0	0	2	0	2014/2/26
Master(每一行代表一个成功成交借款样本，每个样本包含200多个各类字段。
idx：每笔贷款的unique key，可与另外2个文件里的idx相匹配。
UserInfo_*：借款人特征字段
WeblogInfo_*：Info网络行为字段
Education_Info*：学历学籍字段
ThirdParty_Info_PeriodN_*：第三方数据时间段N字段
SocialNetwork_*：社交网络字段
ListingInfo：借款成交时间
Target：违约标签（1 = 贷款违约，0 = 正常还款）。测试集里 不包含target字段。


PPD_Userupdate_Info_3_1_Training_Set
Idx	ListingInfo1	UserupdateInfo1	UserupdateInfo2
10001	2014/3/5	_EducationId	2014/2/20
10001	2014/3/5	_HasBuyCar	2014/2/20
Userupdate_Info（借款⼈修改信息）
idx：每⼀笔贷款的unique key
ListingInfo1：借款成交时间
UserupdateInfo1：修改内容
UserupdateInfo2：修改时间

'''

##UDF
def TimeWindowSelection(df,daysCol,time_windows):
    '''
    时间窗口累计
    '''
    freq_tw={}
    for tw in time_windows:
        freq=sum(df[daysCol].apply(lambda x:int(x<=tw)))
        freq_tw[tw]=freq
    return freq_tw

def DevideByZero(nominator,denominator):
    '''
    （计算公式）分母为0时返回0；否则返回计算值
    '''
    if denominator==0:
        return 0
    else :
        return nominator*1.0/denominator
    
def ChangeContent(x):
    '''
    清洗字段值
    '''
    y=x.upper()
    if y=='_MOBILEPHONE':
        y='_PHONE'
    return y

def MissingCategorial(df,col):
    missing_vals=df[col].map(lambda x:int(x!=x))    #在数学中inf==inf,在Python中nan与nan不等，谁也无法确定无限大那个数是多少，故可以用是否等于自身判断是否为nan
    return sum(missing_vals)*1.0/df.shape[0]

    
# Step 1: 从PPD_LogInfo_3_1_Training_Set &  PPD_Userupdate_Info_3_1_Training_Set数据中衍生特征#

data2['city_match']=data2.apply(lambda x:int(x.UserInfo_2==x.UserInfo_4==x.UserInfo_8==x.UserInfo_20),axis=1)

del data2['UserInfo_2']
del data2['UserInfo_4']
del data2['UserInfo_8']
del data2['UserInfo_20']


### 提取申请日期，计算日期差，查看日期差的分布
data1['loginfo']=data1['LogInfo3'].map(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['Listinginfo']=data1['Listinginfo1'].map(lambda x:datetime.datetime.strptime(x,'%Y-%m-%d'))
data1['ListingGap']=data1[['loginfo','Listinginfo']].apply(lambda x:(x[1]-x[0]).days,axis=1)
plt.hist(data1['ListingGap'],bins=200)
plt.title('Days between log and list')
ListingGap2=data1['ListingGap'].apply(lambda x:min(x,365))
plt.hist(ListingGap2,bins=200)

timeWindows=TimeWindowSelection(data1,'ListingGap',range(30,361,30))

'''
使用180天作为最大的时间窗口计算新特征
所有可以使用的时间窗口可以有7 days, 30 days, 60 days, 90 days, 120 days, 150 days and 180 days.
在每个时间窗口内，计算总的登录次数，不同的登录方式，以及每种登录方式的平均次数
'''
time_window=[7,30,60,90,120,150,180]
var_list=['LogInfo1','LogInfo2']
data1GroupbyIdx=pd.DataFrame({'Idx':data1['Idx'].drop_duplicates()})

for tw in time_window:
    #最近tw天内的各种行为df
    data1['TruncateLogInfo']=data1['Listinginfo'].map(lambda x:x+datetime.timedelta(-tw))
    temp=data1.loc[data1['loginfo']>=data1['TruncateLogInfo']]
    for var in var_list:
        #count the frequences of LogInfo1 and LogInfo2
        count_stat=temp.groupby(['Idx'])[var].count().to_dict()
        data1GroupbyIdx[str(var)+'_'+str(tw)+'_count']=data1GroupbyIdx['Idx'].map(lambda x:count_stat.get(x,0))
        
        #count the distinct value of LogInfo1 and LogInfo2
        #观察每个用户在每种信息中的不同类型
        Idx_UserLogType=temp[['Idx',var]].drop_duplicates()
        uniq_stat=Idx_UserLogType.groupby('Idx')[var].count().to_dict()
        data1GroupbyIdx[str(var)+'_'+str(tw)+'_unique']=data1GroupbyIdx['Idx'].map(lambda x:uniq_stat.get(x,0))
        
        #calculate the average count of each value in LogInfo1 and LogInfo2
        data1GroupbyIdx[str(var)+'_'+str(tw)+'_avg']=data1GroupbyIdx[[str(var)+'_'+str(tw)+'_count',str(var)+'_'+str(tw)+'_unique']].apply(lambda x:DevideByZero(x[0],x[1]),axis=1)
    
#print(count_stat[3])
data3['Listinginfo']=data3['ListingInfo1'].map(lambda x:datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['UserupdateInfo']=data3['UserupdateInfo2'].map(lambda x:datetime.datetime.strptime(x,'%Y/%m/%d'))
data3['ListingGap']=data3[['UserupdateInfo','Listinginfo']].apply(lambda x:(x[1]-x[0]).days,axis=1)

'''
对 QQ和qQ, Idnumber和idNumber,MOBILEPHONE和PHONE 进行统一
在时间切片内，计算
 (1) 更新的频率
 (2) 每种更新对象的种类个数
 (3) 对重要信息如IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE的更新
'''
data3['UserupdateInfo1']=data3['UserupdateInfo1'].map(ChangeContent)
data3GroupbyIdx=pd.DataFrame({'Idx':data3['Idx'].drop_duplicates()})

time_window=[7,30,60,90,120,150,180]

for tw in time_window:
    #最近tw天内的各种行为df
    data3['TruncateLogInfo']=data3['Listinginfo'].map(lambda x: x+datetime.timedelta(-tw))
    temp=data3.loc[data3['UserupdateInfo']>=data3['TruncateLogInfo']]

    #count the frequences of UserupdateInfo
    freq_stat=temp.groupby(['Idx'])['UserupdateInfo'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_'+str(tw)+'_count']=data3GroupbyIdx['Idx'].map(lambda x:freq_stat.get(x,0))
    
    #count the distinct value of UserupdateInfo
    Idx_Userupdate=temp[['Idx','UserupdateInfo1']].drop_duplicates()
    uniq_stat=Idx_Userupdate.groupby('Idx')['UserupdateInfo1'].count().to_dict()
    data3GroupbyIdx['UserupdateInfo_'+str(tw)+'_unique']=data3GroupbyIdx['Idx'].map(lambda x:uniq_stat.get(x,0))
    
    #calculate the average count of each value in UserupdateInfo
    data3GroupbyIdx['UserupdateInfo_'+str(tw)+'_avg']=data3GroupbyIdx[['UserupdateInfo_'+str(tw)+'_count','UserupdateInfo_'+str(tw)+'_unique']].apply(lambda x:DevideByZero(x[0],x[1]),axis=1)
    
    #whether the applicant changed items like IDNUMBER,HASBUYCAR, MARRIAGESTATUSID, PHONE
    Idx_Userupdate['UserupdateInfo1']=Idx_Userupdate['UserupdateInfo1'].map(lambda x:[x])
    Idx_Userupdate_v2=Idx_Userupdate.groupby('Idx')['UserupdateInfo1'].sum()
    for item in ['_IDNUMBER','_HASBUYCAR','_MARRIAGESTATUSID','_PHONE']:
        item_dict=Idx_Userupdate_v2.map(lambda x :int(item in x)).to_dict()
        data3GroupbyIdx['UserupdateInfo_'+str(tw)+str(item)]=data3GroupbyIdx['Idx'].map(lambda x :item_dict.get(x,0))
    
#横向合并数据集
allData = pd.concat([data2.set_index('Idx'), data3GroupbyIdx.set_index('Idx'), data1GroupbyIdx.set_index('Idx')],axis= 1)


# Step 2: 对类别型变量和数值型变量进行补缺#

#特征
allFeatures=list(allData.columns)
allFeatures.remove('target')
if 'Idx' in allFeatures:
    allFeatures.remove('Idx')
allFeatures.remove('ListingInfo')

#检查是否有常数型变量(某列只有一种取值)，并且检查是类别型还是数值型变量
numerical_var=[]   #初始化数值型变量名
for col in allFeatures:
    if len(set(allData[col]))==1:
        print('delete {} from dataset because constant'.format(col))
        del allData[col]
        allFeatures.remove(col)
    else:
        uniq_valid_vals=[i for i in allData[col] if i==i]     #顺便剔除空值nan
        uniq_valid_vals_v2=list(set(uniq_valid_vals))
        if len(uniq_valid_vals_v2) >10 and isinstance(uniq_valid_vals_v2[0],numbers.Real):
            numerical_var.append(col)       
    ###函数解释：numbers.Real包括int和float数值类型；isinstance为验证某个值是否为已知的某类型
categorical_var=[i for i in allFeatures if i not in numerical_var]

#检查变量的最多值的占比情况,以及每个变量中占比最大的值
record_count=allData.shape[0]    #有多少条记录
col_most_values,col_large_value={},{}
for col in allFeatures:
    value_count=allData[col].groupby(allData[col]).count()     #按照该字段数值分组，计算各个数值出现次数
    col_most_values[col]=max(value_count)/record_count         #该字段中出现次数最多的值占所有样本的比例
    col_large_value[col]=value_count[value_count==max(value_count)].index[0]    #该字段值出现次数也即占比最大的值
col_most_values_df=pd.DataFrame.from_dict(col_most_values,orient='index')
col_most_values_df.columns=['max_percent']
col_most_values_df=col_most_values_df.sort_values(by='max_percent',ascending=False)
pcnt=list(col_most_values_df[:400]['max_percent'])
plt.bar(range(len(pcnt)),height=pcnt)
plt.title('Largest Percentage of Single Value in Each Variable')

#计算多数值产比超过90%的字段中，少数值的坏样本率是否会显著高于多数值
large_percent_cols=list(col_most_values_df[col_most_values_df['max_percent']>=0.9].index)
bad_rate_diff={}
for col in large_percent_cols:
    large_value=col_large_value[col]
    temp=allData[[col,'target']]
    temp[col]=temp.apply(lambda x:int(x[col]==large_value),axis=1)    #原样本该特征每个数值是否等于那个多数值，等于则记为1
    bad_rate=temp.groupby(col).mean()        #以样本是是少数值还是多数值做分组依据，其他列（’target‘）进行计算
    if bad_rate.iloc[0]['target']==0:
        bad_rate_diff[col]=0
        continue
    bad_rate_diff[col]=np.log(bad_rate.iloc[0]['target']/bad_rate.iloc[1]['target'])
#原理解释：
    #多数值里坏样本占比    
    #temp1=len(temp[(temp['target']==1) & (temp['WeblogInfo_14']==1)])/len(temp[temp['WeblogInfo_14']==1])
    #少数值里坏样本占比
    #temp2=len(temp[(temp['target']==1) & (temp['WeblogInfo_14']==0)])/len(temp[temp['WeblogInfo_14']==0])
bad_rate_diff_sorted=sorted(bad_rate_diff.items(),key=lambda x:x[1],reverse=True)
bad_rate_diff_sorted_values=[x[1] for x in bad_rate_diff_sorted]
plt.bar(x=range(len(bad_rate_diff_sorted_values)),height=bad_rate_diff_sorted_values)    

#结论：所有的少数值的坏样本率并没有显著高于多数值（10倍，ln约为2.3），意味着这些变量可以直接剔除
for col in large_percent_cols:
    if col in numerical_var:
        numerical_var.remove(col)
    else:
        categorical_var.remove(col)
    del allData[col]
    

    
'''
对类别型变量，如果缺失超过80%, 就删除，否则当成特殊的状态
'''
missing_pcnt_threshold_1=0.8
for col in categorical_var:
    missingRate=MissingCategorial(allData,col)
   # print('{} missing rate:{}'.format(col,missingRate))
    if missingRate>missing_pcnt_threshold_1:
        categorical_var.remove(col)
        del allData[col]
    if 0<missingRate<missing_pcnt_threshold_1:
        allData[col]=allData[col].map(lambda x:str(x).upper())  

'''
检查数值型变量
'''
missing_pcnt_threshold_2=0.8
deleted_var=[]
for col in numerical_var:
    missingRate=MissingCategorial(allData,col)
    if missingRate>missing_pcnt_threshold_2:
        deleted_var.append(col)
    else:
        if missingRate>0:
            #补缺
            not_missing=allData.loc[allData[col]==allData[col]][col]
            #loc与iloc：loc取索引为非缺失值得索引的行，[]内可灵活指定；iloc取第几行，【】内为数字
            missing_position=allData[allData[col]!=allData[col]][col].index
            not_missing_sample=random.sample(list(not_missing),len(missing_position))
            allData.loc[missing_position,col]=not_missing_sample
            missingRate2=MissingCategorial(allData,col)
            
if deleted_var!=[]:
    for col in deleted_var:
        numerical_var.remove(col)
        del allData[col]

allData.to_csv(folderOfData+'allData.csv', header=True,encoding='gbk', columns = allData.columns, index=False)



    

















