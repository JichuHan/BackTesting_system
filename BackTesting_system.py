#!/usr/bin/env python
# coding: utf-8

# In[3377]:


from utility import *
import utility as utl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker 
import statsmodels.api as sm
import pymssql as msql
import pymysql
from tqdm import tqdm
from datetime import datetime
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

import warnings
warnings.filterwarnings("ignore")

import pickle
def save_variable(v,filename):
    f = open(filename,'wb')
    pickle.dump(v,f)
    f.close()
    return filename

def load_variable(filename):
    try:
        f = open(filename,'rb')
        r = pickle.load(f)
        f.close()
        return r
    except EOFError:
        return ""

pickle_path = '/home/aifi/script/jupyter_projects/hjc/Pickles/'

conn_jy = msql.connect(server="10.0.185.138",user="jrkj",password="bpkdJ4_atXFJ7",database="JYDB")
conn = pymysql.connect(host="10.20.19.174",user="fasea_ro",password="1SzzsQl@fin",database="fasea")

import cx_Oracle
import os
import time
from datetime import date, timedelta
os.environ["NLS_LANG"] = "SIMPLIFIED CHINESE_CHINA.UTF8"
Oracle_USER = "jrkj"
Oracle_PWD = 'bpkdJ4_atXFJ7'
Oracle_HOST = '10.0.185.137'
Oracle_PORT = '1521'
Oracle_SERV = 'winddb'
conn_addr = Oracle_USER + '/' + Oracle_PWD + '@' + Oracle_HOST + ':' + Oracle_PORT + '/' + Oracle_SERV
conn_wd = cx_Oracle.connect(conn_addr)

import talib as tabs


# In[3378]:


# # 深入探索

# ## （一）数据格式转换

# In[3394]:

# ## （二）回测系统

# In[3400]:


import matplotlib.gridspec as gridspec
def draw_real(return_df,fac_name='Factor'):
    IC,num = return_df['IC'],return_df['Num'].mean()
    pos_rat = (IC>0).sum()/len(IC)
    IR = IC.mean()/IC.std()
    index_port,long_port,short_port = return_df['Index'],return_df['Long'],return_df['Short']
    cum_df = (1+return_df.drop(columns = ['IC','Num'])).cumprod()-1
    color,c = ['royalblue','orangered','green'],0
    
    fig  = plt.figure(figsize=(15,7))
    gs = gridspec.GridSpec(64,3) 
    ax2 = fig.add_subplot(gs[:,:])
    for i in cum_df.columns:
        ax2.plot(cum_df[i],color = color[c])
        c += 1
    ax21 = ax2.twinx()
    ax21.bar(IC.index,list(IC),color = 'deepskyblue')
    ax21.hlines(y = 0,xmin = IC.index[0],xmax = IC.index[-1],color = 'blue')
    ax21.plot(IC.rolling(window=20,min_periods = 1).mean(),color = 'navy')#,linestyle = ':')
    ax2.legend(['Index: shp '+str(round(index_port.mean()/index_port.std()*np.sqrt(252),3)) + '; Fin '+str(round(((1+index_port).cumprod()-1).dropna().iloc[-1],3)) + '; Ret '+str(round(index_port.mean()*252,3))+ '; Vol '+str(round(index_port.std()*np.sqrt(252),3)),
                'Long_port: shp '+str(round(long_port.mean()/long_port.std()*np.sqrt(252),3)) + '; Fin '+str(round(((1+long_port).cumprod()-1).dropna().iloc[-1],3)) + '; Ret '+str(round(long_port.mean()*252,3))+ '; Vol '+str(round(long_port.std()*np.sqrt(252),3)),
                'Short_port: shp '+str(round(short_port.mean()/short_port.std()*np.sqrt(252),3)) + '; Fin '+str(round(((1+short_port).cumprod()-1).dropna().iloc[-1],3)) + '; Ret '+str(round(short_port.mean()*252,3))+ '; Vol '+str(round(short_port.std()*np.sqrt(252),3))
               ],fontsize=15,loc = 'upper left')
    ax21.legend(['IC-roll20 IR: '+str(round(IR,3))+' Port_num: '+str(round(num,3)),'IC IC-mean: '+str(round(IC.mean(),3))+' IC_posr: '+str(round(pos_rat*100,2))+'%']
               ,fontsize=15,loc = 'lower right')
    ax2.set_title(fac_name+' Real portfolio performance',fontsize=20,pad=10)
    ax2.set_xlabel('Date',fontsize=15)
    ax2.grid(True)

def draw_group(level_return_df,fac_name='Factor'):
    color_list = ['orangered','coral','gold','chocolate','limegreen','green','turquoise','dodgerblue','royalblue','darkviolet']
    step = level_return_df.shape[1]-1
    color_sel = color_list[:step]
    steps = np.arange(0,1,1/step)
    group_name = ['Group '+str(i+1) for i in range(0,step)]
    fig  = plt.figure(figsize=(15,7))
    gs = gridspec.GridSpec(64,3) 
    ax3 = fig.add_subplot(gs[:,:])
    ax3.plot((1+level_return_df['Index']).cumprod()-1,color = 'red',linewidth=3,linestyle='--')
    for i in range(0,len(steps)):
        if i==len(steps)-1:
            ax3.plot((1+level_return_df[group_name[i]]).cumprod()-1,color = color_sel[-1])
        else:
            ax3.plot((1+level_return_df[group_name[i]]).cumprod()-1,color = color_sel[i])
    ax3.legend(['Index']+group_name,loc = 'upper left',fontsize=15)
    ax3.set_title(fac_name+' Group performance',fontsize=20,pad=10)
    t = [str(i)[:10] for i in level_return_df.index[::int(len(level_return_df.index)/5)]]
    ax3.set_xticks(t)
    ax3.set_xticklabels(labels = t,rotation=20)
    ax3.set_xlabel('Date',fontsize=15)
    ax3.grid()
    
def draw_group_rank(level_rank_df,fac_name='Factor'):
    color_list = ['orangered','coral','gold','chocolate','limegreen','green','turquoise','dodgerblue','royalblue','darkviolet']
    level_rank_df = level_rank_df.drop(columns='Index')
    step = level_rank_df.shape[1]-1
    color_sel = color_list[:step]
    steps = np.arange(0,1,1/step)
    group_name = ['Group '+str(i+1) for i in range(0,step)]
    fig  = plt.figure(figsize=(15,3))
    gs = gridspec.GridSpec(64,3) 
    ax4 = fig.add_subplot(gs[:,:])
    for i in range(0,len(level_rank_df.columns)):
        col_name = level_rank_df.columns[i]
        if col_name=='Cor':
            ax42 = ax4.twinx()
            ax42.plot(level_rank_df.Cor,color = 'red',linewidth = 5,linestyle=':')
            ax42.hlines(y = 0,xmin = level_rank_df.index[0],xmax = level_rank_df.index[-1]
                       ,color = 'red',linewidth = 2,linestyle = ':')
        elif i==0 or i==len(level_rank_df.columns)-2 or i==int((len(level_rank_df.columns)-2)/2):
            ax4.plot(level_rank_df[col_name],color_sel[i],linewidth = 2)
        else:
            ax4.plot(level_rank_df[col_name],color_sel[i],linewidth = 0.7)
    ax4.legend(level_rank_df.columns,loc='center left',fontsize=10)
    ax42.legend(['Cor: '+str(round(level_rank_df.Cor.mean(),3))],loc = 'lower right',fontsize=15)
    ax4.set_title(fac_name+' Rank performance',fontsize=20,pad=10)
    ax4.set_xlabel('Date',fontsize=15)
    ax4.grid()


# In[3401]:


def uni(factor):
    return (factor-factor.min())/(factor.max()-factor.min())
def uni_rank(factor):
    factor = factor.rank(axis=0, pct = True)
    return factor

def get_mask(Signal,limit):
    Signal_cum = Signal.fillna(axis=1,method='ffill',limit=limit-1).cumsum(axis=1)
    Signal_cum_fill = (Signal*Signal_cum).fillna(axis=1,method='ffill',limit=limit-1)
    mask_count = Signal_cum - Signal_cum_fill + 1
    mask = (~mask_count.isna()).replace([True,False],[1,np.nan])
    return mask,mask_count


def backtest_cdb(factor,ret_df_t, quantile = 0.02,limit = 5,
                 date_range=['20180101','20200101'],fac_name='Factor',
                             cal = ['Count','Real','Group','Rank'],draw=True):
    #三 截取日期区间
    start_date,end_date = datetime.strptime(str(date_range[0]),'%Y%m%d'),datetime.strptime(str(date_range[1]),'%Y%m%d')
    factor = factor.loc[:,start_date:end_date].copy()
    ret_df_t = ret_df_t.loc[:,start_date:end_date].copy()
    #四 计算
    return_dict = {}
    #（1）实盘组合统计：
    if 'Real' in cal:
        index_port = ret_df_t.mean()
        IC = (factor).corrwith((ret_df_t),method = 'spearman')
        factor = factor.rank(axis=0, pct = True,na_option='keep',ascending=True)
        factor = factor - factor.min()/2
        long_signal = (((factor>=1-quantile).replace([True,False],[1,np.nan])))
        short_signal = (factor<=(quantile)).replace([True,False],[1,np.nan])

        if limit != 0 and limit != 1:
            long_mask, long_mask_count = get_mask(long_signal,limit)
            short_mask, short_mask_count = get_mask(short_signal,limit)
        else:
            long_mask,long_mask_count,short_mask,short_mask_count = long_signal,long_signal,short_signal,short_signal
            
        long_port_df = (ret_df_t*long_mask)
        long_port = long_port_df.mean()
        short_port = (ret_df_t*short_mask).mean()
        return_df = pd.DataFrame([index_port,long_port,short_port],index = ['Index','Long','Short']).T
        return_df['IC'] = list(IC)
        return_df['Num'] = list(long_mask.count())
        return_dict['Real'] = return_df
        return_dict['Mask']= long_mask_count
        if draw:
            draw_real(return_dict['Real'],fac_name)
    #（2）分组统计：
    if 'Group' in cal or 'Rank' in cal:
        steps = np.arange(0,1,0.1)
        group_name = ['Index'] + ['Group '+str(i+1) for i in range(0,10)]
        level_return_list = []
        for i in range(0,len(steps)):
            if i==len(steps)-1:
                sel_mask = (factor>=factor.quantile(steps[i]))
            else:
                sel_mask = ((factor>=factor.quantile(steps[i])) & (factor<factor.quantile(steps[i+1])))
            ret_temp = ret_df_t[sel_mask]
            fac_temp = factor[sel_mask]
            level_return_list.append(ret_temp.mean())
        level_return_list.append(ret_df_t.mean())
        level_return_list.reverse()
        level_return_df = pd.DataFrame(level_return_list,index = group_name)
        return_dict['Group'] = level_return_df.T
        if draw and 'Group' in cal:
            draw_group(return_dict['Group'],fac_name)
    #（3）分组rank统计：
    if 'Rank' in cal:
        level_rank_df = level_return_df.T.resample('3M').mean().rank(axis=1,pct = True)#-(level_return_df.min()/2).T#.plot(figsize=(15,3))
        level_rank_df = level_rank_df-level_rank_df.min()/2
        cor_mat = pd.DataFrame(np.array([-np.arange(level_rank_df.shape[1])]
                             ).repeat(level_rank_df.shape[0],axis=0),index = level_rank_df.index
                    ,columns = level_rank_df.columns)
        cor_series = (level_rank_df.corrwith(cor_mat,axis=1,method = 'spearman'))
        level_rank_df['Cor'] = cor_series
        return_dict['Rank'] = level_rank_df
        if draw:
            draw_group_rank(return_dict['Rank'],fac_name)
    
    return return_dict


# In[3402]:


return_dict = backtest_cdb(factor,ret_df_t, quantile = 0.02,limit = 5,
                 date_range=['20180101','20200101'],fac_name='Factor',
                             cal = ['Real'],draw=True)


# In[ ]:


return_dict['Mask'].max().plot()


# # 反转策略

# In[ ]:


main_fac_5_20 = (Close_price_df.rolling(axis=1,window=5,min_periods = 1).mean())/(Close_price_df.rolling(axis=1,window=20,min_periods = 1).mean())
fac_5_20 = -uni(main_fac_5_20)
f,r,n = (fac_5_20*clean_mask).shift(1,axis=1),change_pct_long/100,'MA5_MA20'
rdf_5_20 = backtest_cdb(f,r, quantile = 0.02,limit = 5,
                 date_range=['20180101','20230101'],fac_name=n,
                             cal = ['Real','Group','Rank'],draw=True)


# In[ ]:


