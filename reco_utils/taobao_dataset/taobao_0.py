import os
from datetime import datetime
import pandas as pd
import numpy as np
import pdb 
from tqdm import tqdm
from multiprocessing import Process
#logger = logging.getLogger()



def filter_items_with_multiple_cids(record):

    item_cate = record[['iid', 'cid']].drop_duplicates().groupby('iid').count().reset_index().rename(columns={'cid': 'count'})
    items_with_single_cid = item_cate[item_cate['count'] == 1]['iid']

    record = pd.merge(record, items_with_single_cid, on='iid')

    return record
def downsample(record, col, frac):

    sample_col = record[col].drop_duplicates().sample(frac=frac)

    record = record.merge(sample_col, on=col).reset_index(drop=True)

    return record
def filter_k_core(record, k_core, filtered_column, count_column):

    stat = record[[filtered_column, count_column]] \
            .groupby(filtered_column) \
            .count() \
            .reset_index() \
            .rename(index=str, columns={count_column: 'count'})
    
    stat = stat[stat['count'] >= k_core]

    record = record.merge(stat, on=filtered_column)
    record = record.drop(columns=['count'])

    return record

def manage_seqORses( batch_num,each_process_batch,reviews_batch, sequence_file,sessions_file,session_interval):

    sequence =pd.DataFrame(columns=["uid","iid_list","cid_list",'behavior_list','ts_list'])
    sessions =pd.DataFrame(columns=["uid","iid_session_list","cid_session_list",'behavior_session_list','ts_session_list'])

    for i in   tqdm(  reviews_batch.uid.index.levels[0][batch_num * each_process_batch:(batch_num + 1) * each_process_batch] ):
    #划分session
        reviews_batch.loc[i]['session_state'][reviews_batch.loc[i]['ts'].diff(periods=1)<=session_interval]=0
        reviews_batch.loc[i]['session_state'][reviews_batch.loc[i]['session_state']!=0]=1
        reviews_batch.loc[i]['session_state']= reviews_batch.loc[i]['session_state'].cumsum()
        #划分session
        
        iid_session_list=[]
        cid_session_list=[]
        behavior_session_list=[]
        ts_session_list=[]
        

        for session_num,session_df  in reviews_batch.loc[i].groupby('session_state'):
            iid_session_list.append(session_df['iid'].to_list())
            cid_session_list.append(session_df['cid'].to_list())
            behavior_session_list.append(session_df['behavior_type'].to_list())
            ts_session_list.append(session_df['ts'].to_list())
        #session
        dict_user= {'uid': reviews_batch.loc[i]['uid'].iloc[0] , \
                    'iid_session_list':iid_session_list,\
                    'cid_session_list': cid_session_list,\
                    'behavior_session_list': behavior_session_list,\
                    'ts_session_list': ts_session_list }
        sessions  = sessions.append(dict_user,ignore_index=True)    
        #sequence
        dict_user= {'uid': reviews_batch.loc[i]['uid'].iloc[0] , \
                    'iid_list': reviews_batch.loc[i]['iid'].to_list(),\
                    'cid_list': reviews_batch.loc[i]['cid'].to_list(),\
                    'behavior_list': reviews_batch.loc[i]['behavior_type'].to_list(),\
                    'ts_list': reviews_batch.loc[i]['ts'].to_list() }
        sequence  = sequence.append(dict_user,ignore_index=True)
 
    sequence.to_csv( sequence_file +'_'+str(batch_num), sep='\t', header=True, index=False)
    sessions.to_csv( sessions_file +'_'+str(batch_num) , sep='\t', header=True, index=False)
    
def mark_sessions(batch_num,each_process_batch,reviews_batch,session_interval,reviews_path):
    for i in   tqdm(  reviews_batch.uid.index.levels[0][batch_num * each_process_batch:(batch_num + 1) * each_process_batch ] ):
    #划分session
        reviews_batch.loc[i]['session_state'][reviews_batch.loc[i]['ts'].diff(periods=1)<=session_interval]=0
        reviews_batch.loc[i]['session_state'][reviews_batch.loc[i]['session_state']!=0]=1
        reviews_batch.loc[i]['session_state']= reviews_batch.loc[i]['session_state'].cumsum()
       
        
    reviews_batch.to_csv( reviews_path +'_'+str(batch_num), sep='\t', header=True, index=False)



#1.基本信息
dataset = "taobao"
reviews_name = 'UserBehavior.csv'
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
reviews_file = os.path.join(reviews_path,   reviews_name)
k_core = 10
test_interval = 24*60*60#one day
session_interval = 90*60
cpu_num =100
#2.输出文件
 
sequence_file = os.path.join(reviews_path,r'pre_sequence')
sessions_file = os.path.join(reviews_path,r'pre_sessions')
new_reviews_file = os.path.join(reviews_path,r'new_UserBehavior')

#3.读取文件并过滤

#100150807 rows  其中10434543 是其他行为（fav=2888258 ,cart=5530446 ,buy=2015839）
reviews = pd.read_csv(reviews_file, header=None, names=['uid', 'iid', 'cid', 'behavior', 'ts'])
#3.1 多cate
#1406 个item有多tag ，过滤后record =100088844  rows
reviews = filter_items_with_multiple_cids(reviews)
#3.2 时间
#100033301 rows
start_ts = int(datetime.timestamp(datetime(2017, 11, 25, 0, 0, 0)))
end_ts = int(datetime.timestamp(datetime(2017, 12, 3, 23, 59, 59)))
reviews = reviews[reviews['ts'] >= start_ts]
reviews = reviews[reviews['ts'] <= end_ts]
#3.3 降采样
#5035640 rows
reviews = downsample(reviews, 'uid', 0.05)
#3.4 冷门用户和商品过滤
#2693485 rows没想到过滤了这么多，其他行为有 284705 （其中fav=77978,cart=151771,buy=54956 ）
reviews = filter_k_core(reviews, k_core, 'iid', 'uid')
reviews = filter_k_core(reviews, k_core, 'uid', 'iid')
#4. 统计信息并整理数据
#4.1 统计行为
reviews['behavior_type']=None
reviews['behavior_type'][reviews['behavior']=='pv']=1
reviews['behavior_type'][reviews['behavior']=='cart']=2
reviews['behavior_type'][reviews['behavior']=='fav']=3
reviews['behavior_type'][reviews['behavior']=='buy']=4
reviews.drop('behavior', axis=1, inplace=True)

#uid_list=reviews.uid.unique()
#4.2 针对每个user，对其行为进行排序并整理输出
reviews = reviews.groupby('uid', as_index=False).apply(lambda x: x.sort_values('ts', ascending=True))
 
reviews['session_state']=None 
 
#4.3 整理成pre_sequence or sessions
 
#进程列表
process =[]
each_process_batch  = (len( reviews.uid.index.levels[0]) -1)//cpu_num+1
for i in range(0, cpu_num):
    reviews_batch = reviews.loc[i * each_process_batch:(i + 1) * each_process_batch-1]
    #manage_seqORses(i, each_process_batch, reviews_batch, sequence_file,sessions_file,session_interval)
    process.append(Process(target = manage_seqORses,args =(i, each_process_batch, reviews_batch,sequence_file,sessions_file,session_interval) )  ) 
[p.start() for p in process] 
[p.join() for p in process] 
#合并文件
sequence =pd.DataFrame(columns=["uid","iid_list","cid_list",'behavior_list','ts_list'])
sessions =pd.DataFrame(columns=["uid","iid_session_list","cid_session_list",'behavior_session_list','ts_session_list'])

for i in range(0,cpu_num):  
    sequence_batch =  pd.read_csv(sequence_file +'_' + str(i)  ,sep='\t' ) 
    sessions_batch=   pd.read_csv(sessions_file +'_' + str(i)  ,sep='\t' ) 
        
    sequence=pd.concat([sequence,sequence_batch ],axis=0,ignore_index=True) 
    sessions=pd.concat([sessions, sessions_batch],axis=0,ignore_index=True) 
sequence.to_csv( sequence_file , sep='\t', header=True, index=False)
sessions.to_csv( sessions_file , sep='\t', header=True, index=False)
for i in range(0,cpu_num):
    os.remove (sequence_file +'_' + str(i))
    os.remove (sessions_file +'_' + str(i) )

#4.4 划分训练集，验证集和测试集,标注sessions

#进程列表
process =[]
each_process_batch  = (len( reviews.uid.index.levels[0]) -1)//cpu_num+1
for i in range(0, cpu_num):
    reviews_batch = reviews.loc[i * each_process_batch:(i + 1) * each_process_batch -1]
   # mark_sessions(i,each_process_batch,reviews_batch,session_interval,new_reviews_file) 
    process.append(Process(target = mark_sessions,args =(i,each_process_batch,reviews_batch,session_interval,new_reviews_file) )  ) 

[p.start() for p in process] 
[p.join() for p in process] 
#合并文件
all_reviews = pd.DataFrame(columns= reviews.columns) 
for i in range(0,cpu_num):  
    reviews_batch =  pd.read_csv(new_reviews_file +'_' + str(i)  ,sep='\t' ) 
    if reviews_batch["session_state"].isnull().sum()>0:
        print(i)
        pdb.set_trace()
    all_reviews=pd.concat([all_reviews,reviews_batch ],axis=0,ignore_index=False) 
#划分数据集
#reviews.index = reviews.index.droplevel(0)
all_reviews['state']=0
test_split_time =end_ts  - test_interval
valid_split_time = end_ts - 2*test_interval
all_reviews['state'][all_reviews['ts']>=valid_split_time ] = all_reviews['state'][all_reviews['ts']>=valid_split_time ]+1#369965 其中其他行为占38755 （cart=21278,fav = 10695,buy =6782 ）
all_reviews['state'][all_reviews['ts']>=test_split_time ] = all_reviews['state'][all_reviews['ts']>=test_split_time ]+1#362126 其中其他行为占37035 （cart=20122,fav=10251 ,buy=6662）
all_reviews.to_csv(new_reviews_file,  sep='\t', header=True, index=False)
pdb.set_trace()
for i in range(0,cpu_num):
    os.remove (new_reviews_file +'_' + str(i))
    












