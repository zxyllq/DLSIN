import os
from datetime import datetime
import pandas as pd
import numpy as np
import pdb 
from tqdm import tqdm
from multiprocessing import Process
import _pickle as cPickle
from collections import Counter
from math import log
#logger = logging.getLogger()
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
def mark_sessions(batch_num,  user_batch,reviews_batch,session_interval ,reviews_path):
 
    for user  in tqdm(user_batch) : 
        ts_df = reviews_batch[reviews_batch.uid==user].ts
        for interval in session_interval:
             
            reviews_batch.loc[ (reviews_batch.uid==user) & (ts_df.diff(periods=1)<=60*float(interval)) ,'session_state_'+interval] =0
            reviews_batch.loc[ (reviews_batch.uid==user) & (reviews_batch['session_state_'+interval]!=0) ,'session_state_'+interval] =1
            reviews_batch.loc[ (reviews_batch.uid==user)   ,'session_state_'+interval ] =reviews_batch.loc[ (reviews_batch.uid==user)   ,'session_state_'+interval ] .cumsum()
    reviews_batch.to_csv( reviews_path +'_'+str(batch_num), sep='\t', header=True, index=False)
def manage_sequence( batch_num,   reviews_batch ,sequence_file ):
    
    
        sequence =pd.DataFrame(columns=["uid","iid_sequence","cid_sequence",'behavior_sequence','ts_sequence'])

        for user, interactions in   tqdm(  reviews_batch.groupby('uid')):
  
            dict_user= {'uid': user , \
                        'iid_sequence':interactions['iid'].to_list(),\
                        'cid_sequence': interactions['cid'].to_list(),\
                        'behavior_sequence':interactions['behavior_type'].to_list(),\
                        'ts_sequence': interactions['ts'].to_list()}
            sequence  = sequence.append(dict_user,ignore_index=True)    
             
         
        sequence.to_csv(  sequence_file +'_'+str(batch_num)  , sep='\t', header=True, index=False)

def manage_ses( batch_num,   reviews_batch ,sessions_file,session_interval):
    
    for interval in session_interval:

        sessions =pd.DataFrame(columns=["uid","iid_session_list","cid_session_list",'behavior_session_list','ts_session_list'])

        for user, interactions in   tqdm(  reviews_batch.groupby('uid')):

   
        
            iid_session_list=[]
            cid_session_list=[]
            behavior_session_list=[]
            ts_session_list=[]
            

            for session_num,session_df  in interactions.groupby('session_state_'+interval):
                iid_session_list.append(session_df['iid'].to_list())
                cid_session_list.append(session_df['cid'].to_list())
                behavior_session_list.append(session_df['behavior_type'].to_list())
                ts_session_list.append(session_df['ts'].to_list())
            #session
            dict_user= {'uid': user , \
                        'iid_session_list':iid_session_list,\
                        'cid_session_list': cid_session_list,\
                        'behavior_session_list': behavior_session_list,\
                        'ts_session_list': ts_session_list }
            sessions  = sessions.append(dict_user,ignore_index=True)    
             
         
        sessions.to_csv( os.path.join(sessions_file,  'session_interval_'+interval+'_'+str(batch_num) ), sep='\t', header=True, index=False)


def gini_coef(wealths):
 
    cum_wealths = np.cumsum(sorted(np.append(wealths, 0)))
    sum_wealths = cum_wealths[-1]
    xarray = np.array(range(0, len(cum_wealths))) / np.float(len(cum_wealths)-1)
    yarray = cum_wealths / sum_wealths
    B = np.trapz(yarray, x=xarray)
    A = 0.5 - B
    return 1- A / (A+B) 
def entropy_coef(wealths,length):
    entropy=0
    for  wealth  in  wealths :
        p_i =  wealth/length
        entropy=-p_i*log(p_i)+entropy
    return entropy

def diversity_length_num_statistic(batch_num,output_file,session_lines_batch,sequence_lines_batch ,cut_length):
    file = open(output_file+'_'+str(batch_num), "w")
    
  
    for session_line ,sequence_line in  tqdm(zip(session_lines_batch,sequence_lines_batch))  :
        #读取cid序列或会话
        
        session_words=session_line.split('\t')
        session_cid_all= session_words[2].strip('[').strip(']').split(', [') 
        sequence_words=sequence_line.split('\t')
        sequence_cid_all = sequence_words[2].strip('[').strip(']').split(', ')
        if len(sequence_cid_all )<=cut_length:
            continue 
        session_length=[]
        session_num =0
        coverage_diversity =[]
        entropy_diversity  =[]
        gini_diversity =[]
        #先处理session
        for index  in range(0,len(session_cid_all)) :#index是session index
            # #针对单个session进行统计
            single_session_cid=session_cid_all[index].strip(']').split(', ')#得到单个session
            #coverage
            if len(single_session_cid  )<=cut_length:
                continue
            session_num=session_num+1
            session_length.append(len(single_session_cid  ))
            coverage_diversity.append(len(set(single_session_cid ))/len(single_session_cid  ))
            key,  wealth_tuple = zip(*Counter(single_session_cid ).items())  
          
            #entropy
            
            entropy_diversity.append(entropy_coef(wealth_tuple,len(single_session_cid)))
            #gini
            
            gini_diversity.append(gini_coef(wealth_tuple))
        #处理sequence
        
         
        coverage_diversity.append(len(set(sequence_cid_all))/len(sequence_cid_all  ))
        
        key,  wealth_tuple = zip(*Counter(sequence_cid_all ).items()) 
        #entropy
            
        entropy_diversity.append(entropy_coef(wealth_tuple,len(sequence_cid_all)))
        #gini
        
        gini_diversity.append(gini_coef(wealth_tuple)) 
        session_length.append(len(sequence_cid_all  ))

        #写入文件
        file.write(
            session_words[0]#uid
            +'\t'
            +str(coverage_diversity)
            +'\t'
            +str(entropy_diversity)
            +'\t'
            +str(gini_diversity)
            +'\t'
            +str(session_length)
            +'\t'
            +str(session_num)

            +'\n'
        )
    file.close()


#1.基本信息
dataset = "taobao"
reviews_name = 'new_UserBehavior_with_session_state'#忽略session_state即可用
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
reviews_file = os.path.join(reviews_path,   reviews_name)
session_interval=['30','90','180','360','720'] 
cpu_num =100
cut_length=1

#2.输出文件
#不管是sequence还是session，还是session_w_succ都基于这个文件
session_all_interval_file = os.path.join(reviews_path,'session_all_interval')
os.makedirs(session_all_interval_file, mode = 0o777, exist_ok = True)
reviews_file_all_session_state = os.path.join(session_all_interval_file,   reviews_name )

session_interval_file=[]
for interval in session_interval:
    session_interval_file.append(os.path.join(session_all_interval_file,  'session_interval_'+interval))
diversity_file=[] 
for interval in session_interval:
    diversity_file.append(os.path.join(session_all_interval_file,  'diversity_'+interval))
pre_sequence_file=   os.path.join(session_all_interval_file,  'pre_sequence')
#3. 读取主要文件

reviews = pd.read_csv(reviews_file,  sep='\t' )#names=['uid', 'iid', 'cid',   'ts' ,	"behavior_type","session_state"	, "state" ])
#4 整理session_state 
 
 
for interval in session_interval:
    col_name = 'session_state_'+interval
    reviews[col_name]=None 

# #4.1 多线程处理
user_all =reviews.uid.unique()
process =[]
each_process_batch  = (len( user_all) -1)//cpu_num+1
for i in range(0, cpu_num):
   
    user_batch = user_all[i * each_process_batch:(i + 1) * each_process_batch]
    reviews_batch = reviews.loc[ reviews.uid.isin(user_batch)]

    #mark_sessions(i, user_batch,reviews_batch,session_interval ,reviews_path) 
    process.append(Process(target =  mark_sessions,args =(i, user_batch,reviews_batch,session_interval ,reviews_file_all_session_state) )  ) 
[p.start() for p in process] 
[p.join() for p in process] 
 
all_state_reviews = pd.DataFrame(columns= reviews.columns)   
for i in range(0,cpu_num):  
      
    sessions_state_batch=   pd.read_csv(reviews_file_all_session_state +'_' + str(i)  ,sep='\t' ) 

    all_state_reviews =pd.concat([all_state_reviews , sessions_state_batch],axis=0,ignore_index=True) 
all_state_reviews.pop('session_state') 
all_state_reviews.to_csv( reviews_file_all_session_state , sep='\t', header=True, index=False)
 
for i in range(0,cpu_num):
    os.remove (reviews_file_all_session_state +'_' + str(i))

#5  根据session state 生成各个interval 的文件
#uid	iid	cid	ts	behavior_type	state	session_state_30	session_state_60	session_state_90	session_state_120	session_state_150	session_state_180	session_state_210	session_state_240	session_state_270	session_state_300
all_state_reviews = pd.read_csv(reviews_file_all_session_state,sep='\t')
process =[]
each_process_batch  = (len( user_all) -1)//cpu_num+1
for i in range(0, cpu_num):
 
    user_batch =user_all[i * each_process_batch:(i + 1) * each_process_batch]
    all_state_reviews_batch = all_state_reviews.loc[all_state_reviews.uid.isin(user_batch)]
    #manage_ses( i, all_state_reviews_batch ,session_all_interval_file,session_interval)
    process.append(Process(target =manage_ses,args =(i,  all_state_reviews_batch ,session_all_interval_file,session_interval) )  ) 
[p.start() for p in process] 
[p.join() for p in process] 

for interval in session_interval:
    sessions =pd.DataFrame(columns=["uid","iid_session_list","cid_session_list",'behavior_session_list','ts_session_list'])
    for i in tqdm(range(0,cpu_num)):  
        
        sessions_batch=   pd.read_csv(os.path.join(session_all_interval_file,  'session_interval_'+interval+'_'+str(i) )  ,sep='\t' ) 
            
        
        sessions=pd.concat([sessions, sessions_batch],axis=0,ignore_index=True)
    sessions.to_csv(os.path.join(session_all_interval_file,  'session_interval_'+interval) , sep='\t', header=True, index=False)
    for i in range(0,cpu_num):
         
        os.remove (os.path.join(session_all_interval_file,  'session_interval_'+interval+'_'+str(i) ) )
# #6 计算diversity
all_state_reviews = pd.read_csv(reviews_file_all_session_state,sep='\t')
#先整理对应的sequence
process =[]
each_process_batch  = (len( user_all ) -1)//cpu_num+1
for i in range(0, cpu_num):
    user_batch = user_all[i * each_process_batch:(i + 1) * each_process_batch]
    reviews_batch = reviews.loc[ reviews.uid.isin(user_batch)]
    #manage_sequence( i,   reviews_batch ,pre_sequence_file )
    process.append(Process(target = manage_sequence,args =(i,   reviews_batch ,pre_sequence_file) )  ) 
[p.start() for p in process] 
[p.join() for p in process] 
sequence =pd.DataFrame(columns=["uid","iid_sequence","cid_sequence",'behavior_sequence','ts_sequence'])
for i in tqdm(range(0,cpu_num)):  

    sequence_batch=   pd.read_csv( pre_sequence_file +'_'+str(i)   ,sep='\t' ) 
    

    sequence=pd.concat([sequence,  sequence_batch],axis=0,ignore_index=True)
sequence.to_csv( pre_sequence_file  , sep='\t', header=True, index=False)
for i in range(0,cpu_num):
    
    os.remove (pre_sequence_file+ '_'+str(i)  )


 



 

for interval in session_interval:
    with open(os.path.join(session_all_interval_file,  'session_interval_'+interval) , "r") as f:
        session_lines = f.readlines()[1:]
    with open(pre_sequence_file , "r") as f:
        sequence_lines = f.readlines()[1:]
    process =[]
    each_process_batch  = (len( session_lines ) -1)//cpu_num+1
    for i in range(0, cpu_num):
        #批量处理
        session_lines_batch = session_lines[i * each_process_batch:(i + 1) * each_process_batch]
        sequence_lines_batch = sequence_lines[i * each_process_batch:(i + 1) * each_process_batch]
        output_file = os.path.join(session_all_interval_file,  'diversity_'+interval)
        #diversity_length_num_statistic(i,output_file,session_lines_batch,sequence_lines_batch ,cut_length) 
        process.append(Process(target = diversity_length_num_statistic,args =(i,output_file,session_lines_batch,sequence_lines_batch ,cut_length) )  ) 
    [p.start() for p in process] 
    [p.join() for p in process] 
    f_output= open(output_file , "w")   
    for i in tqdm(range(0,cpu_num)):
        f_batch = open(output_file +'_'+str(i)).read()  
        f_output.write(f_batch) 
    for i in range(0,cpu_num):
        os.remove (output_file +'_'+str(i))

    f_output.close()  
                





    
 
 
 
 
 











