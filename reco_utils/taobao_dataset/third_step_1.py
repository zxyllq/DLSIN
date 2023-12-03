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
 
import random
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

def diversity_length_num_statistic(batch_num,user_batch,reviews_batch,session_all_interval_file,session_interval ,cut_length):
    fo=open(os.path.join(session_all_interval_file,  'sample_diversity_'+session_interval)+'_'+str(batch_num), "w")
     
    for user in tqdm(  user_batch):
        user_interactions =  reviews_batch[ reviews_batch.uid==user]
        sequence_cate_list= user_interactions['cid'].to_list()
        if len(sequence_cate_list )<=cut_length:
            continue 
        
         

         
        #计算每个session的diversity
        coverage_diversity_list =[]
        entropy_diversity_list  =[]
        gini_diversity_list =[]
        session_length_list=[]
         
        

        session_group = user_interactions.groupby("session_state_"+session_interval )
        session_sample=[]
        session_list =list(np.arange(1, len(session_group)+1, 1))
        while len(session_sample)<2 and len(session_list)>0:
            session_i=random.choice(session_list)
            session_list.remove(session_i)
            if  len( session_group.groups[session_i].to_list())<=5 or session_i in session_sample :
                continue
            else:
                session_sample.append(session_i)
               

        if len( session_sample) <2:
            continue
        session_both_cate_list=[]
        for session_index ,session_df in session_group:
            if session_index in session_sample:

             
                session_cate_list = session_df['cid'].to_list()
                session_both_cate_list.extend( session_cate_list)
                coverage_diversity_list.append(len(set(session_cate_list))/len(session_cate_list ))
                key,  wealth_tuple = zip(*Counter(session_cate_list ).items()) 
                entropy_diversity_list.append( entropy_coef(wealth_tuple,len(session_cate_list)) )
                gini_diversity_list.append(gini_coef(wealth_tuple) )
                session_length_list.append(len(session_cate_list))
            else:
                continue
        coverage_diversity_list.append(len(set(session_both_cate_list))/len(session_both_cate_list ))
        key,  wealth_tuple = zip(*Counter(session_both_cate_list).items()) 
        entropy_diversity_list.append( entropy_coef(wealth_tuple,len(session_both_cate_list)) )
        gini_diversity_list.append(gini_coef(wealth_tuple) )
        session_length_list.append(len(session_both_cate_list))            
       
        
        fo.write(
            str(user)#uid
            +'\t'
            +str(coverage_diversity_list)
            +'\t'
            +str(entropy_diversity_list)
            +'\t'
            +str(gini_diversity_list)
            +'\t'
            +str(session_length_list)
           

            +'\n'
        )
  

#1.基本信息
dataset = "taobao"
reviews_name = 'new_UserBehavior_with_session_state'#忽略session_state即可用
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
reviews_file = os.path.join(reviews_path,   reviews_name)
session_interval = '360'  
cpu_num =100
cut_length=1

#2.输出文件
#不管是sequence还是session，还是session_w_succ都基于这个文件
session_all_interval_file = os.path.join(reviews_path,'session_all_interval_information')
os.makedirs(session_all_interval_file, mode = 0o777, exist_ok = True)
reviews_file_session_diversity = os.path.join(session_all_interval_file,   session_interval  )

 
 
 
#3. 读取主要文件

reviews = pd.read_csv(reviews_file,  sep='\t' )[['ref_index','uid', 'iid', 'cid', "behavior_type", 'ts','state',"session_state_"+session_interval]]
#names=[ref_index	uid	iid	cid	ts	behavior_type	session_state_30	session_state_90	session_state_120	session_state_150	session_state_180	session_state_210	session_state_240	session_state_270	session_state_300	session_state_330	session_state_360	session_state_390	session_state_420	session_state_450	session_state_480	session_state_510	session_state_540	session_state_600	session_state_720	session_state_960	state ])
 
 
 
 

#4  多线程处理计算diversity
user_all =reviews.uid.unique()
process =[]
each_process_batch  = (len( user_all) -1)//cpu_num+1
pdb.set_trace()
for i in tqdm( range(0,cpu_num)):
    user_batch = user_all[i * each_process_batch:(i + 1) * each_process_batch]
    reviews_batch = reviews.loc[ reviews.uid.isin(user_batch)]#以用户为单位进行处理

    #diversity_length_num_statistic(i,user_batch,reviews_batch,session_all_interval_file,session_interval,cut_length)
    process.append(Process(target =  diversity_length_num_statistic,args =(i,user_batch,reviews_batch,session_all_interval_file,session_interval,cut_length) )  ) 
[p.start() for p in process] 
[p.join() for p in process] 
pdb.set_trace()
#合并文件
 
fo = open(os.path.join(session_all_interval_file,  'sample_diversity_'+session_interval) , "w")            

for i in tqdm(range(0,cpu_num)):
     

    fo_batch = open(os.path.join(session_all_interval_file,  'sample_diversity_'+session_interval) +'_'+str(i)).read()  
    fo.write(fo_batch )  
                      
 
 
for i in range(0,cpu_num):
    
    os.remove (os.path.join(session_all_interval_file,  'sample_diversity_'+session_interval)+'_'+str(i))

diversity_data= pd.read_csv(os.path.join(session_all_interval_file,  'sample_diversity_'+session_interval),sep='\t',names=['uid','coverage','entropy','gini','sess_length' ])
coverage_dict={'sess_1':[],'sess_2':[],'sess_both':[]}
entropy_dict={'sess_1':[],'sess_2':[],'sess_both':[]}
gini_dict={'sess_1':[],'sess_2':[],'sess_both':[]}
for i in tqdm(range(0,len(diversity_data))):
    coverage =diversity_data.loc[i]['coverage'].strip('[').strip(']').split(', ')
    coverage_dict['sess_1'].append(float(coverage[0]))
    coverage_dict['sess_2'].append(float(coverage[1]))
    coverage_dict['sess_both'].append(float(coverage[2]))  

    entropy =diversity_data.loc[i]['entropy'].strip('[').strip(']').split(', ')
    entropy_dict['sess_1'].append(float(entropy[0]))
    entropy_dict['sess_2'].append(float(entropy[1]))
    entropy_dict['sess_both'].append(float(entropy[2])) 

    gini =diversity_data.loc[i]['gini'].strip('[').strip(']').split(', ')
    gini_dict['sess_1'].append(float(gini[0]))
    gini_dict['sess_2'].append(float(gini[1]))
    gini_dict['sess_both'].append(float(gini[2]))        
#[0.4450309722147521, 0.45321548310900756, 0.37973531025088336]
coverage_mean=[np.average(coverage_dict['sess_1']),np.average(coverage_dict['sess_2']),np.average(coverage_dict['sess_both'])]
#[0.2207001079206959, 0.2228945395500136, 0.16914498016183624]
coverage_std=[np.std(coverage_dict['sess_1']),np.std(coverage_dict['sess_2']),np.std(coverage_dict['sess_both'])]
#[1.3245255426214295, 1.3395353899901365, 1.838249277475239]       
entropy_mean=[np.average(entropy_dict['sess_1']),np.average(entropy_dict['sess_2']),np.average(entropy_dict['sess_both'])]
#[0.636520303335991, 0.6327220086866022, 0.5641034910072434]
entropy_std=[np.std(entropy_dict['sess_1']),np.std(entropy_dict['sess_2']),np.std(entropy_dict['sess_both'])]
#[0.7067773305631063, 0.7078998505552326, 0.6031399286842964]
gini_mean=[np.average(gini_dict['sess_1']),np.average(gini_dict['sess_2']),np.average(gini_dict['sess_both'])]
#[0.1489652716560626, 0.149039129832168, 0.1209484914465856]
gini_std=[np.std(gini_dict['sess_1']),np.std(gini_dict['sess_2']),np.std(gini_dict['sess_both'])]



 
 
 
                





    
 
 
 
 
 











