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

def diversity_length_num_statistic(batch_num,user_batch,reviews_batch,session_all_interval_file,session_interval_list,cut_length):
    fo_dict={}
    for interval in session_interval_list:
        fo_dict[interval] = open(os.path.join(session_all_interval_file,  'diversity_'+interval)+'_'+str(batch_num), "w")

    for user in tqdm(  user_batch):
        user_interactions =  reviews_batch[ reviews_batch.uid==user]
        sequence_cate_list= user_interactions['cid'].to_list()
        if len(sequence_cate_list )<=cut_length:
            continue 
        #计算sequence diversity
        sequence_coverage_diversity = len(set(sequence_cate_list))/len(sequence_cate_list )
        key,  wealth_tuple = zip(*Counter(sequence_cate_list ).items()) 
        sequence_entropy_diversity = entropy_coef(wealth_tuple,len(sequence_cate_list)) 
        sequence_gini_diversity=gini_coef(wealth_tuple) 

        sequence_length = len(sequence_cate_list  ) 

        for interval in session_interval_list:
            #计算每个session的diversity
            coverage_diversity_list =[]
            entropy_diversity_list  =[]
            gini_diversity_list =[]
            session_length_list=[]
            session_num=0
            session_group = user_interactions.groupby("session_state_"+interval)
            for session_index ,session_df in session_group:
                session_num=session_num+1
                session_cate_list = session_df['cid'].to_list()

                coverage_diversity_list.append(len(set(session_cate_list))/len(session_cate_list ))
                key,  wealth_tuple = zip(*Counter(session_cate_list ).items()) 
                entropy_diversity_list.append( entropy_coef(wealth_tuple,len(session_cate_list)) )
                gini_diversity_list.append(gini_coef(wealth_tuple) )
                session_length_list.append(len(session_cate_list))

            coverage_diversity_list.append(sequence_coverage_diversity)
            entropy_diversity_list.append( sequence_entropy_diversity )
            gini_diversity_list.append(sequence_gini_diversity )
            session_length_list.append(sequence_length)
            
            fo_dict[interval].write(
                str(user)#uid
                +'\t'
                +str(coverage_diversity_list)
                +'\t'
                +str(entropy_diversity_list)
                +'\t'
                +str(gini_diversity_list)
                +'\t'
                +str(session_length_list)
                +'\t'
                +str(session_num)

                +'\n'
            )
    for interval in session_interval_list:
        fo_dict[interval].close()


#1.基本信息
dataset = "taobao"
reviews_name = 'new_UserBehavior_with_session_state'#忽略session_state即可用
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
reviews_file = os.path.join(reviews_path,   reviews_name)
session_interval_list=['30','90','120' ,'150','180','210', '240', '270', '300', '330', '360' , '390', '420', '450','480', '510', '540',  '600', '720','960'] 
cpu_num =100
cut_length=1

#2.输出文件
#不管是sequence还是session，还是session_w_succ都基于这个文件
session_all_interval_file = os.path.join(reviews_path,'session_all_interval_information')
os.makedirs(session_all_interval_file, mode = 0o777, exist_ok = True)
reviews_file_all_session_state = os.path.join(session_all_interval_file,   reviews_name )

 
diversity_file=[] 
for interval in session_interval_list:
    diversity_file.append(os.path.join(session_all_interval_file,  'diversity_'+interval))
 
#3. 读取主要文件

reviews = pd.read_csv(reviews_file,  sep='\t' )
#names=[ref_index	uid	iid	cid	ts	behavior_type	session_state_30	session_state_90	session_state_120	session_state_150	session_state_180	session_state_210	session_state_240	session_state_270	session_state_300	session_state_330	session_state_360	session_state_390	session_state_420	session_state_450	session_state_480	session_state_510	session_state_540	session_state_600	session_state_720	session_state_960	state ])
 
 
 
 

#4  多线程处理计算diversity
user_all =reviews.uid.unique()
process =[]
each_process_batch  = (len( user_all) -1)//cpu_num+1
pdb.set_trace()
for i in tqdm( range(0,cpu_num)):
    user_batch = user_all[i * each_process_batch:(i + 1) * each_process_batch]
    reviews_batch = reviews.loc[ reviews.uid.isin(user_batch)]#以用户为单位进行处理

    #diversity_length_num_statistic(i,user_batch,reviews_batch,session_all_interval_file,session_interval_list,cut_length)
    process.append(Process(target =  diversity_length_num_statistic,args =(i,user_batch,reviews_batch,session_all_interval_file,session_interval_list,cut_length) )  ) 
[p.start() for p in process] 
[p.join() for p in process] 
pdb.set_trace()
#合并文件
fo_dict={}
for interval in session_interval_list:
    fo_dict[interval] = open(os.path.join(session_all_interval_file,  'diversity_'+interval) , "w")            

for i in tqdm(range(0,cpu_num)):
    for interval in session_interval_list:

        fo_batch = open(os.path.join(session_all_interval_file,  'diversity_'+interval) +'_'+str(i)).read()  
        fo_dict[interval].write(fo_batch )  
                      
for interval in session_interval_list:
    fo_dict[interval].close()    
 
for i in range(0,cpu_num):
    for interval in session_interval_list:
        os.remove (os.path.join(session_all_interval_file,  'diversity_'+interval)+'_'+str(i))
    
            


        



 
 
 
                





    
 
 
 
 
 











