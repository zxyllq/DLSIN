import os
from datetime import datetime
import pandas as pd
import numpy as np
import pdb 
from tqdm import tqdm
from multiprocessing import Process
#logger = logging.getLogger()
from collections import Counter
from math import log

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
pre_session_file= 'pre_sessions'
pre_sequence_file= 'pre_sequence'
file_main_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
session_file  = os.path.join(file_main_path,   pre_session_file)
sequence_file  = os.path.join(file_main_path,   pre_sequence_file)
cpu_num =10 
cut_length=1
#2.输出文件
#user sessions_diversity  session_length sequences_diversity sequence_length session_num 
diversity_file = os.path.join(file_main_path,r'diversity_length_num')

#3.读取文件 
with open(session_file , "r") as f:
    session_lines = f.readlines()[1:]#第一行是col_names
with open(sequence_file , "r") as f:
    sequence_lines = f.readlines()[1:]

process =[]
each_process_batch  = (len( session_lines ) -1)//cpu_num+1
for i in range(0, cpu_num):
    #批量处理
    session_lines_batch = session_lines[i * each_process_batch:(i + 1) * each_process_batch]
    sequence_lines_batch = sequence_lines[i * each_process_batch:(i + 1) * each_process_batch]
    #
    process.append(Process(target = diversity_length_num_statistic,args =(i,diversity_file,session_lines_batch,sequence_lines_batch ,cut_length) )  ) 

    #diversity_statistic(i,diversity_file,session_lines_batch,sequence_lines_batch ,cut_length)

[p.start() for p in process] 
[p.join() for p in process] 

f_output= open(diversity_file, "w")   
for i in tqdm(range(0,cpu_num)):
    f_batch = open(diversity_file+'_'+str(i)).read()  
    f_output.write(f_batch) 
for i in range(0,cpu_num):
    os.remove (diversity_file+'_'+str(i))

f_output.close()   


              










