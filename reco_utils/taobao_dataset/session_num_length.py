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

 


#1.基本信息
dataset = "taobao"
pre_session_file= 'pre_sessions'
pre_sequence_file= 'pre_sequence'
file_main_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
session_file  = os.path.join(file_main_path,   pre_session_file)
sequence_file  = os.path.join(file_main_path,   pre_sequence_file)
cpu_num =10 
cut_length=3
#2.输出文件
#user sessions_diversity sequences_diversity  
diversity_file = os.path.join(file_main_path,r'diversity')

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
#     process.append(Process(target = diversity_statistic,args =(i,diversity_file,session_lines_batch,sequence_lines_batch ,cut_length) )  ) 

#     #diversity_statistic(i,diversity_file,session_lines_batch,sequence_lines_batch ,cut_length)

# [p.start() for p in process] 
# [p.join() for p in process] 

# f_output= open(diversity_file, "w")   
# for i in tqdm(range(0,cpu_num)):
#     f_batch = open(diversity_file+'_'+str(i)).read()  
#     f_output.write(f_batch) 
# for i in range(0,cpu_num):
#     os.remove (diversity_file+'_'+str(i))

# f_output.close()   


              










