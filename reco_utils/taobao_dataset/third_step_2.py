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
 

#1.基本信息
dataset = "taobao"
 
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
session_interval = '360'  
cpu_num =100
cut_length=1
pdb.set_trace()
#'../../tests/resources/deeprec/sequential/taobao/session_all_interval_information'
session_all_interval_file = os.path.join(reviews_path,'session_all_interval_information')
os.makedirs(session_all_interval_file, mode = 0o777, exist_ok = True)
#'../../tests/resources/deeprec/sequential/taobao/session_all_interval_information/360'
reviews_file_session_diversity = os.path.join(session_all_interval_file,   session_interval  )


#2.输出文件
#session_diversity_excel= os.path.join(reviews_path,'session_all_interval_information')
 
 
#3. 读取主要文件
 
diversity_data= pd.read_csv(os.path.join(session_all_interval_file,  'sample_diversity_'+session_interval),sep='\t',names=['uid','coverage','entropy','gini','sess_length' ])

 
entropy_dict={'entropy_1':[],'entropy_2':[],'entropy_both':[]}
gini_dict={'gini_1':[],'gini_2':[],'gini_both':[]}
for i in tqdm(range(0,len(diversity_data))):
    
    entropy =diversity_data.loc[i]['entropy'].strip('[').strip(']').split(', ')
    entropy_dict['entropy_1'].append(float(entropy[0]))
    entropy_dict['entropy_2'].append(float(entropy[1]))
    entropy_dict['entropy_both'].append(float(entropy[2])) 

    gini =diversity_data.loc[i]['gini'].strip('[').strip(']').split(', ')
    gini_dict['gini_1'].append(float(gini[0]))
    gini_dict['gini_2'].append(float(gini[1]))
    gini_dict['gini_both'].append(float(gini[2]))   

pdb.set_trace()    
#合并字典 
gini_dict.update(entropy_dict)
#转dataframe
diverse_df = pd.DataFrame.from_dict(gini_dict)
#转excel
diverse_df.to_excel(os.path.join(session_all_interval_file,  'diversity_'+session_interval+'.xlsx') )
 
 
 
                





    
 
 
 
 
 











