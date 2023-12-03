import os
from datetime import datetime
import pandas as pd
import numpy as np
import pdb 
from tqdm import tqdm
from multiprocessing import Process
import pickle as pkl
import copy
import random
#logger = logging.getLogger()

def load_dict(filename):
    """Load the vocabularies.

    Args:
        filename (str): Filename of user, item or category vocabulary.

    Returns:
        dict: A saved vocabulary.
    """
    with open(filename, "rb") as f:
        f_pkl = pkl.load(f)
        return f_pkl
def reid_reviews_file_process( batch_num,reviews_batch,reid_reviews_file,userdict,  itemdict,  catedict  ):
    for target_row in  tqdm(reviews_batch.index ):
        reviews_batch["uid"].loc[target_row]=userdict[reviews_batch["uid"].loc[target_row]] if reviews_batch["uid"].loc[target_row] in userdict else 0
        reviews_batch["iid"].loc[target_row]=itemdict[reviews_batch["iid"].loc[target_row]] if reviews_batch["iid"].loc[target_row] in itemdict else 0
        reviews_batch["cid"].loc[target_row]=catedict[reviews_batch["cid"].loc[target_row]] if reviews_batch["cid"].loc[target_row] in catedict else 0
    reviews_batch.to_csv( reid_reviews_file +'_'+str(batch_num), sep='\t', header=True, index=False)

def negtive_sampling(original_file,output_file,reviews,negtive_sample_rate):
    with open(original_file, "r") as f:
        lines = f.readlines()
    negtive_sample_lines   = open(output_file, "w")
    iid_list = reviews["iid"].to_list()
    cid_list = reviews["cid"].to_list()
    for line in  tqdm(lines ) :
        
        words = line.strip().split("\t")
        positive_iid =words[2]
        del words[6:10]#删除succ
        negtive_sample_lines.write("1\t"+"\t".join(words) +"\n")#给加上label
        
        count = 0
        neg_items = set()
        while  count < negtive_sample_rate:
            index= random.choice(reviews.index)
            neg_item = iid_list[index] 
            if  str(neg_item) == positive_iid  or neg_item in neg_items:
                continue
            neg_cate = cid_list[index] 
            count += 1
            neg_items.add(neg_item)
            words[2] = str(neg_item)
            words[3] = str(neg_cate)
            negtive_sample_lines.write("0\t"+"\t".join(words) +"\n")
    


#1.基本信息
dataset = "taobao"
reviews_name = 'new_UserBehavior'
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
reviews_file = os.path.join(reviews_path,   reviews_name)
session_LS_data_file =  os.path.join(reviews_path,   "session_LS")
 
valid_file = os.path.join(session_LS_data_file, r'valid_data')
test_file = os.path.join(session_LS_data_file, r'test_data') 
cpu_num =100
valid_negtive_sample= 4
test_negtive_sample = 99 
user_vocab = os.path.join(reviews_path ,r'user_vocab.pkl')
item_vocab = os.path.join(reviews_path, r'item_vocab.pkl')
cate_vocab = os.path.join(reviews_path,r'category_vocab.pkl')
 
#2.输出文件

negtive_sample_valid_file = os.path.join(session_LS_data_file, r'negtive_sample_valid_data')
negtive_sample_test_file = os.path.join(session_LS_data_file, r'negtive_sample_test_data') 
reid_reviews_file =os.path.join(reviews_path,   'reid'+reviews_name)

#3.读取文件 
userdict,  itemdict,  catedict = (
            load_dict(user_vocab),
            load_dict(item_vocab),
            load_dict(cate_vocab),
        )
 
reviews = pd.read_csv(reviews_file,  sep='\t' )#names=['uid', 'iid', 'cid',   'ts' ,	"behavior_type","session_state"	, "state" ])
#4. reif reviews
process =[]
each_process_batch  = (len(reviews ) -1)//cpu_num+1
for i in range(0, cpu_num):
    
    reviews_batch = reviews.loc[i * each_process_batch:(i + 1) * each_process_batch-1]
    #reid_reviews_file_process( i,reviews_batch,reid_reviews_file,userdict,  itemdict,  catedict  )
    process.append(Process(target = reid_reviews_file_process,args =(i,reviews_batch,reid_reviews_file,userdict,  itemdict,  catedict) )  ) 
[p.start() for p in process] 
[p.join() for p in process] 

reid_reviews = pd.DataFrame(columns= reviews.columns)    
for i in range(0,cpu_num):  
    reviews_batch =  pd.read_csv(reid_reviews_file +'_'+str(i)  ,sep='\t' ) 
     
    reid_reviews=pd.concat([reid_reviews,reviews_batch ],axis=0,ignore_index=True) 
reid_reviews.to_csv(reid_reviews_file,  sep='\t', header=True, index=False)
for i in range(0,cpu_num):
    os.remove (reid_reviews_file +'_' + str(i))
pdb.set_trace()
#5.负采样


negtive_sampling(valid_file,negtive_sample_valid_file , reid_reviews,valid_negtive_sample)
pdb.set_trace()
negtive_sampling(test_file,negtive_sample_test_file , reid_reviews,test_negtive_sample)






    




 
    
      
            
 


                 

    
             

                  
                  
                 

            











