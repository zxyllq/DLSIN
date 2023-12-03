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
import shutil
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
def get_item_cate_history_sequence( item_history_words, cate_history_words,itemdict,catedict):

        item_history_sequence =  get_item_history_sequence(item_history_words,itemdict)
        cate_history_sequence =  get_cate_history_sequence(cate_history_words,catedict)
        return str(item_history_sequence), str(cate_history_sequence)

def get_item_history_sequence( item_history_words,itemdict):

    item_history_sequence = []
    for item in item_history_words:
        item_history_sequence.append(
            itemdict[int(item)] if int(item) in  itemdict else 0
        )
    return item_history_sequence

def get_cate_history_sequence( cate_history_words,catedict):

    cate_history_sequence = []
    for cate in cate_history_words:
        cate_history_sequence.append(
            catedict[int(cate)] if int(cate) in  catedict else 0
        )
    return cate_history_sequence
 
def sequence_data_wno_neg_process(batch_num,lines_batch, output_file,sequence_length,userdict,itemdict,catedict):
    output_data = open(output_file+'_'+str(batch_num),'w' )
    for line in lines_batch:
        words=line.strip().split('\t')
        #截断
        words[1] = str(userdict[int(words[1])]) if int(words[1]) in userdict else '0'
        words[2] = str(itemdict[int(words[2])])  if int(words[2]) in itemdict else '0'
        words[3] = str(catedict [int(words[3])]) if int(words[3]) in catedict else '0'

        words[-1] = str(words[-1].strip('[').strip(']').split(", ")[-sequence_length:] ) 
        words[-2] = str(words[-2].strip('[').strip(']').split(", ")[-sequence_length:] ) 
        words[-3] =  words[-3].strip('[').strip(']').split(", ")[-sequence_length:] 
        words[-4] =  words[-4].strip('[').strip(']').split(", ")[-sequence_length:] 
        words[-4],words[-3]  =  get_item_cate_history_sequence(words[-4],words[-3]  ,itemdict,catedict )
        del words[6:10] #去除succ
        
        output_data.write("\t".join(words) +"\n")
def sequence_data_process(original_file,output_file,cpu_num,sequence_length,userdict,itemdict,catedict):
    sequence_data = open(output_file,'a+' )
    with open(original_file , "r") as f:
        lines = f.readlines()
    process =[]

    each_process_batch  = (len(lines ) -1)//cpu_num+1
    for i in tqdm(range(0,cpu_num)):
       
        lines_batch= lines[i * each_process_batch:(i + 1) * each_process_batch]    
        #sequence_data_process(i,lines_batch, train_sequence_data_file,userdict,itemdict,catedict)
        process.append(Process(target = sequence_data_wno_neg_process,args =(i,lines_batch, output_file,sequence_length,userdict,itemdict,catedict) )  ) 
            
    [p.start() for p in process] 
    [p.join() for p in process] 

    for i in tqdm(range(0,cpu_num)):
            f_batch = open(output_file+'_'+str(i)).read()  
            sequence_data.write(f_batch) 
    for i in range(0,cpu_num):
        os.remove (output_file+'_'+str(i))

    sequence_data.close()

def sequence_data_add_negpos_label(reference_file1,reference_file2,outputfile,negtive_ratio):
     
    positive_and_negtive_iid = pd.read_csv(reference_file1,sep='\t',header=None)[3]
    positive_and_negtive_cid = pd.read_csv(reference_file1,sep='\t',header=None)[4]
    sequence_data =pd.read_csv(reference_file2,sep='\t',header=None)
    negtive_sample_sequence_data = pd.DataFrame(np.repeat(sequence_data.values,negtive_ratio+1 ,axis=0))
    negtive_sample_sequence_data[2]=positive_and_negtive_iid 
    negtive_sample_sequence_data[3]=positive_and_negtive_cid
    negtive_sample_sequence_data.insert(0, 'label',  pd.read_csv(reference_file1,sep='\t',header=None)[0])

    pdb.set_trace()
    negtive_sample_sequence_data.to_csv(outputfile,sep='\t',header=None,index=False) 
        




#1.基本信息
dataset = "taobao"
reviews_name = 'new_UserBehavior'
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
valid_negtive_sample= 4
test_negtive_sample = 99 
session_LS_data_file =  os.path.join(reviews_path, "90min" , "session_LS")
session_data_file =  os.path.join(reviews_path, "90min" ,  "session")
sequence_data_file =  os.path.join(reviews_path, "90min" ,  "sequence") 
os.makedirs(session_LS_data_file, mode = 0o777, exist_ok = True)
os.makedirs(session_data_file, mode = 0o777, exist_ok = True)
os.makedirs(sequence_data_file, mode = 0o777, exist_ok = True)



original_train_file = os.path.join(reviews_path, r'train_data_w_succ')
original_sampled_valid_file = os.path.join(reviews_path, r'sampled_valid_data_w_succ')
original_sampled_test_file = os.path.join(reviews_path, r'sampled_test_data_w_succ')

train_session_LS_data_file = os.path.join(session_LS_data_file, r'train_data')
valid_session_LS_data_file = os.path.join(session_LS_data_file, r'valid_data')
test_session_LS_data_file = os.path.join(session_LS_data_file, r'test_data')
negtive_sample_session_LS_valid_data_file = os.path.join(session_LS_data_file, r'negtive_sample_valid_data')
negtive_sample_session_LS_test_data_file = os.path.join(session_LS_data_file, r'negtive_sample_test_data') 
cpu_num =100
sequence_length=50
user_vocab = os.path.join(reviews_path ,r'user_vocab.pkl')
item_vocab = os.path.join(reviews_path, r'item_vocab.pkl')
cate_vocab = os.path.join(reviews_path,r'category_vocab.pkl')
  
#2.输出文件

train_session_data_file = os.path.join(session_data_file, r'train_data')
valid_session_data_file = os.path.join(session_data_file, r'valid_data')
test_session_data_file = os.path.join(session_data_file, r'test_data')
negtive_sample_valid_session_data_file = os.path.join(session_data_file, r'negtive_sample_valid_data')
negtive_sample_test_session_data_file = os.path.join(session_data_file, r'negtive_sample_test_data')

train_sequence_data_file = os.path.join(sequence_data_file, r'train_data')
valid_sequence_data_file = os.path.join(sequence_data_file, r'valid_data')
test_sequence_data_file = os.path.join(sequence_data_file, r'test_data')
negtive_sample_valid_sequence_data_file = os.path.join(sequence_data_file, r'negtive_sample_valid_data')
negtive_sample_test_sequence_data_file = os.path.join(sequence_data_file, r'negtive_sample_test_data')


#3.读取文件 
 
userdict,  itemdict,  catedict = (
            load_dict(user_vocab),
            load_dict(item_vocab),
            load_dict(cate_vocab),
        )
 
#4. 先处理session_data,直接去除session_LS_data
train_session_data = pd.read_csv(train_session_LS_data_file,sep='\t',names=['index','uid','iid','cid','behavior','session_state', 
                                                                            'current_session_iid_succ','current_session_cid_succ','current_session_behavior_succ','current_session_session_state_succ',  
                                                                            'current_session_iid_pre','current_session_cid_pre','current_session_behavior_pre','current_session_session_state_pre' , 
                                                                            'LS_session-1_iid','LS_session-1_cid','LS_session-1_behavior','LS_session-1_session_state' ,  
                                                                            'LS_session-2_iid','LS_session-2_cid','LS_session-2_behavior','LS_session-2_session_state' ,   
                                                                            'LS_session-3_iid','LS_session-3_cid','LS_session-3_behavior','LS_session-3_session_state' ,   
                                                                            'LS_session-4_iid','LS_session-4_cid','LS_session-4_behavior','LS_session-4_session_state' ,  "valid_LT" ]) 
train_session_data=train_session_data.drop(columns=['current_session_iid_succ','current_session_cid_succ','current_session_behavior_succ','current_session_session_state_succ']  )
train_session_data.to_csv(train_session_data_file ,sep='\t',header=None)
valid_session_data = pd.read_csv(valid_session_LS_data_file,sep='\t',names=['index','uid','iid','cid','behavior','session_state', 
                                                                            'current_session_iid_succ','current_session_cid_succ','current_session_behavior_succ','current_session_session_state_succ',  
                                                                            'current_session_iid_pre','current_session_cid_pre','current_session_behavior_pre','current_session_session_state_pre' , 
                                                                            'LS_session-1_iid','LS_session-1_cid','LS_session-1_behavior','LS_session-1_session_state' ,  
                                                                            'LS_session-2_iid','LS_session-2_cid','LS_session-2_behavior','LS_session-2_session_state' ,   
                                                                            'LS_session-3_iid','LS_session-3_cid','LS_session-3_behavior','LS_session-3_session_state' ,   
                                                                            'LS_session-4_iid','LS_session-4_cid','LS_session-4_behavior','LS_session-4_session_state' ,  "valid_LT" ]) 
valid_session_data=valid_session_data.drop(columns=['current_session_iid_succ','current_session_cid_succ','current_session_behavior_succ','current_session_session_state_succ']  )
valid_session_data.to_csv(valid_session_data_file ,sep='\t',header=None) 
test_session_data = pd.read_csv(test_session_LS_data_file,sep='\t',names=['index','uid','iid','cid','behavior','session_state', 
                                                                            'current_session_iid_succ','current_session_cid_succ','current_session_behavior_succ','current_session_session_state_succ',  
                                                                            'current_session_iid_pre','current_session_cid_pre','current_session_behavior_pre','current_session_session_state_pre' , 
                                                                            'LS_session-1_iid','LS_session-1_cid','LS_session-1_behavior','LS_session-1_session_state' ,  
                                                                            'LS_session-2_iid','LS_session-2_cid','LS_session-2_behavior','LS_session-2_session_state' ,   
                                                                            'LS_session-3_iid','LS_session-3_cid','LS_session-3_behavior','LS_session-3_session_state' ,   
                                                                            'LS_session-4_iid','LS_session-4_cid','LS_session-4_behavior','LS_session-4_session_state' ,  "valid_LT" ]) 
test_session_data=test_session_data.drop(columns=['current_session_iid_succ','current_session_cid_succ','current_session_behavior_succ','current_session_session_state_succ']  )
test_session_data.to_csv(test_session_data_file ,sep='\t',header=None)
# #负采样后数据的直接文件copy就好
shutil.copy( negtive_sample_session_LS_valid_data_file, negtive_sample_valid_session_data_file)
shutil.copy( negtive_sample_session_LS_test_data_file, negtive_sample_test_session_data_file)

#5.处理 sequence 文件
pdb.set_trace()
original_train_data = pd.read_csv(original_train_file,sep='\t',names=['index','uid','iid','cid','behavior','session_state','current_session_iid_succ','current_session_cid_succ','current_session_behavior_succ','current_session_session_state_succ', 'pre_sequence_iid','pre_sequence_cid','pre_sequence_behavior','pre_sequence_session_state' ])

sequence_data_process(original_train_file,train_sequence_data_file,cpu_num,sequence_length,userdict,itemdict,catedict)

sequence_data_process(original_sampled_valid_file ,valid_sequence_data_file,cpu_num,sequence_length,userdict,itemdict,catedict)
sequence_data_process(original_sampled_test_file,test_sequence_data_file,cpu_num,sequence_length,userdict,itemdict,catedict)

sequence_data_add_negpos_label(negtive_sample_session_LS_valid_data_file ,valid_sequence_data_file ,negtive_sample_valid_sequence_data_file ,valid_negtive_sample)
sequence_data_add_negpos_label(negtive_sample_session_LS_test_data_file ,test_sequence_data_file ,negtive_sample_test_sequence_data_file ,test_negtive_sample)
 



    



    
      
            
 


                 

    
             

                  
                  
                 

            











