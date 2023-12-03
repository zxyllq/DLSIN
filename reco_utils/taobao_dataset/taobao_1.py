import os
from datetime import datetime
import pandas as pd
import numpy as np
import pdb 
from tqdm import tqdm
from multiprocessing import Process
import _pickle as cPickle
#logger = logging.getLogger()

def base_dataset_process(batch_num,reviews_batch,user_batch,train_file,valid_file,test_file):
    f_train = open(train_file+'_'+str(batch_num), "w")
    f_valid = open(valid_file+'_'+str(batch_num), "w")
    f_test = open(test_file+'_'+str(batch_num), "w")
    for user in tqdm(user_batch):
        user_interactions=reviews_batch[ reviews_batch["uid"]==user]
        


        for target_row in  user_interactions.index :
            #target
            target_state =  user_interactions["state"].loc[target_row]
            target_iid = user_interactions["iid"].loc[target_row]
            target_cid = user_interactions["cid"].loc[target_row]
            target_behavior =   user_interactions["behavior_type"].loc[target_row]
            target_session_state = user_interactions["session_state"].loc[target_row]
            #current_session_succ
            current_session_succ = user_interactions[user_interactions["session_state"]== target_session_state ].loc[target_row+1:] #+1表示自target之后
            if  current_session_succ.empty ==True :
             
                #target是当前session的最后一个，取下一个session作为successor，为避免数据泄露，需保证下一个session与当前target同属于一个state
                current_session_succ = user_interactions[user_interactions["session_state"]== target_session_state+1 ][user_interactions["state"]== target_state]  
            #如果仍为空，就是[] 
            if current_session_succ.empty ==True :
       
                current_session_iid_succ = [0]
                current_session_cid_succ =  [0]
                current_session_behavior_succ = [0]
                current_session_session_state_succ = [0]
            else:
                current_session_iid_succ = current_session_succ["iid"].to_list()
                current_session_cid_succ =  current_session_succ["cid"].to_list()
                current_session_behavior_succ = current_session_succ["behavior_type"].to_list()
                current_session_session_state_succ = current_session_succ["session_state"].to_list()
            #pre_sequence
            pre_sequence = user_interactions.loc[:target_row-1] 
            if   pre_sequence.empty ==True :
                #无pre，那么就结束本次循环进入下一个target
              
                continue
            pre_sequence_iid = pre_sequence["iid"].to_list()
            pre_sequence_cid= pre_sequence["cid"].to_list()
            pre_sequence_behavior= pre_sequence["behavior_type"].to_list()
            pre_sequence_session_state= pre_sequence["session_state"].to_list()
            if target_state == 0:
                fo = f_train
            elif target_state == 1:
                fo = f_valid
            elif target_state == 2:
                fo = f_test
            fo.write(
                        str(target_row)#将train/valid/test 的index与 原reviews_file对齐
                        + "\t"
                        + str(user)
                        + "\t"
                        + str(target_iid)
                        + "\t"
                        + str(target_cid)
                        + "\t"
                        + str(target_behavior)
                        + "\t"
                        + str(target_session_state)
                        + "\t"
                        + str(current_session_iid_succ)
                        + "\t"
                        + str(current_session_cid_succ)
                        + "\t"
                        + str(current_session_behavior_succ)
                        + "\t"
                        + str(current_session_session_state_succ)
                        + "\t"
                        + str(pre_sequence_iid)
                        + "\t"
                        + str(pre_sequence_cid)
                        + "\t"
                        + str(pre_sequence_behavior)
                        + "\t"
                        + str(pre_sequence_session_state)
                        + "\n"
                    )
    f_train.close()
    f_valid.close()
    f_test.close()
            
                



#1.基本信息
dataset = "taobao"
reviews_name = 'new_UserBehavior'
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
reviews_file = os.path.join(reviews_path,   reviews_name)
 
cpu_num =100
 
sample_probability=0.2
#2.输出文件
#不管是sequence还是session，还是session_w_succ都基于这个文件
train_file = os.path.join(reviews_path, r'train_data_w_succ')
valid_file = os.path.join(reviews_path, r'valid_data_w_succ')
test_file = os.path.join(reviews_path, r'test_data_w_succ')
sampled_valid_file = os.path.join(reviews_path, r'sampled_valid_data_w_succ')
sampled_test_file = os.path.join(reviews_path, r'sampled_test_data_w_succ')
user_vocab = os.path.join(reviews_path ,r'user_vocab.pkl')
item_vocab = os.path.join(reviews_path, r'item_vocab.pkl')
cate_vocab = os.path.join(reviews_path,r'category_vocab.pkl')

#3. 制作用户history 和vocab
reviews = pd.read_csv(reviews_file,  sep='\t' )#names=['uid', 'iid', 'cid',   'ts' ,	"behavior_type","session_state"	, "state" ])
#uid ,target_iid,target_cid,target_behavior, target_session_state,
# current_session_iid_succ,current_session_cid_succ,current_session_behavior_succ,current_session_session_state_succ
# pre_sequence_iid,pre_sequence_cid,pre_sequence_behavior,pre_sequence_session_state
#3.1 制作vocab
#  
train_reviews=reviews[reviews.state==0]
 
qiqi=pd.Series(np.arange(1,len(train_reviews.uid.value_counts())+1))
qiqi.index = train_reviews.uid.value_counts().index
user_voc = qiqi.to_dict()
user_voc["default"]=0
qiqi=pd.Series(np.arange(1,len(train_reviews.iid.value_counts())+1))
qiqi.index = train_reviews.iid.value_counts().index
item_voc  = qiqi.to_dict()
item_voc["default"]=0
qiqi=pd.Series(np.arange(1,len(train_reviews.cid.value_counts())+1))
qiqi.index = train_reviews.cid.value_counts().index
cate_voc  = qiqi.to_dict()
cate_voc["default"]=0
cPickle.dump(user_voc, open(user_vocab, "wb"))
cPickle.dump(item_voc, open(item_vocab, "wb"))
cPickle.dump(cate_voc, open(cate_vocab, "wb"))
 
f_train = open(train_file, 'a+')
f_valid = open(valid_file, 'a+')
f_test = open(test_file, 'a+')
#3.2  制作history
#多进程处理
user_all = reviews.uid.unique()
process =[]
each_process_batch  = (len(user_all ) -1)//cpu_num+1
for i in range(0, cpu_num):
    user_batch = user_all[i * each_process_batch:(i + 1) * each_process_batch]
    reviews_batch = reviews.loc[ reviews.uid.isin(user_batch)]
    
    #base_dataset_process(i,reviews_batch,user_batch,train_file,valid_file,test_file)
    process.append(Process(target = base_dataset_process,args =(i,reviews_batch,user_batch,train_file,valid_file,test_file) )  ) 
    
[p.start() for p in process] 
[p.join() for p in process] 

process =[]
each_process_batch  = (len(user_all ) -1)//cpu_num+1
for i in tqdm(range(0,cpu_num)):
    f_train_batch = open(train_file+'_'+str(i)).read()  
    f_train.write(f_train_batch)  
    f_valid_batch = open(valid_file+'_'+str(i)).read()  
    f_valid.write(f_valid_batch)  
    f_test_batch = open(test_file+'_'+str(i)).read()  
    f_test.write(f_test_batch)                
     
 
f_train.close()
f_valid.close()
f_test.close()
 
for i in range(0,cpu_num):
    os.remove (train_file+'_'+str(i))
    os.remove (valid_file+'_'+str(i))
    os.remove (test_file+'_'+str(i))

#4.对测试集和验证集采样
 
f_valid = open(valid_file, 'r')
f_test = open(test_file, 'r')
f_valid_sampled = open(sampled_valid_file, 'w')
f_test_sampled = open(sampled_test_file, 'w')
for line in f_valid:
    probability = round(np.random.uniform(0, 1), 1)
    if 0 <= probability <=sample_probability:
        f_valid_sampled.writelines(line)
for line in f_test:
    probability = round(np.random.uniform(0, 1), 1)
    if 0 <= probability <=sample_probability:
        f_test_sampled.writelines(line)
f_valid.close()
f_test.close()
f_valid_sampled.close()
f_test_sampled.close()


            

            





    
 
 
 
 
 











