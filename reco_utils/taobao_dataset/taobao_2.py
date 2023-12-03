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
def get_item_cate_history_sequence( item_history_words, cate_history_words,itemdict,catedict):

        item_history_sequence =  get_item_history_sequence(item_history_words,itemdict)
        cate_history_sequence =  get_cate_history_sequence(cate_history_words,catedict)
        return item_history_sequence, cate_history_sequence

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
 
def current_session_succ_process(words,session_length,reversed_choice,itemdict,catedict):
     #注意：t+1,t+2,t+3....所以截断尾部
    current_session_succ ={'session_iid':[],"session_cid":[],"session_behavior":[],"session_state":[]}
    if len( words[9])==0:
        current_session_succ['session_iid'] = [0] 
        current_session_succ["session_cid"] =  [0] 
        current_session_succ["session_behavior"] = [0] 
        current_session_succ["session_state"] = [0] 
    else:
        
        current_session_succ['session_iid'] = words[6].strip('[').strip(']').split(", ")[:session_length]
        current_session_succ["session_cid"] =  words[7].strip('[').strip(']').split(", ")[:session_length]
        current_session_succ["session_behavior"] =  words[8].strip('[').strip(']').split(", ")[:session_length]
        current_session_succ["session_state"] =  words[9].strip('[').strip(']').split(", ")[:session_length]
        #选择是否反转
        if reversed_choice ==True:
            current_session_succ['session_iid']  = list(reversed(current_session_succ['session_iid'] ))
            current_session_succ["session_cid"] = list(reversed(current_session_succ["session_cid"]))
            current_session_succ["session_behavior"] = list(reversed(current_session_succ["session_behavior"]))
        #reindex   
        current_session_succ['session_iid'], current_session_succ["session_cid"] =  get_item_cate_history_sequence(current_session_succ['session_iid'],current_session_succ["session_cid"],itemdict,catedict )

    return current_session_succ

def current_session_pre_process(words,target_session_state,session_length, itemdict,catedict):
    pre_sequence = {'session_iid':[],"session_cid":[],"session_behavior":[],"session_state":[]}
    pre_sequence['session_iid']=  words[10].strip('[').strip(']').split(", ")
    pre_sequence["session_cid"] = words[11].strip('[').strip(']').split(", ")
    pre_sequence["session_behavior"] = words[12].strip('[').strip(']').split(", ")
    pre_sequence["session_state"] =  words[13].strip('[').strip(']').split(", ")

    # 1. current_session 是一定有的，不然不会收录
    # 如果length <2，也没关系，后面可以用去噪模块去掉
    if target_session_state in pre_sequence["session_state"] :   
        #判断target和上一个session的关系 
        session_state = target_session_state
    else :
        session_state =str (int(target_session_state)-1)
    #注意 0，1，2，...,t-1,t 
    #2.统计last1_session_state在pre_sequence的数目，并截取末尾session_length个数据
    tail_length = pre_sequence["session_state"] .count(session_state)
    current_session_pre ={'session_iid':[],"session_cid":[],"session_behavior":[],"session_state":[]}
    current_session_pre['session_iid'] = pre_sequence['session_iid'][-tail_length:][-session_length:]
    current_session_pre["session_cid"] =pre_sequence["session_cid"][-tail_length:][-session_length:]
    current_session_pre["session_behavior"] =pre_sequence["session_behavior"] [-tail_length:][-session_length:]
    current_session_pre["session_state"] = pre_sequence["session_state"] [-tail_length:][-session_length:]
    #reindex   
    current_session_pre['session_iid'], current_session_pre["session_cid"] =  get_item_cate_history_sequence(current_session_pre['session_iid'],  current_session_pre["session_cid"],itemdict,catedict )
    
    #对pre_sequence截断
    pre_sequence['session_iid']=pre_sequence['session_iid'][:-tail_length]
    pre_sequence["session_cid"]=pre_sequence["session_cid"][:-tail_length]
    pre_sequence["session_behavior"] =pre_sequence["session_behavior"] [:-tail_length]
    pre_sequence["session_state"] =pre_sequence["session_state"] [:-tail_length]
    return   pre_sequence,current_session_pre,session_state
    
def Long_term_session_process(pre_sequence,current_session_pre_session_state,session_num):
    LT_session={}
    valid_sess=[]
    session_state = current_session_pre_session_state
    for sess in range(1,session_num):
        LT_session['LT_session'+str(-sess)]={'session_iid':[],"session_cid":[],"session_behavior":[],"session_state":[]}
        #1. pre为[]
        if len(pre_sequence['session_state'])==0:
      
            #赋值为[0]，并退出本次循环进行下次循环，直到所有session均赋值为[0]
            LT_session['LT_session'+str(-sess)]['session_iid']=[0]
            LT_session['LT_session'+str(-sess)]['session_cid']=[0]
            LT_session['LT_session'+str(-sess)]['session_behavior']=[0]
            LT_session['LT_session'+str(-sess)]['session_state']=[0]
            valid_sess.append(0)
            continue
        #2. pre 不为空
        #2.1 选取上一个session_state作为备选state
        session_state = str(int(session_state)-1)
        #2.2 统计备选state的数目
        tail_length = pre_sequence["session_state"].count(session_state)
        #2.3 在长期兴趣这里，对于长度小于2的·还是删除的比较好
       
        # 2.3.1 session_length <2
        cumsum_drop =0
 
        while tail_length <2 and int(session_state) >1:
            #继续往前找
            # cumsum_drop  ，是废角料 ，用于对pre_sequence截断
            
            cumsum_drop =tail_length+cumsum_drop
            session_state = str(int(session_state)-1)
            tail_length = pre_sequence["session_state"].count(session_state)
        # 2.3.1.1 往前找不到   
        if len(pre_sequence['session_state'][:-cumsum_drop]) <2 and cumsum_drop >0:
          
            #已找不到长期序列，清空pre_sequence，并安排好当前session为空
            pre_sequence['session_state']=[]#用于接下来session的处理判别标准
            LT_session['LT_session'+str(-sess)]['session_iid']=[0]
            LT_session['LT_session'+str(-sess)]['session_cid']=[0]
            LT_session['LT_session'+str(-sess)]['session_behavior']=[0]
            LT_session['LT_session'+str(-sess)]['session_state']=[0]
            valid_sess.append(0)
            continue
        #2.3.1.2  找到了
        #减去废尾后再开始其他工作
 
        if  cumsum_drop >0:
           
            pre_sequence['session_iid']=pre_sequence['session_iid'][:-cumsum_drop]
            pre_sequence['session_cid']=pre_sequence['session_cid'][:-cumsum_drop]
            pre_sequence['session_behavior']= pre_sequence['session_behavior'][:-cumsum_drop]
            pre_sequence['session_state']=pre_sequence['session_state'][:-cumsum_drop]
        #正式录入长期session
        #对于长期session的较长序列，只取最后session_num个交互恐有不妥，可以采取随机起始位截断
        if tail_length >session_length: 
            random_position=random.randint(0, tail_length-session_length)
        else:
            random_position=0
                
        LT_session['LT_session'+str(-sess)]['session_iid'] = pre_sequence['session_iid'][-tail_length:][random_position:random_position+session_length]
        LT_session['LT_session'+str(-sess)]['session_cid']= pre_sequence['session_cid'][-tail_length:][random_position:random_position+session_length]
        LT_session['LT_session'+str(-sess)]['session_behavior'] =  pre_sequence['session_behavior'][-tail_length:][random_position:random_position+session_length]
        LT_session['LT_session'+str(-sess)]['session_state'] = pre_sequence['session_state'][-tail_length:][random_position:random_position+session_length]
        #reindex   
        LT_session['LT_session'+str(-sess)]['session_iid'],  LT_session['LT_session'+str(-sess)]['session_cid'] =  get_item_cate_history_sequence(LT_session['LT_session'+str(-sess)]['session_iid'], LT_session['LT_session'+str(-sess)]['session_cid'],itemdict,catedict )
        valid_sess.append(len(LT_session['LT_session'+str(-sess)]['session_state']))
        #对pre_sequence截断 ,以方便录入下一个session
        pre_sequence['session_iid']=pre_sequence['session_iid'][:-tail_length]
        pre_sequence['session_cid']=pre_sequence['session_cid'][:-tail_length]
        pre_sequence['session_behavior']=  pre_sequence['session_behavior'][:-tail_length]
        pre_sequence['session_state']=pre_sequence['session_state'][:-tail_length]
    return LT_session,valid_sess

def process_batch_lines(batch_num,lines_batch,file_path):
     file = open(file_path+'_'+str(batch_num), "w")
     #uid ,target_iid,target_cid,target_behavior, 
    # current_session_iid_succ,current_session_cid_succ,current_session_behavior_succ,
    # current_session_iid_pre,current_session_cid_pre,current_session_behavior_pre,
    # last1_session_iid,last1_session_cid_pre,last1_session_behavior_pre,
    # last2_session_iid,last2_session_cid_pre,last2_session_behavior_pre,
    # last3_session_iid,last3_session_cid_pre,last3_session_behavior_pre,
    # last4_session_iid,last4_session_cid_pre,last4_session_behavior_pre,
    # pre_session_length_list=[x,x,x,x] 
     for line in tqdm(lines_batch) :
        words=line.strip().split('\t')
        #4.1 reindex target
        
        uid  = userdict[int(words[1])] if int(words[1]) in userdict else 0
        target_iid = itemdict[int(words[2])]  if int(words[2]) in itemdict else 0
        target_cid = catedict [int(words[3])]  if int(words[3]) in catedict else 0
        target_session_state=words[5]
        #4.2  处理current_session_succ,#注意：t+1,t+2,t+3....
        #注意：t+1,t+2,t+3....所以截断尾部
        current_session_succ = current_session_succ_process(words,session_length,reversed_choice,itemdict,catedict)
                
        #4.3 处理 pre_sequence
        #4.3.1 处理 current_session_pre
        
        pre_sequence,current_session_pre,current_session_pre_session_state = current_session_pre_process(words,target_session_state,session_length, itemdict,catedict)
        
        #4.3.2 Long-term_session
        LT_session,valid_LT_sess = Long_term_session_process(pre_sequence,current_session_pre_session_state,session_num)
        
        # 4.4 写入文件
       
        LT_all="\t"

        for sess in range(1,session_num):
            LT_all = LT_all  +str(LT_session['LT_session'+str(-sess)]['session_iid']) +"\t"\
                           +str(LT_session['LT_session'+str(-sess)]['session_cid']) +"\t"\
                            +str(LT_session['LT_session'+str(-sess)]['session_behavior']) +"\t"\
                             +str(LT_session['LT_session'+str(-sess)]['session_state']) +"\t"
            
        file.write(words[0]#将train/valid/test 的index与 原reviews_file对齐
                        + "\t"
                        + str(uid)
                        + "\t"
                        + str(target_iid)
                        + "\t"
                        + str(target_cid)
                        + "\t"
                        + words[4]
                        + "\t"
                        +  target_session_state 
                        + "\t"
                        + str(current_session_succ['session_iid'])
                        + "\t"
                        + str(current_session_succ['session_cid'])
                        + "\t"
                        + str(current_session_succ['session_behavior'])
                        + "\t"
                        + str(current_session_succ['session_state'])
                        + "\t"
                        + str(current_session_pre['session_iid'])
                        + "\t"
                        + str(current_session_pre['session_cid'])
                        + "\t"
                        + str(current_session_pre['session_behavior'])
                        + "\t"
                        + str(current_session_pre['session_state'])
                       
                        +LT_all
                        +str(valid_LT_sess)
                        + "\n"

        )
        
def multiprocess_file(origin_file ,output_file ,cpu_num):
    f_output = open(output_file, "w")
    process =[]
    each_process_batch  = (len( open(origin_file , "r").readlines() ) -1)//cpu_num+1
    with open(origin_file , "r") as f:
            lines = f.readlines()
    #index,uid ,target_iid,target_cid,target_behavior, target_session_state,
    #6~9 current_session_iid_succ,current_session_cid_succ,current_session_behavior_succ,current_session_session_state_succ
    #10~13 pre_sequence_iid,pre_sequence_cid,pre_sequence_behavior,pre_sequence_session_state
    for i in range(0, cpu_num):
        lines_batch = lines[i * each_process_batch:(i + 1) * each_process_batch]
        
        #process_batch_lines(i,lines_batch,output_train_file)
        process.append(Process(target = process_batch_lines,args =(i,lines_batch,output_file) )  ) 
        
    [p.start() for p in process] 
    [p.join() for p in process] 
    
    for i in tqdm(range(0,cpu_num)):
        f_train_batch = open(output_file+'_'+str(i)).read()  
        f_output.write(f_train_batch) 
    for i in range(0,cpu_num):
        os.remove (output_file+'_'+str(i))

    f_output.close()




#1.基本信息
dataset = "taobao"
reviews_name = 'new_UserBehavior'
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
reviews_file = os.path.join(reviews_path,   reviews_name)
session_LS_data_file =  os.path.join(reviews_path,   "session_LS")
os.makedirs(session_LS_data_file, mode = 0o777, exist_ok = True)
 
cpu_num =100
session_num =5
session_length = 10
reversed_choice=True
user_vocab = os.path.join(reviews_path ,r'user_vocab.pkl')
item_vocab = os.path.join(reviews_path, r'item_vocab.pkl')
cate_vocab = os.path.join(reviews_path,r'category_vocab.pkl')
origin_train_file = os.path.join(reviews_path, r'train_data_w_succ')
origin_sampled_valid_file = os.path.join(reviews_path, r'sampled_valid_data_w_succ')
origin_sampled_test_file = os.path.join(reviews_path, r'sampled_test_data_w_succ')

#2.输出文件

output_train_file = os.path.join(session_LS_data_file, r'train_data')
output_valid_file = os.path.join(session_LS_data_file, r'valid_data')
output_test_file = os.path.join(session_LS_data_file, r'test_data')


#3.读取文件 
userdict,  itemdict,  catedict = (
            load_dict(user_vocab),
            load_dict(item_vocab),
            load_dict(cate_vocab),
        )
 


#4.处理session_w_succ
 
multiprocess_file(origin_train_file ,output_train_file,cpu_num)

multiprocess_file(origin_sampled_valid_file ,output_valid_file,cpu_num)
multiprocess_file(origin_sampled_test_file ,output_test_file,cpu_num)

    
            
 


                 

    
             

                  
                  
                 

            











