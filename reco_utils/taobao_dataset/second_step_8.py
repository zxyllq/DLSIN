import os
from datetime import datetime
import pandas as pd
import numpy as np
import pdb 
from tqdm import tqdm
from multiprocessing import Process
#logger = logging.getLogger()
import warnings
#from pandas.core.common import SettingWithCopyWarning
import _pickle as cPickle
import pickle as pkl
import random
import shutil
from operator import itemgetter
#warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
import copy

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
            itemdict[ item ] if  item  in  itemdict else 0
        )
    return item_history_sequence

def get_cate_history_sequence( cate_history_words,catedict):

    cate_history_sequence = []
    for cate in cate_history_words:
        cate_history_sequence.append(
            catedict[ cate ] if  cate  in  catedict else 0
        )
    return cate_history_sequence


def vocab_make(data_file,reviews):
    user_vocab_file = os.path.join(data_file ,r'user_vocab.pkl')
    item_vocab_file = os.path.join(data_file, r'item_vocab.pkl')
    cate_vocab_file = os.path.join(data_file,r'category_vocab.pkl')
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
    cPickle.dump(user_voc, open(user_vocab_file, "wb"))
    cPickle.dump(item_voc, open(item_vocab_file, "wb"))
    cPickle.dump(cate_voc, open(cate_vocab_file, "wb"))



def sampling(data_file,sample_probability,batch_num):
    f_all_valid = open(os.path.join( data_file , r'all_valid_data')+'_'+str(batch_num), "r")
    f_all_test = open(os.path.join( data_file , r'all_test_data')+'_'+str(batch_num), "r")
    f_sampled_valid = open(os.path.join( data_file , r'sample_valid_data')+'_'+str(batch_num), "w")
    f_sampled_test  =  open(os.path.join( data_file , r'sample_test_data')+'_'+str(batch_num), "w")
    for line in f_all_valid:
        probability = round(np.random.uniform(0, 1), 1)
        if 0 <= probability <=sample_probability:
            f_sampled_valid.writelines(line)
    for line in f_all_test:
        probability = round(np.random.uniform(0, 1), 1)
        if 0 <= probability <=sample_probability:
            f_sampled_test.writelines(line)
    f_all_valid.close()
    f_all_test.close()
    f_sampled_valid.close()
    f_sampled_test.close()

def negtive_sampling(original_file,output_file,reviews,negtive_sample_rate,item_vocab_file,cate_vocab_file):
    with open(original_file, "r") as f:
        lines = f.readlines()
    negtive_sample_lines   = open(output_file, "w")
    #注意这里的reviews是没有reid的
    iid_list = reviews["iid"].to_list()
    cid_list = reviews["cid"].to_list()
    itemdict,  catedict = ( load_dict(item_vocab_file), load_dict(cate_vocab_file) )
    for line in  tqdm(lines ) :
        
        words = line.strip().split("\t")
        negtive_sample_lines.write("1\t"+"\t".join(words) +"\n") 
        positive_iid =words[2]
         
        
        count = 0
        neg_items = set()
        while  count < negtive_sample_rate:
            index= random.choice(reviews.index)
            neg_item = itemdict[iid_list[index] ]  if iid_list[index] in itemdict else 0
            if  str(neg_item) == positive_iid  or neg_item in neg_items:
                continue
            neg_cate = catedict[cid_list[index] ] if cid_list[index] in catedict else 0
            count += 1
            neg_items.add(neg_item)
            words[2] = str(neg_item)
            words[3] = str(neg_cate)
            negtive_sample_lines.write("0\t"+"\t".join(words) +"\n")

def merge_file(sequence_data_file,cpu_num):
    f_train_output = open(os.path.join(sequence_data_file , r'train_data'), "w") 
    f_all_valid_output = open(os.path.join(sequence_data_file , r'all_valid_data'), "w") 
    f_all_test_output = open(os.path.join(sequence_data_file , r'all_test_data'), "w") 
    f_sample_valid_output = open(os.path.join(sequence_data_file , r'sample_valid_data'), "w") 
    f_sample_test_output = open(os.path.join(sequence_data_file , r'sample_test_data'), "w") 
    for i in tqdm(range(0,cpu_num)):
        f_train_batch = open(os.path.join(sequence_data_file , r'train_data')+'_'+str(i)).read()  
        f_all_valid_batch = open(os.path.join(sequence_data_file , r'all_valid_data')+'_'+str(i)).read() 
        f_all_test_batch = open(os.path.join(sequence_data_file , r'all_test_data')+'_'+str(i)).read() 
        f_sample_valid_batch = open(os.path.join(sequence_data_file , r'sample_valid_data')+'_'+str(i)).read() 
        f_sample_test_batch = open(os.path.join(sequence_data_file , r'sample_test_data')+'_'+str(i)).read() 
        f_train_output.write(f_train_batch) 
        f_all_valid_output.write(f_all_valid_batch) 
        f_all_test_output.write(f_all_test_batch) 
        f_sample_valid_output.write(f_sample_valid_batch) 
        f_sample_test_output.write(f_sample_test_batch) 
    for i in range(0,cpu_num):
        os.remove (os.path.join(sequence_data_file , r'train_data')+'_'+str(i))
        os.remove (os.path.join(sequence_data_file , r'all_valid_data')+'_'+str(i))
        os.remove (os.path.join(sequence_data_file , r'all_test_data')+'_'+str(i))
        os.remove (os.path.join(sequence_data_file , r'sample_valid_data')+'_'+str(i))
        os.remove (os.path.join(sequence_data_file , r'sample_test_data') +'_'+str(i))

def label_1_make( data_file  ):
    file_type=["train_data","all_valid_data","all_test_data"]
     
    for  file_i in  file_type :
        file_path =os.path.join( data_file, file_i)
        f_output=  pd.read_csv(file_path,sep='\t',header=None)
        f_output.insert(0, 'label',  1 ) 
    
        f_output.to_csv(file_path,sep='\t',header=None,index=False) 


def sequence_dataset_process(batch_num,reviews_batch,user_batch, sequence_data_file, max_length,sample_probability,user_vocab_file,item_vocab_file,cate_vocab_file):
    f_train = open(os.path.join(sequence_data_file , r'train_data')+'_'+str(batch_num), "w")
    f_all_valid = open(os.path.join(sequence_data_file , r'all_valid_data')+'_'+str(batch_num), "w")
    f_all_test = open(os.path.join(sequence_data_file , r'all_test_data')+'_'+str(batch_num), "w")
    userdict,  itemdict,  catedict = (
            load_dict(user_vocab_file),
            load_dict(item_vocab_file),
            load_dict(cate_vocab_file),
        )
     
    for user in tqdm(user_batch):
        user_interactions=reviews_batch[ reviews_batch["uid"]==user]
        user = userdict[user] if  user  in userdict else 0
        for target_row in  user_interactions.index :#这个index需要保留原reviews信息

            ref_index =user_interactions["ref_index"].loc[target_row]
            #target
             
            
            target_state =  user_interactions["state"].loc[target_row]
            target_iid = user_interactions["iid"].loc[target_row]
            target_iid  = itemdict[target_iid ] if target_iid in itemdict else 0
            target_cid = user_interactions["cid"].loc[target_row]
            target_cid  = catedict[target_cid ] if target_cid in catedict else 0
            target_behavior =   user_interactions["behavior_type"].loc[target_row]
            target_ts =    user_interactions["ts"].loc[target_row]
             
            #pre_sequence
            # all_reviews.drop_duplicates(subset=['uid', 'iid'])
            pre_sequence = user_interactions.loc[:target_row-1][-max_length :] 
            if   pre_sequence.empty ==True :
                #无pre，那么就结束本次循环进入下一个target  
                continue
            pre_sequence_iid = pre_sequence["iid"].to_list() 
            pre_sequence_cid= pre_sequence["cid"].to_list()
            pre_sequence_behavior= pre_sequence["behavior_type"].to_list()
            pre_sequence_ts= pre_sequence["ts"].to_list()
            pre_sequence_iid , pre_sequence_cid = get_item_cate_history_sequence(  pre_sequence_iid , pre_sequence_cid,itemdict,catedict)


            

          
            if target_state == 0:
                fo = f_train
            elif target_state == 1:
                fo = f_all_valid
            elif target_state == 2:
                fo = f_all_test
            fo.write(
                        str(ref_index )#将train/valid/test 的index与 原reviews_file对齐
                        + "\t"
                        + str(user)
                        + "\t"
                        + str(target_iid)
                        + "\t"
                        + str(target_cid)
                        + "\t"
                        + str(target_behavior)
                        + "\t"
                     
                        + str(target_ts)
                        + "\t"
                        + str(pre_sequence_iid)
                        + "\t"
                        + str(pre_sequence_cid)
                        + "\t"
                        + str(pre_sequence_behavior)
                        + "\t"
                        + str(pre_sequence_ts)
                 
                        + "\n"
                    )
    f_train.close()
    f_all_valid.close()
    f_all_test.close()
    #4.对测试集和验证集采样
    
    sampling(sequence_data_file,sample_probability,batch_num)
     

def sequence_make(reviews,cpu_num, sequence_data_file,max_length,sample_probability,  valid_negtive_sample,test_negtive_sample ):
    user_vocab_file = os.path.join(sequence_data_file ,r'user_vocab.pkl')
    item_vocab_file = os.path.join(sequence_data_file, r'item_vocab.pkl')
    cate_vocab_file = os.path.join(sequence_data_file,r'category_vocab.pkl')
    
    all_user =reviews.uid.unique()
    process=[]
    each_process_batch  = (len( all_user ) -1)//cpu_num+1
    for i in range(0, cpu_num):
        user_batch = all_user[i * each_process_batch:(i + 1) * each_process_batch]
        reviews_batch =  reviews.loc[ reviews.uid.isin(user_batch)]
        
        #sequence_dataset_process(i,reviews_batch,user_batch, sequence_data_file,max_length ,sample_probability,user_vocab_file,item_vocab_file,cate_vocab_file)
        process.append(Process(target = sequence_dataset_process,args =(i,reviews_batch, user_batch,sequence_data_file,max_length ,sample_probability,user_vocab_file,item_vocab_file,cate_vocab_file) )  ) 
        
    [p.start() for p in process] 
    [p.join() for p in process] 
    #合并文件和删除碎片
    merge_file(sequence_data_file,cpu_num)
    
  

    negtive_sampling(os.path.join(sequence_data_file , r'sample_valid_data'),os.path.join(sequence_data_file , r'sample_valid_data'),reviews,valid_negtive_sample,item_vocab_file,cate_vocab_file)
    negtive_sampling(os.path.join(sequence_data_file , r'sample_test_data'),os.path.join(sequence_data_file , r'sample_test_data'),reviews,test_negtive_sample,item_vocab_file,cate_vocab_file)

 

 

def Long_term_session_process(pre_sequence_df ,max_sess_num,max_session_length ,session_state,itemdict,catedict) :
    #1.先将其全部转为list
    pre_sequence = {'iid':[],"cid":[] ,session_state:[]}
     #注意 0，1，2，...,t-1,t 
    pre_sequence['iid']=  pre_sequence_df['iid'].to_list()
    pre_sequence["cid"] = pre_sequence_df["cid"].to_list()
   # pre_sequence["behavior_type"] = pre_sequence_df["behavior_type"].to_list()
    pre_sequence[ session_state ] = pre_sequence_df[ session_state ].to_list()
    
    LT_session={}
    valid_sess=[]
    
    for sess in range(max_sess_num,0,-1):
        LT_session['LT_session'+str(-sess)]={'iid':[],"cid":[],session_state:[] }
        #1. len pre <=2 不记录
        if len(pre_sequence[ session_state ]  )<2:
            #赋值为[0]，并退出本次循环进行下次循环，直到所有session均赋值为[0]
            LT_session['LT_session'+str(-sess)]['iid']=[0]
            LT_session['LT_session'+str(-sess)]['cid']=[0]
            LT_session['LT_session'+str(-sess)][session_state]=[0]
           
            valid_sess.append(0)
            continue
        #2. len pre >2 
        #2.1 len session  <=2 
        #在长期兴趣这里，对于长度小于等于2的，还是删除的比较好
        bak_session_state =  pre_sequence[ session_state ][0]
        session_length =  pre_sequence[session_state ].count(bak_session_state) 
        cumsum_drop =0
        while  session_length<2 and cumsum_drop<len(pre_sequence[ session_state ]) :
            #寻找下一个
            
            cumsum_drop=cumsum_drop+session_length
            bak_session_state=bak_session_state+1
            session_length = pre_sequence[session_state ].count(bak_session_state) 
        #全程都未能找到
        if   cumsum_drop==len(pre_sequence[ session_state ]):
             
            pre_sequence[ session_state ] =[]#用于接下来session的处理判别标准
            LT_session['LT_session'+str(-sess)]['iid']=[0]
            LT_session['LT_session'+str(-sess)]['cid']=[0]
            LT_session['LT_session'+str(-sess)][session_state]=[0]
            valid_sess.append(0)
            continue
        #找到了，先截断cumsum_drop
         
        pre_sequence['iid']=pre_sequence['iid'][ cumsum_drop:]
        pre_sequence['cid']=pre_sequence['cid'][ cumsum_drop:]
        pre_sequence[session_state]=pre_sequence[session_state][ cumsum_drop:]
        #录用
        if session_length <= max_session_length :
            
            LT_session['LT_session'+str(-sess)]['iid']=pre_sequence['iid'][:session_length]
            LT_session['LT_session'+str(-sess)]['cid']=pre_sequence['cid'][:session_length]
            LT_session['LT_session'+str(-sess)][session_state] =  pre_sequence[session_state][:session_length]
            valid_sess.append(len(LT_session['LT_session'+str(-sess)]['iid']))
        else:
            #任选10个
            
            index= [i for i in range( session_length )]
            random_choice =  sorted(random.sample(index, max_session_length) )
            LT_session['LT_session'+str(-sess)]['iid']=itemgetter(*random_choice)( pre_sequence['iid'])
            LT_session['LT_session'+str(-sess)]['cid']=itemgetter(*random_choice)( pre_sequence['cid'])
            LT_session['LT_session'+str(-sess)][session_state]=itemgetter(*random_choice)( pre_sequence[session_state])
            valid_sess.append(len(LT_session['LT_session'+str(-sess)]['iid']))    
        #能走到这里，就说明给录用了，需要reid
        LT_session['LT_session'+str(-sess)]['iid']  , LT_session['LT_session'+str(-sess)]['cid'] =  get_item_cate_history_sequence( LT_session['LT_session'+str(-sess)]['iid']  , LT_session['LT_session'+str(-sess)]['cid'],itemdict,catedict )
        #截断，方便下次循环
        pre_sequence['iid']=pre_sequence['iid'][ session_length:]
        pre_sequence['cid']=pre_sequence['cid'][ session_length:]
        pre_sequence[session_state]=pre_sequence[session_state][ session_length:]

          
         
         
       
    return LT_session,valid_sess
def current_session_succ_process(user_interactions,target_session_state , max_session_length,session_state  ):
    
    current_session_succ = user_interactions[user_interactions[ session_state ]== target_session_state ]  #+1表示自target之后,先全部取到
    if  current_session_succ.empty ==True :
        #target是当前session的最后一个，取下一个session作为successor
        #对于下一个session，不建议取的太久远，如果下一个session长度比较小，也是可以接受的
        current_session_succ = user_interactions[user_interactions[session_state]== target_session_state+1 ] 
    #如果仍为空，就是[] 
    if current_session_succ.empty ==True :

        current_session_iid_succ = [0]
        current_session_cid_succ =  [0]
        
    else:#先全部取到，再截断，再反转 
        current_session_iid_succ =  list(reversed(current_session_succ["iid"].to_list()[:max_session_length]))
        current_session_cid_succ =   list(reversed(current_session_succ["cid"].to_list()[:max_session_length]))
        
    return current_session_iid_succ,current_session_cid_succ 

def  current_session_pre_process(pre_sequence_df,target_session_state , max_session_length,session_state  ):
    #1.先将其全部转为list
    pre_sequence = {'iid':[],"cid":[], session_state:[]}
     #注意 0，1，2，...,t-1,t 
    pre_sequence['iid']=  pre_sequence_df['iid'].to_list()
    pre_sequence["cid"] = pre_sequence_df["cid"].to_list()
    
    pre_sequence[ session_state ] = pre_sequence_df[ session_state ].to_list()

    #2.上一个session和target_session_state只有两种关系，要么相等，要么小1
    #判断target和上一个session的关系 
    if target_session_state in pre_sequence[ session_state ] :   
        #pre_session与 target_session_state 同属一个session
        target_session_state = target_session_state
    else :
        #pre_session先于 target_session_state 
        target_session_state = target_session_state -1 
   
    #3.统计last1_session_state在pre_sequence的数目，并截取末尾session_length个数据
     #注意 0，1，2，...,t-1,t 
    tail_length = pre_sequence[ session_state ].count(target_session_state )
    current_session_iid_pre=pre_sequence['iid'][-tail_length:][-max_session_length:]
    current_session_cid_pre=pre_sequence['cid'][-tail_length:][-max_session_length:]
    
    current_session_session_state_pre=pre_sequence[session_state][-tail_length:][-max_session_length:]

     

    return  current_session_iid_pre,current_session_cid_pre 

def session_LS_dataset_process_4(batch_num,reviews_batch,  user_batch, session_LS_data_file,max_session_length,max_sess_num, user_vocab_file,item_vocab_file,cate_vocab_file,session_interval,sample_probability):
    f_train = open(os.path.join(session_LS_data_file , r'train_data')+'_'+str(batch_num), "w")
    f_all_valid = open(os.path.join(session_LS_data_file, r'all_valid_data')+'_'+str(batch_num), "w")
    f_all_test = open(os.path.join(session_LS_data_file , r'all_test_data')+'_'+str(batch_num), "w")
    userdict,  itemdict,  catedict = (
            load_dict(user_vocab_file),
            load_dict(item_vocab_file),
            load_dict(cate_vocab_file),
        )
    
    session_state ="session_state_"+session_interval
    for user in tqdm(user_batch):
        #每次处理一个用户
        
        user_interactions=reviews_batch[ reviews_batch["uid"]==user]
        user =  userdict[user] if  user  in userdict else 0
        for target_row in  user_interactions.index :
            target_session_state = user_interactions[session_state].loc[target_row]
            #session_length = len(user_interactions[ user_interactions[session_state]==target_session_state ])
            # if session_length<=2:#不进行收录,但愿用不到这个
            #     continue

            ref_index =user_interactions["ref_index"].loc[target_row]
            #target
            target_state =  user_interactions["state"].loc[target_row]#区分 target_session_state 
            target_iid = user_interactions["iid"].loc[target_row]
            target_iid  = itemdict[target_iid ] if target_iid in itemdict else 0
            target_cid = user_interactions["cid"].loc[target_row]
            target_cid  = catedict[target_cid ] if target_cid in catedict else 0
            target_ts= user_interactions["ts"].loc[target_row]
            target_behavior =   user_interactions["behavior_type"].loc[target_row]
            
            #current_session_succ
            #(user_interactions.loc[target_row+1:] 把target之后的interactions全部传进去，一个一个翻找,但不能超出当前state
            #，为避免数据泄露，需保证下一个session与当前target同属于一个state
            # #current session 都两倍长度
            # current_session_iid_succ,current_session_cid_succ,current_session_behavior_succ ,current_session_session_state_succ =current_session_succ_process(user_interactions[user_interactions["state"]== target_state].loc[target_row+1:] ,target_session_state , max_session_length*2,session_state )
            # current_session_iid_succ , current_session_cid_succ =  get_item_cate_history_sequence( current_session_iid_succ , current_session_cid_succ,itemdict,catedict )
            
            #pre_sequence
            #先把pre_sequence 全部取到
            pre_sequence =  user_interactions.loc[:target_row-1].drop_duplicates(subset=['uid', 'iid'],keep='last')[-max_session_length*max_sess_num :] 
            if   pre_sequence.empty ==True :
                #无pre，那么就结束本次循环进入下一个target             
                continue
             
            pre_sequence_iid = pre_sequence["iid"].to_list() 
            pre_sequence_cid= pre_sequence["cid"].to_list()
            pre_sequence_ts= pre_sequence["ts"].to_list()
            pre_sequence_behavior= pre_sequence["behavior_type"].to_list()
            pre_sequence_session_state= pre_sequence[session_state].to_list()
            pre_sequence_iid , pre_sequence_cid = get_item_cate_history_sequence(  pre_sequence_iid , pre_sequence_cid,itemdict,catedict)

            # Long-term_session 
            # 从距离target较远的session开始去取，
            # 另外对session_length有要求,session_length>2
            #  session_4,session_3,session_2,session_1,session_0
         
            LT_session,valid_LT_sess =Long_term_session_process(pre_sequence ,max_sess_num,max_session_length, session_state,itemdict,catedict) 
            # 4.4 写入文件
            # if np.sum(valid_LT_sess):#不强制规定至少有一个长期 session, 
            #     continue

            LT_all="\t"
            for sess in range(max_sess_num,0,-1):
                LT_all = LT_all  +str(LT_session['LT_session'+str(-sess)]['iid']) +"\t"\
                           +str(LT_session['LT_session'+str(-sess)]['cid']) +"\t" #+str(LT_session['LT_session'+str(-sess)][session_state]) +"\t" 
                           
            current_session_iid_pre,current_session_cid_pre  = current_session_pre_process(user_interactions.loc[:target_row-1],target_session_state , max_session_length,session_state  )                
            current_session_iid_pre,current_session_cid_pre  = get_item_cate_history_sequence( current_session_iid_pre,current_session_cid_pre,itemdict,catedict)
            current_session_iid_succ,current_session_cid_succ = current_session_succ_process(user_interactions[user_interactions["state"]== target_state].loc[target_row+1:],target_session_state , max_session_length,session_state  )
            current_session_iid_succ,current_session_cid_succ =get_item_cate_history_sequence(current_session_iid_succ,current_session_cid_succ ,itemdict,catedict)
            if target_state == 0:
                fo = f_train
            elif target_state == 1:
                fo = f_all_valid
            elif target_state == 2:
                fo = f_all_test
            
            fo.write(str(ref_index )#将train/valid/test 的index与 原reviews_file对齐
                        + "\t"
                        + str(user)
                        + "\t"
                        + str(target_iid)
                        + "\t"
                        + str(target_cid)
                        + "\t"
                        +str(target_ts)
                        + "\t"
                        + str(target_behavior )
                        + "\t"
                       # + str( target_session_state )
                       #  + "\t"
                        +str(pre_sequence_iid)
                        + "\t"
                        +str(pre_sequence_cid)
                        + "\t"
                        +str(pre_sequence_ts)
                         #+ "\t"
                    #    +str(pre_sequence_behavior)
                     #   + "\t"
                      #  +str(pre_sequence_session_state)

                        
                       
                        +LT_all
                        +str(valid_LT_sess)
                        + "\t"
                      +str(current_session_iid_pre)
                      + "\t"
                      +str(current_session_cid_pre)
                      + "\t"
                      +str(current_session_iid_succ)
                    + "\t"
                      +str(current_session_cid_succ)
                        + "\n"

                        )

    f_train.close()
    f_all_valid.close()
    f_all_test.close()

    sampling(session_LS_data_file,sample_probability,batch_num)    
            
 
 
        

def session_LS_make(reviews,cpu_num, session_LS_data_file,  max_session_length,max_sess_num,   session_interval ,valid_negtive_sample,test_negtive_sample,sample_probability ):
    
    user_vocab_file = os.path.join(session_LS_data_file ,r'user_vocab.pkl')
    item_vocab_file = os.path.join(session_LS_data_file, r'item_vocab.pkl')
    cate_vocab_file = os.path.join(session_LS_data_file,r'category_vocab.pkl')
     
    process =[]
    all_user =reviews.uid.unique()
    each_process_batch  = (len( all_user ) -1)//cpu_num+1
    for i in range(0, cpu_num):
        user_batch = all_user[i * each_process_batch:(i + 1) * each_process_batch]
        reviews_batch =  reviews.loc[ reviews.uid.isin(user_batch)]
  
        #session_LS_dataset_process_4(i,reviews_batch, user_batch,  session_LS_data_file,max_session_length,max_sess_num, user_vocab_file,item_vocab_file,cate_vocab_file,session_interval ,sample_probability)
        process.append(Process(target = session_LS_dataset_process_4,args =(i,reviews_batch, user_batch,  session_LS_data_file,max_session_length,max_sess_num, user_vocab_file,item_vocab_file,cate_vocab_file,session_interval,sample_probability) )  ) 
    
    [p.start() for p in process] 
    [p.join() for p in process] 
    #合并文件，和删除碎片
   
    merge_file(session_LS_data_file,cpu_num)
    # 对照sequence 文件降采样和负采样
     
     
    negtive_sampling(os.path.join(session_LS_data_file , r'sample_valid_data'),os.path.join(session_LS_data_file , r'sample_valid_data'),reviews,valid_negtive_sample,item_vocab_file,cate_vocab_file)
    negtive_sampling(os.path.join(session_LS_data_file , r'sample_test_data'),os.path.join(session_LS_data_file , r'sample_test_data'),reviews,test_negtive_sample,item_vocab_file,cate_vocab_file)

    #align_negtive_sample_method(sequence_data_file,session_LS_data_file,valid_negtive_sample,test_negtive_sample)


def noise_insert(reviews,noise_rate):
    reviews_train_valid =reviews[reviews.state<2]#仅仅对于train和valid加噪音
  #  iid_list = reviews ["iid"].to_list()
  #  cid_list = reviews ["cid"].to_list()
    noise_index = random.sample(reviews_train_valid.index.to_list(),k=int(len(reviews_train_valid.index)*noise_rate))
    noise_replace_index =  random.choices(reviews.index,k=int(len(reviews_train_valid.index)*noise_rate))
    noise_replace_df=  reviews.loc[noise_replace_index]
    noise_df = reviews.loc[noise_index]
    noise_replace_df.index=noise_df.index
    reviews.loc[noise_index,['iid'] ] =noise_replace_df['iid'] 
    reviews.loc[noise_index,['cid'] ] =noise_replace_df['cid'] 
    return   reviews,noise_index

 



     
 
    





#1.基本信息
dataset = "taobao"
session_reviews_name = 'new_UserBehavior_with_session_state'
sequence_reviews_name = 'new_UserBehavior_with_session_state_for_sequence_model'
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
sequence_reviews_file = os.path.join(reviews_path,   sequence_reviews_name)
session_reviews_file = os.path.join(reviews_path,   session_reviews_name)
session_sample_probability = 0.15
sequence_sample_probability = 0.2
noise_rate=0.3
valid_negtive_sample= 4
test_negtive_sample = 99 
#sequence session_LS session 
#session_interval=['30','90','120' ,'150','180','210', '240', '270', '300', '330', '360' , '390', '420', '450','480', '510', '540',  '600', '720','960'] 
session_interval= '360' 
max_sequence_length=50
max_session_length =10
max_sess_num = 5
session_LS_data_file_clean =  os.path.join(reviews_path,   "session_LS",session_interval+'_clean_w_suc' )
os.makedirs(session_LS_data_file_clean, mode = 0o777, exist_ok = True) 
 
sequence_data_file_clean =  os.path.join(reviews_path,   "sequence"+'_clean' )
os.makedirs(sequence_data_file_clean, mode = 0o777, exist_ok = True) 

session_LS_data_file_noise =  os.path.join(reviews_path,   "session_LS",session_interval+'_noise_'+str(noise_rate)+'_w_suc' )
os.makedirs(session_LS_data_file_noise, mode = 0o777, exist_ok = True) 
 
sequence_data_file_noise =  os.path.join(reviews_path,   "sequence"+'_noise_' +str(noise_rate))
os.makedirs(sequence_data_file_noise, mode = 0o777, exist_ok = True) 

cpu_num =100
#2.输出文件
# user_vocab_file = os.path.join(reviews_path ,r'user_vocab.pkl')
# item_vocab_file = os.path.join(reviews_path, r'item_vocab.pkl')
# cate_vocab_file = os.path.join(reviews_path,r'category_vocab.pkl')

# session_LS_train_file = os.path.join(session_LS_data_file , r'train_data')
# session_LS_all_valid_file = os.path.join(session_LS_data_file , r'all_valid_data')
# session_LS_all_test_file = os.path.join(session_LS_data_file , r'all_test_data')
# session_LS_sample_valid_file = os.path.join(session_LS_data_file , r'sample_valid_data')
# session_LS_sample_test_file = os.path.join(session_LS_data_file , r'sample_test_data')

 
# sequence_train_file = os.path.join(sequence_data_file , r'train_data')
# sequence_all_valid_file = os.path.join(sequence_data_file , r'all_valid_data')
# sequence_all_test_file = os.path.join(sequence_data_file , r'all_test_data')
# sequence_sample_valid_file = os.path.join(sequence_data_file , r'sample_valid_data')
# sequence_sample_test_file = os.path.join(sequence_data_file , r'sample_test_data')

#3.读取文件,注意，仅读取指定session_interval 

pdb.set_trace() 
session_reviews = pd.read_csv(session_reviews_file, sep='\t' )[['ref_index','uid', 'iid', 'cid', "behavior_type", 'ts','state',"session_state_"+session_interval]]
sequence_reviews = pd.read_csv(sequence_reviews_file, sep='\t'  )[['ref_index','uid', 'iid', 'cid', "behavior_type", 'ts','state' ]]
 
# #制作sequence file
# vocab_make( sequence_data_file_clean,session_reviews )
# sequence_make(sequence_reviews ,cpu_num, sequence_data_file_clean, max_sequence_length,sequence_sample_probability, valid_negtive_sample,test_negtive_sample )
# label_1_make( sequence_data_file_clean  )
# #制作session_LS_file
vocab_make( session_LS_data_file_clean ,session_reviews )
session_LS_make(session_reviews,cpu_num, session_LS_data_file_clean,  max_session_length,max_sess_num   ,session_interval,valid_negtive_sample,test_negtive_sample ,session_sample_probability)
label_1_make( session_LS_data_file_clean  )

#4.加噪声，训练集和验证集有噪声，训练集干净，但是这个干净到底怎么定义呢？只是target吗？还是sequence也要干净？
# pdb.set_trace()
# vocab_make( sequence_data_file_noise ,sequence_reviews )#还是制作vocab,谁也别依赖谁
# new_sequence_reviews = copy.deepcopy(sequence_reviews)
# new_sequence_reviews,sequence_noise_index =noise_insert(new_sequence_reviews ,noise_rate)
# sequence_make(new_sequence_reviews,cpu_num, sequence_data_file_noise,   max_sequence_length,sequence_sample_probability,valid_negtive_sample,test_negtive_sample  )
# label_1_make( sequence_data_file_noise  )
 

 
vocab_make( session_LS_data_file_noise ,session_reviews )#还是制作vocab,谁也别依赖谁
new_session_reviews=copy.deepcopy(session_reviews )
new_session_reviews,session_noise_index =noise_insert(new_session_reviews,noise_rate)
session_LS_make(new_session_reviews,cpu_num, session_LS_data_file_noise,  max_session_length,max_sess_num   ,session_interval,valid_negtive_sample,test_negtive_sample ,session_sample_probability)
 
label_1_make( session_LS_data_file_noise  )
 


