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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import MultipleLocator
def diversity_col(col_name,origin_file):
     
     
    diversity_data= pd.read_csv(origin_file,sep='\t',names=['uid','coverage','entropy','gini','sess_length','sess_num'])
   
    col_diversity_data =diversity_data[col_name].tolist()
    col_length_data =diversity_data['sess_length'].tolist()
    col_sess_num_data =diversity_data['sess_num'].tolist()
    sess_data =[]
    sequence_data=[]
    sess_length_data =[]
    sequence_length_data=[]
    sess_num_data=col_sess_num_data
    for i in range(0,len(diversity_data)):
        diversity_row=col_diversity_data[i].strip('[').strip(']').split(', ')
        length_row=col_length_data[i].strip('[').strip(']').split(', ')
        sequence_data.append( diversity_row.pop()  )  #最后一个是sequence信息
        sequence_length_data.append(length_row.pop())
        if len(diversity_row)==0:
            continue
        sess_data.extend(diversity_row) 
        sess_length_data.extend(length_row) 
    sess_data = list(map(float,sess_data))
    sess_length_data = list(map(int,sess_length_data))
    sequence_data = list(map(float,sequence_data))
    sequence_length_data = list(map(int,sequence_length_data))
    
    return sess_data ,sequence_data , sess_length_data,sequence_length_data, sess_num_data
def no_length_info_diversity_plot(session_interval ,plot_interval_dict,picture_file,diversity_type):
    color_choose=[ '#FFFF00','#87CEEB','#0000FF','#800080','#006400','#FFFF00','#FFA500',"#01a2d9","#dc2624",'#098154' ,'#FF0000','#87CEEB','#808080'] 
   
    plt.figure(figsize=(10, 8), dpi=80)
    sns.set(style="whitegrid", font_scale=1.1)
    for key,values in tqdm(plot_interval_dict.items()) :
        data=np.array(values)
        
        if   diversity_type=="coverage":    
            data=data[(data!=1.0)&(data!=0.5)]
        
        sns.kdeplot( data ,bw_adjust=2,  shade=False,  color=color_choose.pop(), label=key, alpha=.7)
        
        
    plt.title(diversity_type, fontsize=18)
    plt.legend()
    plt.savefig(picture_file,dpi=600)
def with_length_info_diversity_plot(session_interval ,sess_length,plot_interval_dict,picture_file,diversity_type):
    
 
    diversity_length_df= pd.DataFrame(columns=[diversity_type,"interval","length"])
    sns.set(style="whitegrid", font_scale=1.1)
    for interval  in tqdm(session_interval) :
        interval_diversity_length_df= pd.DataFrame(columns=[diversity_type,"interval","length"])
        interval_diversity_length_df[diversity_type]=plot_interval_dict[diversity_type+"_"+interval]
        interval_diversity_length_df["interval"]= interval 
        interval_diversity_length_df["length"]=sess_length["length_"+interval]
        diversity_length_df=diversity_length_df.append(interval_diversity_length_df)
    
    length_split=[10,25  ]
    
    if diversity_type=="coverage":
        
        diversity_length_df=diversity_length_df[(diversity_length_df["coverage"]!=2/3)&(diversity_length_df["coverage"]!=1.0)&(diversity_length_df["length"]>2) ]
         
    fig, ax = plt.subplots(2,2,figsize=(20,20))
    

    color_choose=[ '#FFFF00','#87CEEB','#0000FF','#800080','#006400','#FFFF00','#FFA500',"#01a2d9","#dc2624",'#098154' ,'#FF0000','#87CEEB','#808080'] 
    for interval  in tqdm(session_interval) :
        sns.kdeplot( data= diversity_length_df[   diversity_length_df.interval == interval ][diversity_type], shade=False,  bw_adjust=3,    color=color_choose.pop(), label=interval, alpha=.7,ax=ax[0][0])      
    ax[0][0].set_xlabel(diversity_type+'_all')    

    color_choose=[ '#FFFF00','#87CEEB','#0000FF','#800080','#006400','#FFFF00','#FFA500',"#01a2d9","#dc2624",'#098154' ,'#FF0000','#87CEEB','#808080'] 
    for interval  in tqdm(session_interval) :
        sns.kdeplot( data= diversity_length_df[(diversity_length_df.length<=length_split[0])&(diversity_length_df.interval == interval)][diversity_type], shade=False,bw_adjust=2,color=color_choose.pop(), label=interval, alpha=.7,ax=ax[0][1])      
    ax[0][1].set_xlabel( diversity_type+'<='+str(length_split[0]))   
    color_choose=[ '#FFFF00','#87CEEB','#0000FF','#800080','#006400','#FFFF00','#FFA500',"#01a2d9","#dc2624",'#098154' ,'#FF0000','#87CEEB','#808080'] 
    for interval  in tqdm(session_interval) :
        sns.kdeplot(  data=diversity_length_df[(diversity_length_df.length<=length_split[1])&(diversity_length_df.length >length_split[0])&(diversity_length_df.interval == interval)][diversity_type],  shade=False,  color=color_choose.pop(), label=interval, alpha=.7,ax=ax[1][0])      
    ax[1][0].set_xlabel(  str(length_split[0])+'<'+diversity_type+'<='+str(length_split[1]))   
    color_choose=[ '#FFFF00','#87CEEB','#0000FF','#800080','#006400','#FFFF00','#FFA500',"#01a2d9","#dc2624",'#098154' ,'#FF0000','#87CEEB','#808080'] 
    for interval  in tqdm(session_interval) :
        sns.kdeplot(  data=diversity_length_df[ (diversity_length_df.length >length_split[1])&(diversity_length_df.interval == interval)][diversity_type], shade=False,  color=color_choose.pop(), label=interval, alpha=.7,ax=ax[1][1])      
    ax[1][1].set_xlabel(  str(length_split[1])+'<'+diversity_type)   
    color_choose=[ '#FFFF00','#87CEEB','#0000FF','#800080','#006400','#FFFF00','#FFA500',"#01a2d9","#dc2624",'#098154' ,'#FF0000','#87CEEB','#808080'] 
    
        
        
    #plt.title(diversity_type, fontsize=18)
    plt.legend()
    plt.savefig(picture_file,dpi=600)
    #all length





 
def sess_num_length_density(sess_num ,sess_length ,session_interval,picture_file):
    pdb.set_trace()
    plt.figure(dpi=120)
    sns.set(style='whitegrid')
    color_choose=[ '#FFFF00','#87CEEB','#0000FF','#800080','#006400','#FFFF00','#FFA500',"#01a2d9","#dc2624",'#098154' ,'#FF0000','#87CEEB','#808080'] 
    for key,values in tqdm(sess_length.items()) :
        sns.distplot( values ,  hist=True, kde_kws={'linestyle':'--','linewidth':'1' }, color=color_choose.pop(), label=key)   #kde_kws设置外框线属性   
     
    x_major_locator=MultipleLocator(5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 50)
    plt.legend()
    plt.savefig(os.path.join(picture_file,'sess_length_density'),dpi=600)               

    plt.figure(dpi=120)
    sns.set(style='whitegrid')
    color_choose=[ '#FFFF00','#87CEEB','#0000FF','#800080','#006400','#FFFF00','#FFA500',"#01a2d9","#dc2624",'#098154' ,'#FF0000','#87CEEB','#808080'] 
    for key,values in tqdm(sess_num.items()) :
        sns.distplot( values , bins=10,  hist=True, kde_kws={'linestyle':'--','linewidth':'1' }, color=color_choose.pop(), label=key)   #kde_kws设置外框线属性   
     
    x_major_locator=MultipleLocator(5)
    ax=plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    plt.xlim(0, 20)
    plt.legend()
    plt.savefig(os.path.join(picture_file,'sess_num_density'),dpi=600)  

def sess_num_length_box(sess_num ,sess_length ,session_interval,picture_file):

    sess_num_df = pd.DataFrame( columns=["num","interval"])
    sess_length_df = pd.DataFrame( columns=["length","interval"])
    for key,values in tqdm(sess_num.items()) :
        interval_df = pd.DataFrame( columns=["num","interval"])
        interval_df['num']=values
        interval_df['interval']=key
        sess_num_df = sess_num_df.append(interval_df)
    for key,values in tqdm(sess_length.items()) :
        interval_df = pd.DataFrame( columns=["length","interval"])
        interval_df['length']=values
        interval_df['interval']=key
        sess_length_df = sess_length_df.append(interval_df)

       
    plt.figure(dpi=120)
    sns.set_style('whitegrid')
    
    sns.boxplot( data= sess_num_df,x='interval',y='num'  ,palette="ch:rot=-.5,d=.3_r"  )   #kde_kws设置外框线属性   
  
    plt.title('session_num', fontsize=18)
    plt.savefig(os.path.join(picture_file,'sess_num_box'),dpi=600)  

    plt.figure(dpi=120)
    sns.set_style('whitegrid')
    
    sns.boxplot( data= sess_length_df,x='interval',y='length'  ,palette="ch:rot=-.5,d=.3_r"  )   #kde_kws设置外框线属性   
    plt.ylim(0, 30)
    plt.title('session_length', fontsize=18)
    plt.savefig(os.path.join(picture_file,'sess_length_box'),dpi=600)  







         
     
    

                




#1.基本信息
dataset = "taobao"
#plot_interval_key=['90min','all'] 
 
session_all_interval_file = os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset,'session_all_interval_information')
session_interval=['30','90', '180', '360','720'] 
#session_interval=['30','60','90','120','150','180','210','240','270','300'] 
diversity_file = []
for interval in session_interval:
    diversity_file.append(os.path.join(session_all_interval_file,  'diversity_'+interval))
#2.输出文件
#user sessions_diversity sequences_diversity  
picture_hist_length= os.path.join(session_all_interval_file,r'picture_hist_length.png')
picture_hist_num= os.path.join(session_all_interval_file,r'picture_hist_num.png')
# picture_desity_coverage_all =os.path.join(session_all_interval_file,r'picture_desity_coverage_all.png')
# picture_desity_entropy_all =os.path.join(session_all_interval_file,r'picture_desity_entropy_all.png')
# picture_desity_gini_all =os.path.join(session_all_interval_file,r'picture_desity_gini_all.png')
picture_desity_coverage  =os.path.join(session_all_interval_file,r'picture_desity_coverage.png')
picture_desity_entropy  =os.path.join(session_all_interval_file,r'picture_desity_entropy.png')
picture_desity_gini  =os.path.join(session_all_interval_file,r'picture_desity_gini.png')
picture_box =os.path.join(session_all_interval_file,r'picture_box.png')
#3.读取文件 

sess_coverage={}
sess_entropy={}
sess_gini={}
sess_length={}
sess_num={}
for interval in session_interval:
    sess_coverage['coverage_'+interval]=[]
    sess_entropy['entropy_'+interval]=[]
    sess_gini['gini_'+interval]=[]
    sess_length['length_'+interval]=[]
    sess_num['num_'+interval]=[]
for interval in tqdm(session_interval):
    sess_data ,sequence_data, sess_length_data,sequence_length_data, sess_num_data =diversity_col(col_name='coverage',origin_file=os.path.join(session_all_interval_file,  'diversity_'+interval))
    sess_coverage['coverage_'+interval]=sess_data
    sess_data ,sequence_data, sess_length_data,sequence_length_data, sess_num_data =diversity_col(col_name='entropy',origin_file=os.path.join(session_all_interval_file,  'diversity_'+interval))
    sess_entropy['entropy_'+interval]=sess_data
    sess_data ,sequence_data, sess_length_data,sequence_length_data, sess_num_data =diversity_col(col_name='gini',origin_file=os.path.join(session_all_interval_file,  'diversity_'+interval))
    sess_gini['gini_'+interval]=sess_data
    sess_length['length_'+interval] = sess_length_data
    sess_num['num_'+interval] = sess_num_data
for interval in session_interval:
    sess_coverage['coverage_'+interval]

 
#sess_num_length_density(sess_num ,sess_length ,session_interval,session_all_interval_file)
sess_num_length_box(sess_num ,sess_length ,session_interval,session_all_interval_file)
  


# no_length_info_diversity_plot(session_interval ,sess_coverage,picture_desity_coverage_all ,'coverage')
# no_length_info_diversity_plot(session_interval ,sess_entropy,picture_desity_entropy_all,'entropy')
# no_length_info_diversity_plot(session_interval ,sess_gini,picture_desity_gini_all,'gini')
with_length_info_diversity_plot(session_interval,sess_length,sess_coverage,picture_desity_coverage,'coverage')
with_length_info_diversity_plot(session_interval,sess_length,sess_entropy,picture_desity_entropy,'entropy')
with_length_info_diversity_plot(session_interval,sess_length,sess_gini,picture_desity_gini,'gini')










