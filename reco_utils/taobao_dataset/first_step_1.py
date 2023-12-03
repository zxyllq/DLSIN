import os
from datetime import datetime
import pandas as pd
import numpy as np
import pdb 
from tqdm import tqdm
from multiprocessing import Process
#logger = logging.getLogger()
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


 


#1.基本信息
dataset = "taobao"
 
session_reviews_name = 'new_UserBehavior_with_session_state'
sequence_reviews_name = 'new_UserBehavior_with_session_state_for_sequence_model'
reviews_path=os.path.join("..", "..", "tests", "resources", "deeprec", "sequential",dataset) 
noise_rate  = 0.4

#2.输出文件
#uid	iid	cid	ts	behavior_type		state  session_state_xx
new_session_reviews= os.path.join(reviews_path, str(noise_rate)+'_'+'new_UserBehavior_with_session_state')
new_sequence_reviews = os.path.join(reviews_path, str(noise_rate)+'_'+'new_UserBehavior_with_session_state_for_sequence_model' )
#3.读取文件并过滤

#100150807 rows  其中10434543 是其他行为（fav=2888258 ,cart=5530446 ,buy=2015839）
session_reviews = pd.read_csv( os.path.join(reviews_path,session_reviews_name), header=None, names=['ref_index'	,'uid',	'iid',	'cid',	'ts',	'behavior_type',	'session_state_30',\
                           'session_state_90'	,'session_state_120',	'session_state_150',	'session_state_180'	,'session_state_210',	'session_state_240',	'session_state_270',	'session_state_300'	,\
                           'session_state_330',	'session_state_360'	,'session_state_390',	'session_state_420'	,'session_state_450',	'session_state_480',	'session_state_510'	,'session_state_540',\
                           'session_state_600'	,'session_state_720',	'session_state_960'	,'state'])

sequence_reviews = pd.read_csv( os.path.join(reviews_path,sequence_reviews_name), header=None, names=['ref_index'	,'uid',	'iid',	'cid',	'ts',	'behavior_type',	'session_state_30',\
                           'session_state_90'	,'session_state_120',	'session_state_150',	'session_state_180'	,'session_state_210',	'session_state_240',	'session_state_270',	'session_state_300'	,\
                           'session_state_330',	'session_state_360'	,'session_state_390',	'session_state_420'	,'session_state_450',	'session_state_480',	'session_state_510'	,'session_state_540',\
                           'session_state_600'	,'session_state_720',	'session_state_960'	,'state'])

 
 
 
 








