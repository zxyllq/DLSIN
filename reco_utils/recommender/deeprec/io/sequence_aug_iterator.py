 # Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import json
import pickle as pkl
import random
import os
import time

from reco_utils.recommender.deeprec.io.iterator import BaseIterator
from reco_utils.recommender.deeprec.deeprec_utils import load_dict
import pdb 

__all__ = ["SequenceAugIterator"  ]

class SequenceAugIterator(BaseIterator):
    def __init__(self, hparams, graph, col_spliter="\t"):
        """Initialize an iterator. Create necessary placeholders for the model.
        
        Args:
            hparams (obj): Global hyper-parameters. Some key settings such as #_feature and #_field are there.
            graph (obj): the running graph. All created placeholder will be added to this graph.
            col_spliter (str): column spliter in one line.
        """
        self.col_spliter = col_spliter
      

        self.max_seq_length = hparams.max_seq_length
        self.max_session_count =hparams.max_session_count
        self.batch_size = hparams.batch_size
        self.augment_rate =hparams.augment_rate
        self.train_iter_data = dict()

       # self.time_unit = hparams.time_unit

        self.graph = graph
        with self.graph.as_default():
            self.labels = tf.placeholder(tf.float32, [None, 1], name="label")
            self.users = tf.placeholder(tf.int32, [None], name="users")
            self.index = tf.placeholder(tf.float32, [None], name="index")
            self.items = tf.placeholder(tf.int32, [None], name="items")
            self.cates = tf.placeholder(tf.int32, [None], name="cates")
            self.behaviors = tf.placeholder(tf.int32, [None], name="behaviors")
            


            
            self.item_sequence = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="item_sequence"
            )
            self.item_cate_sequence = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="item_cate_sequence"
            )
            self.sequence_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="sequence_mask"
            )

            self.item_sequence_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="item_sequence_1"
            )
            self.item_cate_sequence_1 = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="item_cate_sequence_1"
            )
            self.sequence_1_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="sequence_1_mask"
            )
            self.item_sequence_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="item_sequence_2"
            )
            self.item_cate_sequence_2 = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="item_cate_sequence_2"
            )
            self.sequence_2_mask = tf.placeholder(
                tf.int32, [None, self.max_seq_length*self.max_session_count], name="sequence_2_mask"
            )
             


             

            

    def parse_file(self, input_file):
        """Parse the file to a list ready to be used for downstream tasks
        
        Args:
            input_file: One of train, valid or test file which has never been parsed.
        
        Returns: 
            list: A list with parsing result
        """
        with open(input_file, "r") as f:
            lines = f.readlines()
        res = []
        for line in lines:
            if not line:
                continue
            res.append(self.parser_one_line(line))
        return res

    def parser_one_line(self, line):
        """Parse one string line into feature values.
            a line was saved as the following format:
            label \t index \t user_hash \t item_hash \t item_cate \t behavior_type\t sess_state \t 
            item_session_succ  \t item_cate_session_succ \t behavior_type_session_succ \t sess_state_session_succ \t
            item_session_0  \t item_cate_session_0 \t behavior_type_session_0 \t sess_state_session_0 \t
            item_session_1  \t item_cate_session_1 \t behavior_type_session_1 \t sess_state_session_1 \t
            item_session_2  \t item_cate_session_2 \t behavior_type_session_2 \t sess_state_session_2 \t
            item_session_3  \t item_cate_session_3 \t behavior_type_session_3 \t sess_state_session_3 \t
            item_session_4  \t item_cate_session_4 \t behavior_type_session_4 \t sess_state_session_4 \t
            valid_sess \n[1,2,3,4]

        Args:
            line (str): a string indicating one instance

        Returns:
            tuple/list: Parsed results including label, user_id, target_item_id, target_category, item_history, cate_history(, timeinterval_history,
            timelast_history, timenow_history, mid_mask, seq_len, learning_rate)

        """

        words = line.strip().split(self.col_spliter)
        label = float(words[0])
       # index = float(words[1])#用于记录loss的变化
        user_id =  int(words[2])
        item_id = int(words[3])
        item_cate = int(words[4])
        item_ts = float(words[5])
        item_behavior = int(words[6])
       

        item_sequence_words = words[7].strip('[').strip(']').split(", ")
        cate_sequence_words = words[8].strip('[').strip(']').split(", ")
        item_sequence    = [int(i) for i in item_sequence_words]
        cate_sequence   = [int(i) for i in cate_sequence_words ]


         

        
       
 

        return (
            label,
        
            user_id,
            item_id,
            item_cate,
            item_ts,
            item_behavior,

            item_sequence  ,
            cate_sequence,
           
         
        )

     

     

    def load_data_from_file(self, infile, batch_num_ngs=0, min_seq_length=1):
        """Read and parse data from a file.
        
        Args:
            infile (str): Text input file. Each line in this file is an instance.
            batch_num_ngs (int): The number of negative sampling here in batch. 
                0 represents that there is no need to do negative sampling here.
            min_seq_length (int): The minimum number of a sequence length. 
                Sequences with length lower than min_seq_length will be ignored.

        Returns:
            obj: An iterator that will yields parsed results, in the format of graph feed_dict.
        """
       
        label_list = []
        user_list = []
        item_list = []
        item_cate_list = []
        item_ts_list =[]
        item_behavior_list =[]
         
        item_sequence_batch = []
        item_cate_sequence_batch = []
       


        cnt = 0

        if infile not in self.train_iter_data :
            lines = self.parse_file(infile)
            self.train_iter_data[infile] = lines
        else:
            lines = self.train_iter_data[infile]

        if batch_num_ngs > 0:
            random.shuffle(lines)

        for line in lines:
            if not line:
                continue

            (
                label,
         
                user_id,
                item_id,
                item_cate,
                item_ts,
                item_behavior,
                
                item_sequence,
                item_cate_sequence,
               
              

                 
            ) = line
           

            label_list.append(label)
            
            user_list.append(user_id)
            item_list.append(item_id)
            item_cate_list.append(item_cate)
            item_ts_list.append(item_ts)
            item_behavior_list.append(item_behavior)
            
            item_sequence_batch.append(item_sequence)
            item_cate_sequence_batch.append(item_cate_sequence)
           
            cnt += 1
            if cnt == self.batch_size:
                res = self._convert_data(
                    label_list,
             
                    user_list,
                    item_list,
                    item_cate_list,
                    item_ts_list,
                    item_behavior_list,
                   
                    item_sequence_batch,
                    item_cate_sequence_batch,
                    
                
                    batch_num_ngs,
                )
                batch_input = self.gen_feed_dict(res)
                yield batch_input if batch_input else None
                label_list = []
                
                user_list = []
                item_list = []
                item_cate_list = []
                item_ts_list =[]
                item_behavior_list =[]
                
                item_sequence_batch = []
                item_cate_sequence_batch = []
                
                cnt = 0
        if cnt > 0:
            res = self._convert_data(
                label_list,
                 
                    user_list,
                    item_list,
                    item_cate_list,
                     item_ts_list,
                    item_behavior_list,
                   
                    item_sequence_batch,
                    item_cate_sequence_batch,
                    
                batch_num_ngs,
            )
            batch_input = self.gen_feed_dict(res)
            yield batch_input if batch_input else None

     
 
    def _convert_data(
        self,
       label_list,
       
        user_list,
        item_list,
        item_cate_list,
        item_ts_list,
        item_behavior_list,
        
        item_sequence_batch,
        item_cate_sequence_batch,
        
        batch_num_ngs,
    ):
        """Convert data into numpy arrays that are good for further model operation.
        
        Args:
            label_list (list): a list of ground-truth labels.
            user_list (list): a list of user indexes.
            item_list (list): a list of item indexes.
            item_cate_list (list): a list of category indexes.
            item_history_batch (list): a list of item history indexes.
            item_cate_history_batch (list): a list of category history indexes.
            time_list (list): a list of current timestamp.
            time_diff_list (list): a list of timestamp between each sequential opertions.
            time_from_first_action_list (list): a list of timestamp from the first opertion.
            time_to_now_list (list): a list of timestamp to the current time.
            batch_num_ngs (int): The number of negative sampling while training in mini-batch.

        Returns:
            dict: A dictionary, contains multiple numpy arrays that are convenient for further operation.
        """
        if batch_num_ngs:
            instance_cnt = len(label_list)
            if instance_cnt < 5:
                return
            # 标量 变量
            label_list_all = []
            item_list_all = []
            item_cate_list_all = []
            #标量 不变量
            user_list_all = np.asarray(
                [[user] * (batch_num_ngs + 1) for user in user_list], dtype=np.int32
            ).flatten()
            
            behavior_list_all = np.asarray(
                [[ behavior] * (batch_num_ngs + 1) for  behavior in  item_behavior_list], dtype=np.int32
            ).flatten()
        

            # history_lengths = [len(item_history_batch[i]) for i in range(instance_cnt)]#标量 不变量，后面应该要扩展
            max_seq_length  = self.max_seq_length
            max_session_count= self.max_session_count#4
            # 向量 list 不变量 先扩展位置，填充0

            item_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length *max_session_count)
            ).astype("int32") 


            item_sequence_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("int32")
            item_cate_sequence_1_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length *max_session_count)
            ).astype("int32") 
            
            item_sequence_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length*max_session_count )
            ).astype("int32")
            item_cate_sequence_2_batch_all = np.zeros(
                (instance_cnt * (batch_num_ngs + 1), max_seq_length *max_session_count)
            ).astype("int32") 
            

           



            #mask也是list
            
            mask_sequence = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length *max_session_count)
            ).astype("float32")
            mask_sequence_1 = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length *max_session_count)
            ).astype("float32")
            mask_sequence_2 = np.zeros(
                (instance_cnt * (1 + batch_num_ngs), max_seq_length *max_session_count)
            ).astype("float32")
            

            #先处理历史序列
            
            for i in range(instance_cnt):
                #已经做过截断了，这里直接用
                    
                #sequence 
                
                sequence_length = len(item_sequence_batch[i])
                remove_index_1 = random.sample(list(range(sequence_length)),k=int(np.ceil(self.augment_rate*sequence_length)))
                remove_index_2 = random.sample(list(range(sequence_length)),k=int(np.ceil(self.augment_rate*sequence_length)))
                item_sequence_1   =  [ item_sequence_batch[i][index] if index not in remove_index_1 else 0  for index in range(sequence_length)   ] 
                item_cate_sequence_1   =  [item_cate_sequence_batch[i][index]  if index not in remove_index_1 else 0 for index in range(sequence_length)]

                item_sequence_2 =  [item_sequence_batch[i][index] if index not in remove_index_2 else 0 for index in range(sequence_length) ]
                item_cate_sequence_2   =  [item_cate_sequence_batch[i][index] if index not in remove_index_2 else 0 for index in range(sequence_length)  ]


                 
                for index in range(batch_num_ngs + 1):
                    item_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(item_sequence_batch[i] , dtype=np.int32)
                    item_cate_sequence_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        item_cate_sequence_batch[i] , dtype=np.int32
                    )

                    item_sequence_1_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(item_sequence_1  , dtype=np.int32)
                    item_cate_sequence_1_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        item_cate_sequence_1  , dtype=np.int32
                    )

                    item_sequence_2_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(item_sequence_2  , dtype=np.int32)
                    item_cate_sequence_2_batch_all[
                        i * (batch_num_ngs + 1) + index, :sequence_length
                    ] = np.asarray(
                        item_cate_sequence_2  , dtype=np.int32
                    )

                    mask_sequence[i * (batch_num_ngs + 1) + index, :sequence_length] = 1.0
                    mask_sequence_1[i * (batch_num_ngs + 1) + index, :sequence_length] = 1.0
                    mask_sequence_2[i * (batch_num_ngs + 1) + index, :sequence_length] = 1.0
                    

            #处理负采样
            for i in range(instance_cnt):
                positive_item = item_list[i]
                label_list_all.append(1)
                item_list_all.append(positive_item)
                item_cate_list_all.append(item_cate_list[i])
                count = 0
                while batch_num_ngs:
                    random_value = random.randint(0, instance_cnt - 1)
                    negative_item = item_list[random_value]
                    if negative_item == positive_item:
                        continue
                    label_list_all.append(0)
                    item_list_all.append(negative_item)
                    item_cate_list_all.append(item_cate_list[random_value])
                    count += 1
                    if count == batch_num_ngs:
                        break

            res = {}
           
           
            res["labels"] = np.asarray(label_list_all, dtype=np.float32).reshape(-1, 1)
             
            res["users"] = np.asarray(user_list_all, dtype=np.int32)
            res["behaviors"] = np.asarray(behavior_list_all, dtype=np.int32)
            res["items"] = np.asarray(item_list_all, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list_all, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_1"] = item_sequence_1_batch_all
            res["item_cate_sequence_1"] = item_cate_sequence_1_batch_all
            res["item_sequence_2"] = item_sequence_2_batch_all
            res["item_cate_sequence_2"] = item_cate_sequence_2_batch_all
 

            res["sequence_mask"] =mask_sequence
            res["sequence_1_mask"] =mask_sequence_1
            res["sequence_2_mask"] =mask_sequence_2
            
          


             


        else:
            #测试和验证
            instance_cnt = len(label_list)
             
            max_seq_length  = self.max_seq_length
            max_session_count =self.max_session_count


            item_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            item_cate_sequence_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32") 
            mask_sequence = np.zeros(
                (instance_cnt, max_seq_length* max_session_count )
            ).astype("int32")    


            item_sequence_1_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            item_cate_sequence_1_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            mask_sequence_1 = np.zeros(
                (instance_cnt, max_seq_length* max_session_count )
            ).astype("int32")   



            item_sequence_2_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            item_cate_sequence_2_batch_all = np.zeros(
                (instance_cnt  , max_seq_length* max_session_count )
            ).astype("int32")
            mask_sequence_2 = np.zeros(
                (instance_cnt, max_seq_length* max_session_count )
            ).astype("int32")    



            


             

           

   
            for i in range(instance_cnt):
                #sequence
                sequence_length = len(item_sequence_batch[i])
                item_sequence_batch_all[i, :sequence_length] = item_sequence_batch[i] 
                item_cate_sequence_batch_all[i, :sequence_length] = item_cate_sequence_batch[i] 
                

                mask_sequence[i, :sequence_length] = 1.0
                
                
     
            
            res = {}
           
            res["labels"] = np.asarray(label_list, dtype=np.float32).reshape(-1, 1)
            
            res["users"] = np.asarray(user_list, dtype=np.int32)
            res["behaviors"] = np.asarray(item_behavior_list, dtype=np.int32)
            res["items"] = np.asarray(item_list, dtype=np.int32)
            res["cates"] = np.asarray(item_cate_list, dtype=np.int32)

            res["item_sequence"] = item_sequence_batch_all
            res["item_cate_sequence"] = item_cate_sequence_batch_all
            res["item_sequence_1"] = item_sequence_1_batch_all
            res["item_cate_sequence_1"] = item_cate_sequence_1_batch_all
            res["item_sequence_2"] = item_sequence_2_batch_all
            res["item_cate_sequence_2"] = item_cate_sequence_2_batch_all
            
            res["sequence_mask"] =mask_sequence
            res["sequence_1_mask"] =mask_sequence_1
            res["sequence_2_mask"] =mask_sequence_2
            
            

             


        return res

         
    def gen_feed_dict(self, data_dict):
        """Construct a dictionary that maps graph elements to values.
        
        Args:
            data_dict (dict): a dictionary that maps string name to numpy arrays.

        Returns:
            dict: a dictionary that maps graph elements to numpy arrays.

        """
        if not data_dict:
            return dict()
        feed_dict = {
            self.labels: data_dict["labels"],
            #self.index: data_dict["index"],
            self.users: data_dict["users"],
            self.items: data_dict["items"],
            self.cates: data_dict["cates"],
            self.behaviors:data_dict["behaviors"],
            self.item_sequence: data_dict["item_sequence"],
            self.item_cate_sequence: data_dict["item_cate_sequence"],
            self.item_sequence_1: data_dict["item_sequence_1"],
            self.item_cate_sequence_1: data_dict["item_cate_sequence_1"],
            self.item_sequence_2: data_dict["item_sequence_2"],
            self.item_cate_sequence_2: data_dict["item_cate_sequence_2"],
            
            self.sequence_mask : data_dict["sequence_mask"],
            self.sequence_1_mask : data_dict["sequence_1_mask"],
            self.sequence_2_mask : data_dict["sequence_2_mask"],
           


            
           
        }
        return feed_dict


 
