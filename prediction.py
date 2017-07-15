# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 16:56:47 2017

@author: Administrator
"""
import numpy as np
import tensorflow as tf
import gc
import csv

fileName = './ieee_zhihu_cup/question_eval_set.txt'
fileName1 = './ieee_zhihu_cup/write.txt'

dic = {}
with open('./ieee_zhihu_cup/topic_info.txt') as file3:
    cnt = 0
    for line in file3:
        dic[line.split()[0]] = cnt
        cnt+=1
new_dic = {v:k for k,v in dic.items()} 

#y=[]
x_text=[]
with open(fileName1) as file:
    for item in file:
#        tmp = item.split()[1].split(',')
#        if len(tmp)>5:tmp=tmp[0:5]
#        label = np.zeros(1999)
#        for tmp_item in tmp:
#            label[int(tmp_item)] = 1
#        y.append(label)
        x_text.append(item.split()[0])

max_document_len = 72
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_len)
vocab_processor.fit_transform(x_text)

del x_text
gc.collect()


#x_train,x_dev=x[:270000],x[270000:]
#y_train,y_dev=y[:270000],y[270000:]

input_x = tf.placeholder(tf.int32,[None,72],name='input_x')
input_y = tf.placeholder(tf.float32,[None,1999],name='input_y')
dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

#最长词汇数
sequence_len=72
#分类数
num_classes=1999
#总词汇数
vocab_size=len(vocab_processor.vocabulary_)
#词向量长度
embedding_size=256
#卷积核尺寸3，4，5
filter_sizes=list([3,4,5])
#卷积和数量
num_filters=1024

Weights = tf.Variable(tf.random_uniform([vocab_size,embedding_size],-1.0,1.0),name='Weights')
#shape:[None,sequence_len,embedding_size]
embedded_chars = tf.nn.embedding_lookup(Weights,input_x)
#添加一个纬度，shape:[None,sequence_len,embedding_size，1]
embedded_chars_expanded = tf.expand_dims(embedded_chars,-1)


#######Create a convolution + maxpool layer for each filter size#######
pooled_outputs = []
for i,filter_size in enumerate(filter_sizes):
    with tf.name_scope('conv-maxpool-%s' % filter_size):
        filter_shape = [filter_size,embedding_size,1,num_filters]
        W = tf.Variable(
                tf.truncated_normal(filter_shape,stddev=0.1),name='W')
        b = tf.Variable(
                tf.constant(0.1,shape=[num_filters]),name='b')
        conv = tf.nn.conv2d(
                embedded_chars_expanded,
                W,
                strides=[1,1,1,1],
                padding='VALID',
                name='conv')
        h = tf.nn.relu(tf.nn.bias_add(conv,b),name='relu')
        pooled = tf.nn.max_pool(
                h,
                ksize=[1,sequence_len-filter_size+1,1,1],
                strides=[1,1,1,1],
                padding='VALID',
                name='pool')
        pooled_outputs.append(pooled)
        
num_filters_total = num_filters*len(filter_sizes)
print('num_filters_total:',num_filters_total)
h_pool = tf.concat(pooled_outputs,3)
h_pool_flat = tf.reshape(h_pool,[-1,num_filters_total])



######################ADD dropout##########################
#Add dropout
with tf.name_scope('dropout'):h_drop=tf.nn.dropout(h_pool_flat,dropout_keep_prob)

# Final (unnarmalized) scores and predictions
with tf.name_scope('output'):
    W = tf.get_variable(
            'W',
            shape=[num_filters_total,num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.constant(0.1,shape=[num_classes]), name='b')
    scores = tf.nn.xw_plus_b(h_drop,W,b,name='scores')
    
##定义loss
#with tf.name_scope('loss'):
#    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=input_y))
#    
##定义优化器
#with tf.name_scope('optimizer'):
#    optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
#
#
## 生成批次数据
#def batch_iter(data, batch_size, num_epochs, shuffle=False):
#    """
#    Generate a batch iterator for a dataset
#    """
#    data = np.array(data)
#    data_size = len(data)
#    # 每个epoch的num_batch
#    num_batches_per_epoch = int((len(data)-1)/batch_size)+1
#    print('num_batches_per_epoch:',num_batches_per_epoch)
#    for epoch in range(num_epochs):
#        # Shuffle the data at each epoch
#        if shuffle:
#            shuffle_indices = np.random.permutation(np.arange(data_size))
#            shuffled_data = data[shuffle_indices]
#        else:
#            shuffled_data = data
#            for batch_num in range(num_batches_per_epoch):
#                start_index = batch_num * batch_size
#                end_index = min((batch_num +1)*batch_size,data_size)
#                yield shuffled_data[start_index:end_index]
                
                
## 知乎提供的评测方案
#def eval(predict_label_and_marked_label_list):
#    """
#    :param predict_label_and_marked_label_list:一个元祖列表。例如
#    [([1,2,3,4,5],[4,5,6,7]),
#    ([3,2,1,4,7],[5,7,3])
#    ]
#    需要注意这里 predict_label 是去重复的，例如[1,2,3,2,4,1,5],去重后变成[1,2,3,4,6]
#    
#    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
#    [0,0,0,1,1] (4,5命中)
#    [1,0,0,0,1] (3,7命中)
#    
#    """
#    right_label_num = 0 #总命中标签数量
#    right_label_at_pos_num = [0,0,0,0,0] #在各个位置上总命中数量
#    sample_num = 0 #总问题数量
#    all_marked_label_num = 0 #总标签数量
#    for predict_labels,marked_labels in predict_label_and_marked_label_list:
#        sample_num += 1
#        marked_label_set = set(marked_labels)
#        all_marked_label_num += 1
#        for pos,label in zip(range(0,min(len(predict_labels),5)), predict_labels):
#            if label in marked_label_set:
#                right_label_num += 1
#                right_label_at_pos_num[pos] += 1
#
#    precision = 0.0
#    for pos,right_num in zip(range(0,5), right_label_at_pos_num):
#        precision += ((right_num/float(sample_num))) / math.log(2.0 + pos) #下标0-4映射到posl+1,所以最终+2
#    recall = float(right_label_num) / all_marked_label_num
#                  
#    return 2*(precision * recall) / (precision + recall)


# 定义saver,只保存最新的5个模型
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

with tf.Session() as sess:
    predict_top_5 = tf.nn.top_k(scores, k=5)
    label_top_5 = tf.nn.top_k(input_y, k=5)
    ckpt = tf.train.get_checkpoint_state('models')
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess,ckpt.model_checkpoint_path)
        i_tmp = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print('i_tmp:',i_tmp)
    else:
        sess.run(tf.global_variables_initializer())
        i_tmp= 0
        
#    #生成数据
#    batches = batch_iter(
#            list(zip(x_train, y_train)), 64, 10)
    with open(fileName) as file,open('predict.csv','w', newline='') as csv_file:
        cnt = 0
        writer = csv.writer(csv_file)
        for item in file:
            cnt += 1
            x = next(vocab_processor.transform(item.split()[2].split())).tolist()     
            # 得到一个batch的数据
            x_batch = np.reshape(np.array(x),(-1,len(x)))
            
            predict_5 = sess.run(predict_top_5,feed_dict={input_x:x_batch,dropout_keep_prob:1.0})
            lst = list([item.split()[0]])
            for item in predict_5[1][:5][0]:
                lst.append(new_dic[item])
            writer.writerow(lst)
            if cnt%2000 == 0 :
                print(cnt)


            
#            print('label:',label_5[1][:5])
#            print('predict:',predict_5[1][:5])
#            print('predict:',predict_5[0][:5])
#            print('loss:',_loss)
#            predict_label_and_marked_label_list = []
#            for predict,label in zip(predict_5[1],label_5[1]):
#                predict_label_and_marked_label_list.append((list(predict),list(label)))
#            score = eval(predict_label_and_marked_label_list)
#            print('score:',score)
            

          