# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 21:05:14 2017

@author: Administrator
"""

#import tensorflow as tf

name1 = './ieee_zhihu_cup/question_train_set.txt'
name2 = './ieee_zhihu_cup/question_topic_train_set.txt'
name3 = './ieee_zhihu_cup/topic_info.txt'
name4 = './ieee_zhihu_cup/write.txt'
cnt = 0
dic = {}

with open(name3) as file3:
    cnt = 0
    for line in file3:
        dic[line.split()[0]] = cnt
        cnt+=1

with open(name1) as file1,open(name2) as file2,open(name4,'w') as file4:
    cnt = 0
    for line1,line2 in zip(file1,file2):
        cnt+=1 
        if cnt<300000:
            strem = ''
            for item in line2.split()[1].split(','):
                strem = strem + str(dic[item]) +','
            line2.split()[1]
            line = line1.split()[2] + ' ' + strem[:-1] + '\n'
#            print(line)
            file4.write(line)
        else:
            break
