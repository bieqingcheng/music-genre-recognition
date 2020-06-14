from random import shuffle
import random
import pickle
import numpy as np
import os
def write_txt(list0, train_path, test_path, idx, dict0, genre):
    #import pudb;pu.db
    with open(test_path, 'a') as f:
        print(genre,'fold:',idx,len(dict0[idx]))
        for i in range(len(dict0[idx])):
            name = list0[dict0[idx][i]]
            line =genre + '\t' +  name + '\n'
            f.write(line)
    with open(train_path,'a') as f1:
        for i in range(10):
            if i!=idx:
                for j in range(len(dict0[i])):
                    line =genre + '\t' + list0[dict0[i][j]] + '\n'
                    f1.write(line)
def split_10(path):
    wr_root = '/home/zwj/mgr/ballroom_feat/10_split_txt'
    genre = path.split('/')[-1]
    data = pickle.load(open(path,'rb'))
    genre = genre.split('.')[0]
    #import pudb;pu.db
    name_list = list(data[genre].keys())
    name_list.sort()
    random.seed(0)
    aa = list(range(len(name_list)))
    shuffle(aa)
    len_fold = len(name_list)//10
    dict_10 = dict()
    #assert len(name_list) > 10 * len_fold
    for i in range(10):
        dict_10[i]=aa[i*len_fold:(i+1)*len_fold]
    if len(name_list) > 10 * len_fold:
        rest = aa[10*len_fold:]
        len_r = len(rest)
        for u in range(len_r):
            dict_10[u] = dict_10[u] + [rest[u]]
    for i in range(10):
        test_path = os.path.join(wr_root,'test_split_{}.txt'.format(i))
        train_path = os.path.join(wr_root,'train_split_{}.txt'.format(i))
        write_txt(name_list, train_path, test_path, i, dict_10, genre)
            
if __name__ == "__main__":
    root = '/home/zwj/mgr/ballroom_feat/mel'
    dirs = os.listdir(root)
    dirs.sort()
    for i in range(len(dirs)):
        dirpath = os.path.join(root,dirs[i])
        split_10(dirpath)
