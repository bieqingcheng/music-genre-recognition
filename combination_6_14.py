import os
from itertools import combinations
import pickle
import numpy as np


classes=['Chacha','Foxtrot','Jive','Pasodoble','Quickstep','Rumba','Salsa','Samba',
'Slowwaltz','Tango','Viennesewaltz','Waltz','Wcswing']
aa = ['mel','cqt','percussive','harmonic','scatter','transfer','timbre']

classes.sort()
class_to_idx = {classes[i]: i for i in range(len(classes))}

def cal_acc(dict0):
    sum_t = 0 
    total = 0
    for a,b in dict0.items():
        for name,value in b.items():
            total += 1
            if class_to_idx[a] == np.argmax(value):
                sum_t += 1
    acc = float(sum_t) / total * 100
    return acc
                

list_t = list()
for i in range(8):
    if i>1:
        cc = list(combinations(aa,i))
        for i in range(len(cc)):
            list_t.append(cc[i])
print(len(list_t))

for i in range(len(list_t)):
        print(list_t[i])

best_acc = 0.0
best_std = 0.0
bestname = ''
root = '/home/zwj/mgr/extract_coding'

#fold = 0
rank = dict()

for i in range(len(list_t)):
    dirs = list_t[i]
    line = ''
    for j in range(len(dirs)):
        line += dirs[j]+'_'
    rank[line] = np.zeros(10)
    '''
    content = dict()
    for u in classes：
        content[u] = dict()
    '''
    for fold in range(10):
        for j in range(len(dirs)):
            dir0 = dirs[j]
            dirpath0 = os.path.join(root,dir0)
            dirpath = os.path.join(dirpath0,'test')
            file0 = '{}_output_result_{}.pkl'.format(dir0,fold)
            filepath = os.path.join(dirpath,file0)
            a = pickle.load(open(filepath,'rb'))
            if j == 0:
                content = a
            else:
                for key,value in a.items():
                    for cc,dd in value.items():
                        assert dd.shape[0]==13
                        content[key][cc] += dd
        #acc = cal_acc(content)
    
        #import pudb;pu.db
        acc = cal_acc(content)
        rank[line][fold] = acc
    acc1 = np.mean(rank[line])
    std1 = np.std(rank[line])
    if acc1 > best_acc:
        best_acc=acc1
        bestname = line
        best_std = std1

print('best_acc:{}, std:{}, name:{}\n'.format(best_acc, best_std, bestname))

rank_new = dict()
for key,value in rank.items():
    rank_new[key] = np.mean(rank[key])
            
sort0 = sorted(rank_new.items(),key= lambda item:item[1])
len0 = len(sort0)
for i in range(6):
    idx = len0-1 -i
    dx = list(sort0[idx])
    print(dx,'\n')
    for v in range(len(dx)):
        name = dx[0]
    print('{}, mean:{}, std:{}'.format(name,np.mean(rank[name]),np.std(rank[name]))) 

