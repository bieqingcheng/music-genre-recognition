import os
import numpy as np
import pickle

classes = ['Chacha','Foxtrot','Jive','Pasodoble','Quickstep','Rumba','Salsa','Samba',
'Slowwaltz','Tango','Viennesewaltz','Waltz','Wcswing']
classes.sort()
class_to_idx = {classes[i]: i for i in range(len(classes))}
idx_to_class = { i:classes[i] for i in range(len(classes))}

def cal_precision_recall_F1(M):
    n = len(M)
    M1 = np.zeros((15,14))
    M1[:n,:n]= M
    for i in range(n):
        rowsum, colsum = sum(M[i,:]), sum(M[r][i] for r in range(n))
        precision = M[i][i]/float(colsum)
        recall = M[i][i]/float(rowsum)
        F1 = 2*(precision*recall/(precision+recall))
        M1[i,13] = recall
        M1[13,i] = precision
        M1[14,i] = F1
        #M1[9,i] = acc[i]
    return M1

def combine_10_fold(path,dict0):
    out_test = pickle.load(open(path,'rb'))
    for a,b in out_test.items():
        for name,value in b.items():
            predict = np.argmax(value)
            idx = class_to_idx[a]
            dict0[idx,predict] += 1
def cal_col(idx,M):
    sum_t = 0
    for i in range(13):
        if i!=idx:
            sum_t += M[i,idx]
    return sum_t

def cal_row(idx,M):
    sum_t = 0
    for i in range(13):
        if i!=idx:
            sum_t += sum(M[i,:13])-M[i,i]
    return sum_t
    

if __name__ == '__main__':
    root = '/home/zwj/mgr/extract_coding/st/test/st_output_result_{}.pkl'
    dict0 = np.zeros((13,13))
    for i in range(10):
        path = root.format(i)
        combine_10_fold(path,dict0)
    print(dict0.sum(axis=1))
    print(dict0)
    print('\n')
    print(classes)
    Ma = cal_precision_recall_F1(dict0)
    row,col = Ma.shape
    Mb = np.zeros((15,14))
    for i in range(row):
        for j in range(col):
            if i< 13 and j<13:
                print('{} '.format(int(Ma[i,j])),end='')
                Mb[i,j] = int(Ma[i,j])
            else:
                print("%.4f "%Ma[i,j],end='')
                Mb[i,j] = round(Ma[i,j],4)
        print('\n')
    np.savetxt('temp_confusion.csv',Mb, delimiter = ",")    
    ac = np.zeros(13)
    tt = 0
    for i in range(13):
        tt += Ma[i,i]
    for i in range(13):
        a1 = cal_col(i,Ma[:13,:13])
        b1 = cal_row(i,Ma[:13,:13])
        c1 = tt/(tt + a1 +b1)
        ac[i] = c1
        print('col:{},row:{},tt:{}'.format(a1,b1,tt))
    print(ac)
    ac = ac.reshape(1,ac.shape[0])
    np.savetxt('temp_ac.csv',ac, delimiter = ",")
