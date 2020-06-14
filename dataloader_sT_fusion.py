import pickle
import os
import csv
import numpy as np
from config import Config
from torch.utils.data import DataLoader, Dataset
import torch
import torchvision.transforms as transforms

class my_dataset(Dataset):
    def __init__(self, melspectrogram, classes, name_list, mode='train'):
        self.melspectrogram = melspectrogram
        classes, class_to_idx,idx_to_class = self.find_classes(classes)
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        self.name_list = name_list


    def find_classes(self, dir):
        classes = dir
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        idx_to_class = {i: classes[i] for i in range(len(classes))}
        return classes, class_to_idx,idx_to_class


    def __getitem__(self, index):
        
        index1 = index 
        #remainder = index % 3
        name = self.name_list[index1].strip('\n')
        a,b,c = name.split('.')
        genre = a
        name = b +'.' + c

        melspectrogram_data = self.melspectrogram[genre][name]
        #melspectrogram_data[melspectrogram_data == -float('inf')] = -18.4207
        melspectrogram_data = torch.FloatTensor(melspectrogram_data)
        #melspectrogram_data = melspectrogram_data.unsqueeze(0)
        #import pudb;pu.db
        target = self.class_to_idx[genre]
        '''
        m_mean = melspectrogram_data.mean(dim = 0)#(dim =0 )
        m_std = melspectrogram_data.std(dim=0)#(dim=0)
        melspectrogram_data = (melspectrogram_data - m_mean) / m_std
        melspectrogram_data = melspectrogram_data.transpose(1,0)  #(647,544)
        '''
        name = genre+ '-' +name

        return (name,melspectrogram_data,target)

    def __len__(self):
        #return len(self.pitch['melody'])
        return  len(self.name_list)

class datatype(object):
    def __init__(self):
        self.classes = Config["normal_config"]["classes"]
        self.classes.sort()
        #self.feat = self.load()
        self.fold = Config["normal_config"]['fold_index']#0
        self.train_list,self.test_list = self.name_txt()
        #self.dirs = ['scatter', 'transfer']
        self.dirs = ['scatter','scatter','scatter','transfer','transfer']
        self.train_feat,self.test_feat = self.fold_10cv()
    
    def instance_a_loader(self, t="train"):
        if t == "train":
            shuffle = True
            dataset = my_dataset(self.train_feat,self.classes,self.train_list,mode=t)
        else:
            shuffle = False
            dataset = my_dataset(self.test_feat,self.classes,self.test_list,mode=t)

        return DataLoader(dataset, batch_size=16,shuffle=shuffle, num_workers=Config["normal_config"]["num_workers"])
    
    def load(self):
        #root = '/home/zwj/mgr/ballroom_feat/mel'
        #root = '/home/zwj/mgr/ballroom_feat/percussive'
        #root = '/home/zwj/mgr/ballroom_feat/harmonic'
        root = '/home/zwj/mgr/ballroom_feat/transfer_feat/transfer_12345.pkl'
        #root = '/home/zwj/mgr/ballroom_feat/transfer/transfer_1234.pkl'
        #root = '/home/zwj/mgr/ballroom_feat/transfer/transfer_123.pkl'
        content = pickle.load(open(root,'rb'))
        return content

    def name_txt(self):
        txt_root = '/home/zwj/mgr/ballroom_feat/10_split_txt'
        train_txt = os.path.join(txt_root,'train_split_{}.txt'.format(self.fold))
        test_txt = os.path.join(txt_root,'test_split_{}.txt'.format(self.fold))
        train = list()
        with open(train_txt) as f_path:
            for line in  f_path.readlines():
                line = line.strip('\n')
                genre, name = line.split('\t')
                name = genre + '.' + name
                train.append(name)
        test = list()
        with open(test_txt) as f:
            for line in  f.readlines():
                line = line.strip('\n')
                genre, name = line.split('\t')
                name = genre + '.' + name
                test.append(name)
        train.sort()
        test.sort()
        return train,test
    def fold_10cv(self):
        path_r = '/home/zwj/mgr/extract_coding'
        dirs = self.dirs
        for i in range(len(dirs)):
            dir0 = dirs[i]
            path_dir = os.path.join(path_r,dir0)
            path_train = os.path.join(path_dir,'train')
            path_test = os.path.join(path_dir,'test')
            file_name = '{}_extract_feature_{}.pkl'.format(dir0,self.fold)
            train_path = os.path.join(path_train,file_name)
            test_path = os.path.join(path_test,file_name)
            train_f = pickle.load(open(train_path,'rb'))
            test_f = pickle.load(open(test_path,'rb'))
            if not dir0.startswith('tran'):
                for a,b in train_f.items():
                    for name,value in b.items():
                        train_f[a][name] = value/3
                for a,b in test_f.items():
                    for name,value in b.items():
                        test_f[a][name] = value/3
            if i==0:
                train_feat = train_f
                test_feat = test_f
            else:
                for a,b in train_f.items():
                    for name,value in b.items():
                        value = value.reshape(1,value.shape[0])
                        if train_feat[a][name].shape[0] == 1024:
                            cc = train_feat[a][name]
                            train_feat[a][name] = cc.reshape(1,cc.shape[0])
                        train_feat[a][name] = np.concatenate([train_feat[a][name],value],axis=0)

                for a,b in test_f.items():
                    for name,value in b.items():
                        value = value.reshape(1,value.shape[0])
                        if test_feat[a][name].shape[0] == 1024:
                            cc = test_feat[a][name]
                            test_feat[a][name] = cc.reshape(1,cc.shape[0])
                        test_feat[a][name] = np.concatenate([test_feat[a][name],value],axis=0)
                

        return train_feat,test_feat

if __name__ == '__main__':
    batch = datatype()
    train_dataloader = batch.instance_a_loader(t='train')
    test_dataloader = batch.instance_a_loader(t='test')
    for batch_idx, ss in enumerate(train_dataloader):
        #import pudb;pu.db
        #print(batch_idx,ss[0])
        print('{}/{} \n'.format(batch_idx,len(ss)))

    for batch_idx, ss in enumerate(test_dataloader):
        #import pudb;pu.db
        print(batch_idx,ss[1].shape)
        #print('{}/{} \n'.format(batch_idx,len(ss)))

