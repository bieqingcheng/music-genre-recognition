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
        
        index1 = index // 3
        remainder = index % 3
        name = self.name_list[index1].strip('\n')
        a,b,c = name.split('.')
        genre = a
        name = b +'.' + c

        melspectrogram_data = self.melspectrogram[genre][name][remainder]
        melspectrogram_data[melspectrogram_data == -float('inf')] = -18.4207
        melspectrogram_data = torch.FloatTensor(melspectrogram_data)

        #import pudb;pu.db
        target = self.class_to_idx[genre]
        '''
        m_mean = melspectrogram_data.mean(dim = 0)#(dim =0 )
        m_std = melspectrogram_data.std(dim=0)#(dim=0)
        melspectrogram_data = (melspectrogram_data - m_mean) / m_std
        melspectrogram_data = melspectrogram_data.transpose(1,0)  #(647,544)
        '''
        name = genre+ '-' +name + '-' +str(remainder)

        return (name,melspectrogram_data,target)

    def __len__(self):
        #return len(self.pitch['melody'])
        return  len(self.name_list) * 3

class datatype(object):
    def __init__(self):
        self.classes = Config["normal_config"]["classes"]
        self.classes.sort()
        self.feat = self.load()
        self.fold = Config["normal_config"]['fold_index']#0
        self.train_list,self.test_list = self.name_txt()
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
        root = '/home/zwj/mgr/ballroom_feat/harmonic'
        #root = '/home/zwj/mgr/ballroom_feat/cqt'
        dirs = os.listdir(root)
        dirs.sort()
        content = dict()
        for i in range(len(dirs)):
            filename = dirs[i]
            genre = filename.split('.')[0]
            filepath = os.path.join(root,filename)
            data = pickle.load(open(filepath,'rb'))
            content[genre] = data[genre]
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
        feat = self.feat
        classes = self.classes
        classes.sort()
        train_data = dict()
        test_data = dict()
        for i in range(len(classes)):
            train_data[classes[i]] = dict()
            test_data[classes[i]] = dict()
        train_list = self.train_list
        for i in range(len(train_list)):
            a,b,c= train_list[i].split('.')
            genre = a
            name = b +'.'+c
            train_data[genre][name] = feat[genre][name]

        test_list = self.test_list
        for i in range(len(test_list)):
            a,b,c= test_list[i].split('.')
            genre = a
            name = b +'.'+c
            test_data[genre][name] = feat[genre][name]
        return train_data,test_data

if __name__ == '__main__':
    batch = datatype()
    train_dataloader = batch.instance_a_loader(t='train')
    test_dataloader = batch.instance_a_loader(t='test')
    for batch_idx, ss in enumerate(train_dataloader):
        import pudb;pu.db
        #print(batch_idx,ss[0])
        print('{}/{} \n'.format(batch_idx,len(ss)))

    for batch_idx, ss in enumerate(test_dataloader):
        #import pudb;pu.db
        print(batch_idx,ss[0])
        #print('{}/{} \n'.format(batch_idx,len(ss)))

