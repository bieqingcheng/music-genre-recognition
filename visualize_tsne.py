import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.manifold import TSNE

classes = ['Chacha','Foxtrot','Jive','Pasodoble','Quickstep','Rumba','Salsa','Samba',
'Slowwaltz','Tango','Viennesewaltz','Waltz','Wcswing']
classes.sort()
class_to_idx = {classes[i]: i for i in range(len(classes))}
idex_to_class = {i:classes[i] for i in range(len(classes))}
def visualize(feat,label,fold):
    #plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999','#000000','#9ACD32','#2F4F4F']
    plt.figure()
    classes.sort()
    
    for i in range(len(classes)):
        value = list()
        for j in range(len(label)):
            if label[j] == i:
                value.append(feat[j])
        value = np.array(value)
        #value = feat[classes[i]]
        plt.plot(value[:,0], value[:, 1], '.', c=c[i])
    #plt.legend(classes, loc = 'bottom right')
    #plt.xlim(xmin=-8,xmax=8)
    #plt.ylim(ymin=-8,ymax=8)

    plt.savefig('./images/ballroom_fold%d.jpg' % fold)



if __name__ == '__main__':
    tsne = TSNE(n_components = 2, init='pca',random_state = 0)
    root = '/home/zwj/mgr/extract_coding/st/test/st_extract_feature_{}.pkl'
    #out = '/home/zwj/mgr/extract_coding/st/test/st_output_result_{}.pkl'
    #feat = list()
    #label = list()
    for i in range(10):
        feat = list()
        label = list()
        a = pickle.load(open(root.format(i),'rb'))
        for genre,temp in a.items():
            for name,value in temp.items():
                #import pudb;pu.db
                feat.append(value)
                label.append(class_to_idx[genre])

        value = np.array(feat)
        result = tsne.fit_transform(value)
        #feat[genre] = result
        visualize(result,label,i)
       
    
    
    
