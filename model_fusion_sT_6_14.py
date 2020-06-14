import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class prefusion(nn.Module):
    def __init__(self,width = 120):
        super(prefusion,self).__init__()
        # mel, scatter, transfer
        self.Conv1 = nn.Conv1d(5,32,kernel_size=5,stride =2)
        self.Bn1= nn.BatchNorm1d(32)
        self.Elu1 = nn.ELU() #(b,12,510)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2) #(b,12,255)

        self.Conv4 = nn.Conv1d(32, 64, kernel_size=5)
        self.Bn4 = nn.BatchNorm1d(64)
        self.Elu4 = nn.ELU()
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.Conv5 = nn.Conv1d(64, 32, kernel_size=3)
        self.Bn5 = nn.BatchNorm1d(32)
        self.Elu5 = nn.ELU()

    def forward(self, x):
        fusion_feat_0  = self.Conv1(x)#shape(b,3,1024)->(b,16,510)
        fusion_feat = self.Bn1(fusion_feat_0)
        #fusion_feat = self.maxpool(fusion_feat)
        fusion_feat = self.Elu1(fusion_feat)
        fusion_feat = self.maxpool1(fusion_feat) #(b,16,255)

        fusion_feat = self.Conv4(fusion_feat) #(b,64,251)
        fusion_feat = self.Bn4(fusion_feat)
        fusion_feat = self.Elu4(fusion_feat)#shape(b,64,251)
        fusion_feat = self.maxpool4(fusion_feat) #(b,64,125)
        fusion_feat = self.Conv5(fusion_feat) #(b,32,123)
        fusion_feat = self.Bn5(fusion_feat)
        fusion_feat = self.Elu5(fusion_feat)

        return fusion_feat.permute(0,2,1) #(b,123,32)

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    '''
    输入(batch,n_head,224,20),in_dim = 224
    '''

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        #(b,224->out_dim,20-kernel_size+1)=(b,224,20)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
    def forward(self, x):
        """
            inputs :
                x : input feature maps(batch,n_head,W,H)
            returns :
                out : self attention value + input feature
                attention: (B,N,N) (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        # (batch,n_head,W*H)->(batch,W*H,n_head)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        # (batch,n_head,W*H)
        energy = torch.bmm(proj_query, proj_key)
        # transpose check  (batch,W*H,W*H)
        attention = self.softmax(energy)  # (batch,W*H,W*H)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        # (batch,n_head,W*H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        #(batch,n_head,W*H) * (batch,W*H,W*H) = (batch,n_head,W*H)
        out = out.view(m_batchsize, C, width, height)
        #(batch,n_head,W,H)

        out = self.gamma * out + x
        return out, attention

class model_audio(nn.Module):
    def __init__(self):
        super(model_audio, self).__init__()
        # self.features = features
        self.premerge = prefusion() #(b,123,32)
        self.attn1 = Self_Attn(5, 'relu')
        self.attn2 = Self_Attn(5, 'relu')
        self.fc1 = nn.Linear( 123*32*5 ,1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ELU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 13)

    def forward(self, x):
        #import pudb;pu.db
        x = self.premerge(x)
        out = x.expand(5,-1,-1,-1).permute(1, 0, 2, 3)
        out,_ = self.attn1(out) #out (batch,n_head,W,H)
        out = out.permute(0,1,3,2)
        out,_ = self.attn2(out)
        out = out.view(out.size(0),out.size(2),-1) #out (batch,W,H*n_head)
        x = out.view(out.size(0), -1)
        x = self.fc1(x)
        x = self.bn6(x)
        x1 = self.relu6(x)
        x = self.dropout(x1)
        x = self.fc2(x)
        x = F.softmax(x ,dim=1)
        return x
        #return (x,x1)#return x 3_20
if __name__ == "__main__":
    model = model_audio()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print(model)

    #x = torch.from_numpy(np.arange(1,13).reshape(4,3)).float()
    x = torch.FloatTensor(np.random.randint(1, 4, (5,1024)))
    x = x.expand(10,-1,-1)
    print(type(x))
    print('\n')
    x = torch.autograd.Variable(x)
    print(type(x))
    y = model(x)
    print(y.size())

