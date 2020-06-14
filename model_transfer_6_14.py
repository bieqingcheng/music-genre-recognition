import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class prefusion(nn.Module):
    def __init__(self,width = 128):
        super(prefusion,self).__init__()
        #self.mvdFc = nn.Linear(420,width,bias=False)
        #self.rhFc = nn.Linear(160, width, bias=False)
        #self.rpFc = nn.Linear(1440, width, bias=False)
        #self.ssdFc = nn.Linear(168, width, bias=False)
        #self.trhFc = nn.Linear(420, width, bias=False)
        #self.tssdFc = nn.Linear(1176, width, bias=False)
        
        self.rhythmConv1 = nn.Conv1d(1,32,kernel_size=5,padding=2)
        self.rhythmBn1= nn.BatchNorm1d(32)
        self.rhythmElu1 = nn.ELU()
        self.rhythmConv2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.rhythmBn2 = nn.BatchNorm1d(32)
        self.rhythmElu2 = nn.ELU()
 
    def forward(self, x):
 
        rhythm =self.rhythmConv1(x)#shape(b,1,160)->(b,32,160)
        rhythm = self.rhythmBn1(rhythm)
        rhythm = self.rhythmElu1(rhythm)
        rhythm = self.rhythmConv2(rhythm)
        rhythm = self.rhythmBn2(rhythm)
        rhythm_1 = self.rhythmElu2(rhythm)#shape(b,32,160)

 
        return rhythm_1.permute(0,2,1) #(b,160,32)

class NetVLAD(nn.Module):

    def __init__(self, num_clusters=64, dim=128, alpha=100.0,
                 normalize_input=True):

        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        sy = (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        sc = sy.data
        #self.conv.weight = nn.Parameter((2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)) 
        # （20，32，1，1）
        self.conv.weight = nn.Parameter(sc)
        sy1 = - self.alpha * self.centroids.norm(dim=1)
        sc1 = sy1.data
        #self.conv.bias = nn.Parameter(- self.alpha * self.centroids.norm(dim=1))  # (20)
        self.conv.bias = nn.Parameter(sc1)
    def forward(self, x):
        N, C = x.shape[:2]  # (29,32,4,4)

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim  在维度1上实现L2距离

        # soft-assignment

        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        # soft_assign (40,20,4,4)->(40,20,16)
        soft_assign = F.softmax(soft_assign, dim=1)  # (40,32,16)
        # soft_assign (40,20,16)

        x_flatten = x.view(N, C, -1)  # (40,32,16)

        # calculate residuals to each clusters  (40,20,32,16)-(1, 20, 32, 16)=(40,20,32,16)
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)  # (40,20,32,16)*(40,20,1,16)=(40,20,32,16)
        vlad = residual.sum(dim=-1)  # (40,20,32)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        _, cluster, channel = vlad.shape
        vlad = vlad.view(x.size(0), -1)  # flatten(40,20,32) -> (40,20*32)
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        vlad = vlad.view(vlad.size(0),channel, cluster)
        return vlad


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
		self.premerge = prefusion() #(b,160,32)

		self.attn1 = Self_Attn(2, 'relu')

		#self.fc1 = nn.Linear( 128*400 ,1024)
		self.fc1 = nn.Linear( 160*32*2 ,1024)
		self.bn6 = nn.BatchNorm1d(1024)
		self.relu6 = nn.ELU()
		self.dropout = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(1024, 13)

	def forward(self, x):
		#import pudb;pu.db
		x = self.premerge(x)

		out = x.expand(2,-1,-1,-1).permute(1, 0, 2, 3)
		out,_ = self.attn1(out) #out (batch,n_head,W,H)
		out = out.view(out.size(0),out.size(2),-1) #out (batch,W,H*n_head)
		##############

		x = out.view(out.size(0), -1)
		# import pudb;pu.db
		x = self.fc1(x)
		x = self.bn6(x)
		# x = F.relu(x)
		#x = self.relu6(x)
		x1 = self.relu6(x)
		x = self.dropout(x1)
		x = self.fc2(x)
		x = F.softmax(x ,dim=1)
		#print('no softmax')
		return (x,x1)
		#return x

if __name__ == "__main__":
    model = model_audio()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print(model)

    #x = torch.from_numpy(np.arange(1,13).reshape(4,3)).float()
    x = torch.FloatTensor(np.random.randint(1, 4, (160)))
    x = x.unsqueeze(0)
    x = x.expand(10,-1,-1)
    print(type(x),x.shape)
    print('\n')
    x = torch.autograd.Variable(x)
    print(type(x))
    y = model(x)
    print(y.size())
