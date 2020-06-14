import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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

class Attention(nn.Module):
    #输入：(6,64,20)
    #在concat注意力机制中，权值V是不断学习的所以要是parameter类型，
    # 不直接使用一个torch.nn.Linear()可能是因为学习的效果不好。
    #通过做下面的实验发现，linear里面的weight和bias就是parameter类型，
    # 且不能够使用tensor类型替换，
    # 还有linear里面的weight甚至可能通过指定一个
    # 不同于初始化时候的形状进行模型的更改。
    #torch.nn.Parameter()

    def __init__(self):
        super(Attention, self).__init__()
        #emb_size = 224
        self.head_num = 10
        self.att_emb_size = 20
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.w_query = nn.Linear(self.att_emb_size, self.att_emb_size*self.head_num, bias=False)
        self.w_keys = nn.Linear(self.att_emb_size, self.att_emb_size*self.head_num, bias=False)
        self.w_values = nn.Linear(self.att_emb_size, self.att_emb_size*self.head_num, bias=False)
        self.w_resnet = nn.Linear(self.att_emb_size, self.att_emb_size*self.head_num, bias=False)

    def forward(self, x):
        '''
        :param x:  （batch ， m, emb_size） m=一阶+二阶+无交互+3
        :return: y batch
        '''
        batch, m, _ = x.size()
        query = self.w_query(x).view(self.head_num, batch, m, self.att_emb_size)
        keys = self.w_keys(x).view(self.head_num, batch, self.att_emb_size, m)
        values = self.w_values(x).view(self.head_num, batch, m, self.att_emb_size)

        inner_pro = query.matmul(keys)  # head_num * batch * m * m
        att_score = self.softmax(inner_pro)

        #result = att_score.matmul(values)  # head_num * batch * m * att_emb_size
        result = (att_score.permute(0, 1, 3, 2)).matmul(values)
        #(10,b,224,224) *(10,b,224,20) = (10,b,224,20)
        # head compress, may need some modification
        # result = t.mean(result, 0)  # batch * m * att_emb_size
        result = result.view(batch, m, -1)  # batch, m * att_emb_size * head_num

        # theme from resnet
        result = F.relu(self.gamma * result + self.w_resnet(x))  # batch , m , att_emb_size*head_num

        return result


class model_audio(nn.Module):
    def __init__(self):
        super(model_audio, self).__init__()
        # self.features = features
        self.conv1 = nn.Conv2d(1 ,32 ,kernel_size =3,padding =1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ELU(  )  # nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size =2 ,stride =2)

        self.conv2 = nn.Conv2d(32, 32, kernel_size =3,padding =1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ELU(  )  # nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 32, kernel_size =3,padding =1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ELU(  )  # nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(32, 64, kernel_size =3,padding =1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ELU(  )  # nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.relu5 = nn.ELU()  # nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.attention = Attention()
        self.attn1 = Self_Attn(10, 'relu')
        self.net_vlad1 = NetVLAD(num_clusters=10, dim=32, alpha=1.0)
        self.net_vlad2 = NetVLAD(num_clusters=10, dim=32, alpha=1.0)
        self.net_vlad3 = NetVLAD(num_clusters=10, dim=32, alpha=1.0)
        self.net_vlad4 = NetVLAD(num_clusters=10, dim=64, alpha=1.0)
        self.net_vlad5 = NetVLAD(num_clusters=10, dim=64, alpha=1.0)

        self.fc1 = nn.Linear( 224*100 ,1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ELU()
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 13)

    def forward(self, x):
        #import pudb;pu.db
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # x = F.relu(x)
        feature1 = self.maxpool1(x)
        feature11 = self.net_vlad1(feature1)

        x = self.conv2(feature1)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = F.relu(x)
        feature2 = self.maxpool2(x)
        feature21 = self.net_vlad2(feature2)

        x = self.conv3(feature2)
        x = self.bn3(x)
        x = self.relu3(x)
        # x = F.relu(x)
        feature3 = self.maxpool3(x)
        feature31 = self.net_vlad3(feature3)

        x = self.conv4(feature3)
        x = self.bn4(x)
        x = self.relu4(x)
        # x = F.relu(x)
        feature4 = self.maxpool4(x)
        feature41 = self.net_vlad4(feature4)

        x = self.conv5(feature4)
        x = self.bn5(x)
        x = self.relu5(x)
        # x = F.relu(x)
        feature5 = self.maxpool5(x)
        feature51 = self.net_vlad5(feature5)

        feature_cat = torch.cat((feature11, feature21,feature31,feature41,feature51), dim=1)
        #feature_cat = nn.C((feature11, feature21,feature31,feature41,feature51), dim=1)
        #############
        #out = self.attention(feature_cat)


        ##########
        #feature_cat (batch,224,20)
        out = feature_cat.expand(10,-1,-1,-1).permute(1, 0, 2, 3)
        out,_ = self.attn1(out) #out (batch,n_head,W,H)
        out = out.view(out.size(0),out.size(2),-1) #out (batch,W,H*n_head)
        ##############

        x = out.view(out.size(0), -1)
        # import pudb;pu.db
        x = self.fc1(x)
        x = self.bn6(x)
        # x = F.relu(x)
        x1 = self.relu6(x)
        x = self.dropout(x1)
        x = self.fc2(x)
        x = F.softmax(x ,dim=1)
        #return x
        return (x,x1)

if __name__ == "__main__":
    model = model_audio()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    print(model)

    #x = torch.from_numpy(np.arange(1,13).reshape(4,3)).float()
    x = torch.FloatTensor(np.random.randint(1, 4, (216, 128)))
    #x = torch.FloatTensor(np.random.randint(1, 4, (647, 544)))
    x = x.unsqueeze(0)
    x = x.expand(10,-1,-1,-1)
    print(type(x))
    print('\n')
    x = torch.autograd.Variable(x)
    print(type(x))
    y = model(x)
    print(y.size())

