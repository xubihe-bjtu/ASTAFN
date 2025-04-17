import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import NearestNeighbors

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class proxy_site_alignment(nn.Module):
    def __init__(self, k,d_align,num_node,seq_len,input_dim,device,init=True):
        super(proxy_site_alignment, self).__init__()
        self.k=k
        self.d_align=d_align
        self.init=init
        self.seq_len=seq_len
        self.in_dim=input_dim
        self.query_layer = nn.Conv2d(in_channels=seq_len,out_channels=d_align,kernel_size=(1, 1))
        self.key_layer = nn.Conv2d(in_channels=seq_len,out_channels=d_align,kernel_size=(1, 1))
        self.query_layer_c = nn.Conv2d(in_channels=2, out_channels=d_align, kernel_size=(1, 1))
        self.key_layer_c = nn.Conv2d(in_channels=2, out_channels=d_align, kernel_size=(1, 1))
        self.device=device
        self.bias=nn.Parameter(torch.zeros( 1,self.in_dim, num_node, self.k).to(self.device), requires_grad=True).to(self.device)

    def forward(self,obs_his,era_k,csta,cera):
        '''
        obs_his:(B,C,N,L)
        era_k:(B,C,N,k,L)
        pan_k:(B,C,N,k,L)
        csta:(N,2)
        cera:(N,k,2)
        '''
        B, C, N, L = obs_his.shape
        obs_his=obs_his.permute(0,3,2,1)
        era_k=era_k.view(B,C,-1,L).permute(0,3,2,1)
        Q=self.query_layer(obs_his)#B,d_model,N,C
        K=self.key_layer(era_k)
        Q=Q.permute(0,3,2,1)#B,C,N,d
        K=K.permute(0,3,2,1)
        K=K.reshape(B,C,N,self.k,self.d_align)#B,C,N,k,d
        attn_scores=torch.einsum('bcnd,bcnkd->bcnk',Q,K)#B,c,N,k
        attn_scores=F.softmax((attn_scores+self.bias),dim=-1)
        return attn_scores

class find_k_nearest_neighbors(nn.Module):
    def __init__(self, k,device):
        super(find_k_nearest_neighbors, self).__init__()
        self.k=k
        self.device=device

    def forward(self,obs_his, era_his, pan_fut, cobs, cera, cpan):
        B, C,N,L = obs_his.shape
        era_his=era_his.reshape(B,C,-1,L)
        pan_fut=pan_fut.reshape(B,C,-1,L)
        cera_flat = cera.reshape(-1, 2)  # (lat * lon, 2)
        cpan_flat = cpan.reshape(-1, 2)  # (lat * lon, 2)

        nbrs_era = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(cera_flat)
        nbrs_pan = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(cpan_flat)

        era_k=[]
        pan_k=[]
        cera_k=[]
        for n in range(N):
            # 获取当前 obs_his 站点的坐标
            station_coord = np.array(cobs[n]).reshape(1,2)  # (2,)

            # 获取该站点最近的 k 个 ERA 和 PAN 网格点索引
            _, indices_era = nbrs_era.kneighbors(station_coord)
            _, indices_pan = nbrs_pan.kneighbors(station_coord)
            indices_era=torch.Tensor(indices_era).to(self.device).long()
            indices_pan=torch.Tensor(indices_pan).to(self.device).long()
            era_his_n=era_his[:,:,indices_era,:]#era_his:(B,C,1,k,L)
            pan_fut_n=pan_fut[:,:,indices_era,:]#pan_fut:(B,C,1,k,L)
            cera_n=torch.Tensor(cera_flat[indices_era.cpu(),:]).to(self.device)
            cpan_n = torch.Tensor(cpan_flat[indices_era.cpu(), :]).to(self.device)
            era_k.append(era_his_n)
            pan_k.append(pan_fut_n)
            cera_k.append(cera_n)
        era_k=torch.cat(era_k,dim=2)#era_k:(B,C,N,k,L)
        pan_k=torch.cat(pan_k,dim=2)#pan_k:(B,C,N,k,L)

        if self.k!=1:
            cera_k=torch.cat(cera_k,dim=0)
        else:
            cera_k=torch.cat(cera_k,dim=0)
            cera_k=cera_k.reshape(N,2)
            cera_k=cera_k.unsqueeze(1)
        return era_k,pan_k,cera_k

class ValueEmbedding(nn.Module):
    def __init__(self, c_in,d_model):
        super(ValueEmbedding, self).__init__()

        self.embed = nn.Conv2d(in_channels=c_in,out_channels=d_model,kernel_size=(1, 1))

    def forward(self, x):
        value_emb=self.embed(x)
        value_emb=value_emb.permute(0,3,2,1)
        return value_emb

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).unsqueeze(2)
        self.register_buffer('pe', pe)
        self.d_model=d_model

    def forward(self, x):
        x=x.permute(0,3,2,1)
        B,L,N,_=x.shape
        # Extract positional encoding for the given sequence length
        pos_enc = self.pe[:, :L, :, :].expand(B, L, N, self.d_model)
        return pos_enc

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model,args, dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = ValueEmbedding(c_in=args.in_dim,d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x)+self.position_embedding(x)
        return self.dropout(x)

class TCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(TCNLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), dilation=(1, dilation),
                              padding=(0, (kernel_size - 1) * dilation // 2))
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, kernel_size, max_dilation=16):
        super(TCN, self).__init__()

        self.layers = nn.ModuleList()

        dilation = 1
        for i in range(num_layers):
            self.layers.append(TCNLayer(in_channels, out_channels, kernel_size, dilation))
            # 每层的膨胀因子加倍
            dilation = min(dilation * 2, max_dilation)

        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))  # 用于输出投影
        self.dropout=0.2


    def forward(self, x,new_adp):
        batch_size, channels, num_nodes, seq_len = x.size()
        residual = x
        h=[]
        for layer in self.layers:
            x = layer(x)
            x = x + residual
            residual = x
            h.append(x)
        x = self.final_conv(x)
        return x,h

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in = (order * support_len + 1) * c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class HierarchicalGCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len, order=2):
        super(HierarchicalGCN, self).__init__()
        self.gcn=gcn(c_in, c_out, dropout, support_len, order=2)

    def forward(self,X,new_adp):
        #x:B,D,N,L
        spatio_list=[]
        for x in X:
            x=x.permute(0,3,2,1)#x:B,L,N,D
            spatio_emb=self.gcn(x,new_adp).permute(0,3,2,1)
            spatio_list.append(spatio_emb)
        return spatio_list

class AdaptiveCombiner(nn.Module):
    def __init__(self,in_dim,out_dim,target,seq_len):
        super(AdaptiveCombiner,self).__init__()
        self.target=target
        self.seq_len=seq_len
        self.conv1=nn.Conv2d(in_channels=in_dim,
                                out_channels=out_dim,
                                kernel_size=(1, 1),
                                bias=True)
        self.conv2=nn.Linear(in_features=self.seq_len,out_features=in_dim*self.seq_len)

        self.relu=nn.ReLU()

    def forward(self,x_emb,y_emb,era_k,obs_his,attn_scores):

        B,_,N,L=obs_his.shape
        tilde_E=(attn_scores * era_k).sum(dim=3)

        tilde_E=tilde_E[:,[self.target],:,:]#B,1,N,L
        obs_his=obs_his[:,[self.target],:,:]#B,1,N,L
        error = tilde_E *obs_his
        error=error.reshape(-1,self.seq_len)
        error_emb=self.conv2(error)
        error_emb = error_emb.reshape(B, -1, N, self.seq_len)
        weight = torch.sigmoid(self.conv1(x_emb+y_emb+error_emb))
        return weight

class CrossTimeAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(CrossTimeAttention, self).__init__()
        self.attention_dim = attention_dim

        self.query_layer1 = nn.Linear(input_dim, attention_dim)
        self.key_layer1 = nn.Linear(input_dim, attention_dim)
        self.value_layer1 = nn.Linear(input_dim, attention_dim)

        self.query_layer2 = nn.Linear(input_dim, attention_dim)
        self.key_layer2 = nn.Linear(input_dim, attention_dim)
        self.value_layer2 = nn.Linear(input_dim, attention_dim)

    def forward(self, x, y):
        """
        x:  (B, L, N, C)
        y: (B, L, N, C)
        return:
            z: (B, L, N, attention_dim)
        """
        B,C,N,L = x.size()


        x = x.permute(0,2,3,1).contiguous()  # (B, N, L, C)
        y = y.permute(0, 2, 3,1).contiguous()  # (B, N, L, C)

        x = x.view(B * N, L, C)  #  (B * N, L, C)
        y = y.view(B * N, L, C)  #  (B * N, L, C)

        q1 = self.query_layer1(x)  # (B * N, L, attention_dim)
        k1 = self.key_layer1(y)    # (B * N, L, attention_dim)
        v1 = self.value_layer1(y)  # (B * N, L, attention_dim)

        q2 = self.query_layer2(y)  # (B * N, L, attention_dim)
        k2 = self.key_layer2(x)  # (B * N, L, attention_dim)
        v2 = self.value_layer2(x)  # (B * N, L, attention_dim)

        k1 = k1.transpose(1, 2)  # (B * N, attention_dim, L)
        k2 = k2.transpose(1, 2)  # (B * N, attention_dim, L)

        scores = torch.bmm(q1, k1) / (self.attention_dim ** 0.5)  # (B * N, L, L)

        attention_weights = torch.softmax(scores, dim=-1)      # (B * N, L, L)

        z1 = torch.bmm(attention_weights, v1)  # (B * N, L, attention_dim)

        z1 = z1.view(B, N, L, self.attention_dim)  # (B, N, L, attention_dim)
        z1 = z1.permute(0, 2, 1, 3).contiguous()  # (B, L, N, attention_dim)

        scores = torch.bmm(q2, k2) / (self.attention_dim ** 0.5)  # (B * N, L, L)
        attention_weights = torch.softmax(scores, dim=-1)  # (B * N, L, L)

        z2 = torch.bmm(attention_weights, v2)  # (B * N, L, attention_dim)

        z2 = z2.view(B, N, L, self.attention_dim)  # (B, N, L, attention_dim)
        z2 = z2.permute(0, 2, 1, 3).contiguous()  # (B, L, N, attention_dim)

        return z1,z2

class HierarchicalCrossDomainAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,label='Time'):
        super(HierarchicalCrossDomainAttention, self).__init__()
        self.layer_num=num_layers
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.time_attention=CrossTimeAttention(self.input_size, self.hidden_size)

        self.label=label

    def forward(self,h,f):
    #h:list,f:list (B,D,N,L)
        h_attn_list,f_attn_list=[],[]
        for i in range(self.layer_num):
            #h_attn:B,L,N,D
            if self.label=='Time':
                h_attn,f_attn=self.time_attention(h[i],f[i])
            else:
                h_attn, f_attn = self.spatio_attention(h[i], f[i])
            h_attn_list.append(h_attn)
            f_attn_list.append(f_attn)
        return h_attn_list,f_attn_list


class NodeEmbeddingGenerator(nn.Module):
    def __init__(self, input_dim, d_node, num_nodes):
        """
        Args:
            input_dim (int):  (c)。
            d_node (int): d
            num_nodes (int):  (N)。
        """
        super(NodeEmbeddingGenerator, self).__init__()

        # 可学习线性层，将输入特征映射到节点嵌入维度
        self.node_embedding_layer = torch.nn.Conv2d(input_dim, d_node, kernel_size=(1, 1))

        # 可学习的节点偏置项
        self.node_bias = nn.Parameter(torch.randn(num_nodes, d_node))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, C,N,L)。

        Returns:
            torch.Tensor:(N, d_node)。
        """

        x_projected = self.node_embedding_layer(x).permute(0,3,2,1)  # (B, d_node, N, L)
        x_mean = x_projected.mean(dim=(0, 1))  # (N, d_node)
        node_embeddings = x_mean  # (N, d_node)
        return node_embeddings

class Model(nn.Module):
    def __init__(self, args, predefined_A):
        super(Model, self).__init__()

        self.supports=predefined_A
        self.in_dim=args.in_dim
        self.d_model=args.d_model
        self.d_align=args.d_align
        self.dropout=0.2
        self.batch_size=args.batch_size
        self.num_nodes=args.num_nodes
        self.device=args.device
        self.num_layers=args.num_layer
        self.attention_dim=args.d_model//self.num_layers
        self.seq_len=args.seq_len
        self.output_len=args.pre_len
        self.k=args.k
        self.target=args.target
        self.d_node=10

        self.nodevecgenerate1 = NodeEmbeddingGenerator(self.in_dim, self.d_node, self.num_nodes)
        self.nodevecgenerate2 = NodeEmbeddingGenerator(self.in_dim, self.d_node, self.num_nodes)
        self.nodevecgenerate3 = NodeEmbeddingGenerator(self.in_dim, self.d_node, self.num_nodes)
        self.nodevecgenerate4 = NodeEmbeddingGenerator(self.in_dim, self.d_node, self.num_nodes)

        self.proxy_site_alignment=proxy_site_alignment(self.k,self.d_align,self.num_nodes,self.seq_len,self.in_dim,self.device)
        self.find_k_nearest_neighbors=find_k_nearest_neighbors(self.k,self.device)
        self.find_nearest_neighbors = find_k_nearest_neighbors(1, self.device)
        self.his_data_embedding = DataEmbedding(self.in_dim, self.d_model, args, self.dropout)
        self.fut_data_embedding = DataEmbedding(self.in_dim, self.d_model, args, self.dropout)
        self.his_tcn = TCN(self.d_model, self.d_model, self.num_layers, kernel_size=3, max_dilation=16)
        self.fut_tcn = TCN(self.d_model, self.d_model, self.num_layers, kernel_size=3, max_dilation=16)
        self.his_gcn = HierarchicalGCN(self.seq_len, self.output_len, self.dropout, support_len=3)
        self.fut_gcn = HierarchicalGCN(self.seq_len, self.output_len, self.dropout, support_len=3)
        self.cross_spatio_attention = HierarchicalCrossDomainAttention(self.d_model, self.attention_dim,
                                                                       num_layers=self.num_layers, label='Time')

        self.adaptivefusion = AdaptiveCombiner(self.d_model,  self.d_model,
                                               self.target, self.seq_len)
        self.predict_layer = nn.Conv2d(in_channels=self.d_model, out_channels=1, kernel_size=(1, 1), bias=True)


    def forward(self,obs_his,era_his,pan_fut,csta,cera,cpan):
        '''
        Input:obs_his:(B,C,N,L)
              era_his:(B,C,lat,lon,L)
              pan_fut:(B,C,lat,lon,L)
              cobs:(24,2)
              cera:(lat,lon,2):25,37,2
              cpan:(lat,lon,2)25,36,2
        '''
        cera=cera[:,:-1,:]
        era_his=era_his[:,:,:,:-1,:]

        era_k,pan_k,cera_k=self.find_k_nearest_neighbors(obs_his, era_his, pan_fut, csta, cera, cpan)

        attn_scores=self.proxy_site_alignment(obs_his,era_k,csta,cera_k)

        attn_scores_expanded = attn_scores.unsqueeze(-1)  # (B, C, N, k, 1)

        tilde_P = (attn_scores_expanded * pan_k).sum(dim=3)  # (B, C, N, L)

        x_emb = self.his_data_embedding(obs_his)
        y_emb = self.fut_data_embedding(tilde_P)

        nodevec1 = self.nodevecgenerate1(obs_his)
        nodevec2 = self.nodevecgenerate2(obs_his)
        nodevec3 = self.nodevecgenerate3(tilde_P)
        nodevec4 = self.nodevecgenerate4(tilde_P)

        adp_h = F.softmax(F.relu(torch.mm(nodevec1, nodevec2.T)), dim=1)
        new_adp_h = self.supports + [adp_h]

        adp_f = F.softmax(F.relu(torch.mm(nodevec3, nodevec4.T)), dim=1)
        new_adp_f = self.supports + [adp_f]

        x_emb = x_emb.permute(0, 3, 2, 1)
        xtime_emb, h = self.his_tcn(x_emb, new_adp_h)
        y_emb = y_emb.permute(0, 3, 2, 1)
        ytime_emb, f = self.fut_tcn(y_emb, new_adp_f)

        xspa_emb = self.his_gcn(h, new_adp_h)  # spa_emb:B,D,N,L
        yspa_emb = self.fut_gcn(f, new_adp_f)
        spatio_attn_f, spatio_attn_h = self.cross_spatio_attention(xspa_emb, yspa_emb)
        spatio_attn_f = torch.cat(spatio_attn_f, dim=-1)
        spatio_attn_h = torch.cat(spatio_attn_h, dim=-1)

        fuse_h = xtime_emb+spatio_attn_h.permute(0,3,2,1)
        fuse_f=ytime_emb+spatio_attn_f.permute(0,3,2,1)

        W = self.adaptivefusion(fuse_h, fuse_f, era_k, obs_his, attn_scores_expanded)
        output = self.predict_layer(fuse_h + W * fuse_f)

        return output