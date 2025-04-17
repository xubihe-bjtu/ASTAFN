import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, C):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size , hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        B, L, N, C = x.shape
        x = x.reshape(B, L, N * C)
        outputs, hidden = self.gru(x)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, C):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(output_size , hidden_size, num_layers, batch_first=True)

    def forward(self, x, hidden):
        B, L, N, C = x.shape
        x = x.reshape(B, L, N * C)
        outputs, hidden = self.gru(x, hidden)
        return outputs, hidden

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

            station_coord = np.array(cobs[n]).reshape(1,2)  # (2,)

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

class Model(nn.Module):
    def __init__(self, args,predefined):
        super(Model, self).__init__()
        self.input_size=args.in_dim
        self.hidden_size=32
        self.num_layers=2
        self.C=4
        self.output_dim=1
        self.num_nodes=args.num_nodes
        self.device=args.device
        self.encoder = Encoder(self.input_size*self.num_nodes, self.hidden_size, self.num_layers, self.C)
        self.decoder = Decoder(self.input_size*self.num_nodes, self.hidden_size, self.num_layers, self.C)
        self.output_layer = nn.Linear(self.hidden_size, self.output_dim*self.num_nodes)
        self.find_nearest_neighbors = find_k_nearest_neighbors(1, self.device)


    def forward(self, obs_his,era_his,pan_fut,csta,cera,cpan):
        input=obs_his.permute(0,3,2,1)
        cera = cera[:, :-1, :]
        era_his = era_his[:, :, :, :-1, :]
        era_1, pan_1, cera_1 = self.find_nearest_neighbors(obs_his, era_his, pan_fut, csta, cera, cpan)
        pan_fut = pan_1.squeeze(3).permute(0,3,2,1)
        encoder_outputs, hidden = self.encoder(input)
        decoder_outputs, _ = self.decoder(pan_fut, hidden)
        decoder_outputs=self.output_layer(decoder_outputs).unsqueeze(dim=-1)
        return decoder_outputs.permute(0,3,2,1)