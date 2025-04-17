import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
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

class Model(nn.Module):
    def __init__(self, args,predefined_A):
        super(Model, self).__init__()
        self.num_nodes = args.num_nodes
        self.input_dim = args.in_dim
        self.rnn_units = 16
        self.output_dim = 1
        self.horizon = args.pre_len
        self.num_layers = 1
        self.cheb_k = 3
        self.ycov_dim = 1
        self.cl_decay_steps = 2000
        self.use_curriculum_learning = True
        self.device=args.device
        self.batch=args.batch_size

        self.target=args.target

        # memory
        self.mem_num = 20
        self.mem_dim = 16
        self.memory = self.construct_memory()

        # encoder
        self.encoder = ADCRNN_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)

        # deocoder
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = ADCRNN_Decoder(self.num_nodes, self.output_dim + self.ycov_dim, self.decoder_dim, self.cheb_k,
                                      self.num_layers)

        # output
        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
        self.proj2=nn.Conv2d(in_channels=4,
                                    out_channels=1,
                                    kernel_size=(1, 1))
        self.proj3 = nn.Conv2d(in_channels=2,
                               out_channels=1,
                               kernel_size=(1, 1))
        self.device = args.device
        self.find_nearest_neighbors = find_k_nearest_neighbors(1, self.device)

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)  # (M, d)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim),
                                         requires_grad=True)  # project to query
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num),
                                          requires_grad=True)  # project memory to embedding
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory['Wq'])  # (B, N, d)
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)  # alpha: (B, N, M)
        value = torch.matmul(att_score, self.memory['Memory'])  # (B, N, d)
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory['Memory'][ind[:, :, 0]]  # B, N, d
        neg = self.memory['Memory'][ind[:, :, 1]]  # B, N, d
        return value, query, pos, neg

    def forward(self, obs_his,era_his,pan_fut,csta,cera,cpan):
        input=obs_his
        cera = cera[:, :-1, :]
        era_his = era_his[:, :, :, :-1, :]
        era_1, pan_1, cera_1 = self.find_nearest_neighbors(obs_his, era_his, pan_fut, csta, cera, cpan)
        pan_fut = pan_1.squeeze(3)
        wrf_fut=pan_fut
        x=input.permute(0,3,2,1)
        y_cov=torch.zeros((self.batch,self.horizon,self.num_nodes,self.ycov_dim)).to(self.device)
        # y_cov = wrf_fut.permute(0,3,2,1)[:,:,:,[self.target]]
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        init_state = self.encoder.init_hidden(x.shape[0])
        h_en, state_en = self.encoder(x, init_state, supports)  # B, T, N, hidden
        h_t = h_en[:, -1, :, :]  # B, N, hidden (last state)

        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)

        ht_list = [h_t] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports)
            go = self.proj(h_de)
            out.append(go)
        output = torch.stack(out, dim=1)

        return output.permute(0,3,2,1)


class AGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(AGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(2 * cheb_k * dim_in, dim_out))  # 2 is the length of support
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)

    def forward(self, x, supports):
        x_g = []
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2])
            support_set.extend(support_ks)
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, 2 * cheb_k * dim_in
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias  # b, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(AGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        # shape of x: (B, T, N, D), shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # return current_inputs, torch.stack(output_hidden, dim=0)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return init_states


class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(ADCRNN_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(AGCRNCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


def print_params(model):
    # print trainable params
    param_count = 0
    print('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    print(f'In total: {param_count} trainable parameters. \n')
    return



