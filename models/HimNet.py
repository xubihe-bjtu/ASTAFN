import torch
import torch.nn as nn
import numpy as np
from sklearn.neighbors import NearestNeighbors

class HimGCN(nn.Module):
    def __init__(self, input_dim, output_dim, cheb_k, embed_dim, meta_axis=None):
        super().__init__()
        self.cheb_k = cheb_k
        self.meta_axis = meta_axis.upper() if meta_axis else None

        if meta_axis:
            self.weights_pool = nn.init.xavier_normal_(
                nn.Parameter(
                    torch.FloatTensor(embed_dim, cheb_k * input_dim, output_dim)
                )
            )

            self.bias_pool = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(embed_dim, output_dim))
            )
        else:
            self.weights = nn.init.xavier_normal_(
                nn.Parameter(torch.FloatTensor(cheb_k * input_dim, output_dim))
            )
            self.bias = nn.init.constant_(
                nn.Parameter(torch.FloatTensor(output_dim)), val=0
            )

    def forward(self, x, support, embeddings):
        x_g = []

        if support.dim() == 2:
            graph_list = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("nm,bmc->bnc", graph, x))
        elif support.dim() == 3:
            graph_list = [
                torch.eye(support.shape[1])
                .repeat(support.shape[0], 1, 1)
                .to(support.device),
                support,
            ]
            for k in range(2, self.cheb_k):
                graph_list.append(
                    torch.matmul(2 * support, graph_list[-1]) - graph_list[-2]
                )
            for graph in graph_list:
                x_g.append(torch.einsum("bnm,bmc->bnc", graph, x))
        x_g = torch.cat(x_g, dim=-1)

        if self.meta_axis:
            if self.meta_axis == "T":
                weights = torch.einsum(
                    "bd,dio->bio", embeddings, self.weights_pool
                )  # B, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)  # B, out_dim
                x_gconv = (
                        torch.einsum("bni,bio->bno", x_g, weights) + bias[:, None, :]
                )  # B, N, out_dim
            elif self.meta_axis == "S":
                weights = torch.einsum(
                    "nd,dio->nio", embeddings, self.weights_pool
                )  # N, cheb_k*in_dim, out_dim
                bias = torch.matmul(embeddings, self.bias_pool)
                x_gconv = (
                        torch.einsum("bni,nio->bno", x_g, weights) + bias
                )  # B, N, out_dim
            elif self.meta_axis == "ST":
                weights = torch.einsum(
                    "bnd,dio->bnio", embeddings, self.weights_pool
                )  # B, N, cheb_k*in_dim, out_dim
                bias = torch.einsum("bnd,do->bno", embeddings, self.bias_pool)
                x_gconv = (
                        torch.einsum("bni,bnio->bno", x_g, weights) + bias
                )  # B, N, out_dim

        else:
            x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias

        return x_gconv


class HimGCRU(nn.Module):
    def __init__(
            self, num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis="S"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = output_dim
        self.gate = HimGCN(
            input_dim + self.hidden_dim, 2 * output_dim, cheb_k, embed_dim, meta_axis
        )
        self.update = HimGCN(
            input_dim + self.hidden_dim, output_dim, cheb_k, embed_dim, meta_axis
        )

    def forward(self, x, state, support, embeddings):
        # x: B, N, input_dim
        # state: B, N, hidden_dim
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, support, embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, support, embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.num_nodes, self.hidden_dim)


class HimEncoder(nn.Module):
    def __init__(
            self,
            num_nodes,
            input_dim,
            output_dim,
            cheb_k,
            num_layers,
            embed_dim,
            meta_axis="S",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HimGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, x, support, embeddings):
        # x: (B, T, N, C)
        batch_size = x.shape[0]
        in_steps = x.shape[1]

        current_input = x
        output_hidden = []
        for cell in self.cells:
            state = cell.init_hidden_state(batch_size).to(x.device)
            inner_states = []
            for t in range(in_steps):
                state = cell(current_input[:, t, :, :], state, support, embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_input = torch.stack(inner_states, dim=1)

        # current_input: the outputs of last layer: (B, T, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        return current_input, output_hidden


class HimDecoder(nn.Module):
    def __init__(
            self,
            num_nodes,
            input_dim,
            output_dim,
            cheb_k,
            num_layers,
            embed_dim,
            meta_axis="ST",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            [HimGCRU(num_nodes, input_dim, output_dim, cheb_k, embed_dim, meta_axis)]
            + [
                HimGCRU(num_nodes, output_dim, output_dim, cheb_k, embed_dim, meta_axis)
                for _ in range(1, num_layers)
            ]
        )

    def forward(self, xt, init_state, support, embeddings):
        # xt: (B, N, D)
        # init_state: (num_layers, B, N, hidden_dim)
        current_input = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.cells[i](current_input, init_state[i], support, embeddings)
            output_hidden.append(state)
            current_input = state
        return current_input, output_hidden

class find_k_nearest_neighbors(nn.Module):
    def __init__(self, k,device):
        super(find_k_nearest_neighbors, self).__init__()
        self.k=k
        self.device=device

    def forward(self,obs_his, era_his, pan_fut, cobs, cera, cpan):

        """
        Find the k nearest neighbors for each obs_his station (N) from the era_his and pan_fut data points.

        Parameters:
            obs_his: ndarray, historical observation data of shape (B, C, N, L)
            era_his: ndarray, ERA historical data of shape (B, C, lat, lon, L)
            pan_fut: ndarray, PAN future data of shape (B, C, lat, lon, L)
            cobs: ndarray, station coordinates of shape (N, 2) (latitude, longitude)
            cera: ndarray, ERA grid coordinates of shape (lat, lon, 2) (latitude, longitude)
            cpan: ndarray, PAN grid coordinates of shape (lat, lon, 2) (latitude, longitude)
            k: int, number of nearest neighbors to find

        Returns:
            era_k: ndarray, ERA neighbor data of shape (B, C, N, k, L)
            pan_k: ndarray, PAN neighbor data of shape (B, C, N, k, L)
        """

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
    def __init__(
            self,args,predefined_A):
        super().__init__()

        self.num_nodes = args.num_nodes
        self.input_dim = args.in_dim
        self.hidden_dim = args.d_model
        self.output_dim = 1
        self.out_steps = args.pre_len
        self.num_layers = 1
        self.cheb_k = 2
        self.ycov_dim = 1
        self.node_embedding_dim =16
        self.st_embedding_dim = 16
        self.tf_decay_steps = 4000
        self.tod_embedding_dim = 8
        self.dow_embedding_dim = 8
        self.use_teacher_forcing = True
        self.device=args.device
        self.target=args.target

        self.encoder_s = HimEncoder(
            self.num_nodes,
            self.input_dim,
            self.hidden_dim,
            self.cheb_k,
            self.num_layers,
            self.node_embedding_dim,
            meta_axis="S",
        )
        self.encoder_t = HimEncoder(
            self.num_nodes,
            self.input_dim,
            self.hidden_dim,
            self.cheb_k,
            self.num_layers,
            self.tod_embedding_dim + self.dow_embedding_dim,
            meta_axis="T",
        )

        self.decoder = HimDecoder(
            self.num_nodes,
            self.output_dim + self.ycov_dim,
            self.hidden_dim,
            self.cheb_k,
            self.num_layers,
            self.st_embedding_dim,
        )

        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

        self.tod_embedding = nn.Embedding(8, self.tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        self.node_embedding = nn.init.xavier_normal_(
            nn.Parameter(torch.empty(self.num_nodes, self.node_embedding_dim))
        )
        self.st_proj = nn.Linear(self.hidden_dim, self.st_embedding_dim)
        self.find_nearest_neighbors = find_k_nearest_neighbors(1, self.device)

    def compute_sampling_threshold(self, batches_seen):
        return self.tf_decay_steps / (
                self.tf_decay_steps + np.exp(batches_seen / self.tf_decay_steps)
        )

    def forward(self, obs_his,era_his,pan_fut,index_his,csta,cera,cpan):
        labels = None
        batches_seen = None
        x=obs_his
        B,C,N,L=x.shape
        era_1, pan_1, cera_1 = self.find_nearest_neighbors(x, era_his, pan_fut, csta, cera, cpan)
        y_cov = pan_1[:,:,:,0,:].permute(0, 3, 2, 1)[:, :, :, [self.target]]
        x=x.permute(0,3,2,1)
        tod = index_his[:, -1, 1]
        dow = index_his[:, -1, 2]
        tod_emb = self.tod_embedding(
            (tod * 8).long()
        )  # (batch_size, tod_embedding_dim)

        dow_emb = self.dow_embedding(dow.long())  # (batch_size, dow_embedding_dim)
        time_embedding = torch.cat([tod_emb, dow_emb], dim=-1)  # (B, tod+dow)

        support = torch.softmax(
            torch.relu(self.node_embedding @ self.node_embedding.T), dim=-1
        )

        h_s, _ = self.encoder_s(x, support, self.node_embedding)  # B, T, N, hidden
        h_t, _ = self.encoder_t(x, support, time_embedding)  # B, T, N, hidden
        h_last = (h_s + h_t)[:, -1, :, :]  # B, N, hidden (last state)

        st_embedding = self.st_proj(h_last)  # B, N, st_emb_dim
        support = torch.softmax(
            torch.relu(torch.einsum("bnc,bmc->bnm", st_embedding, st_embedding)),
            dim=-1,
        )

        ht_list = [h_last] * self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device)
        out = []
        for t in range(self.out_steps):
            h_de, ht_list = self.decoder(
                torch.cat([go, y_cov[:, t, ...]], dim=-1),
                ht_list,
                support,
                st_embedding,
            )
            go = self.out_proj(h_de)
            out.append(go)

        output = torch.stack(out, dim=1).permute(0,3,2,1)

        return output