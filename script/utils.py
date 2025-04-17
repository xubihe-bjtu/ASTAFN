import torch
import numpy as np
import pickle
import scipy.sparse as sp
import os
from tabulate import tabulate
from datetime import datetime
from sklearn.neighbors import NearestNeighbors


def load_adj(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def asym_adj(adj):
    """Asymmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)  # 将邻接矩阵稀疏化, 只将非零数据以列表形式保存 如(206, 155)	0.9271291   (206, 159)	0.78106517
    rowsum = np.array(adj.sum(1)).flatten()  # 计算每一行的和 rowsum 形状 [1,207]
    d_inv = np.power(rowsum, -1).flatten()  # 对rowsum取倒数,数值变成0-1之间,越小说明关注度越高
    d_inv[np.isinf(d_inv)] = 0.  # 把无穷小转化为0,d_inv表示每个传感器重要度倒数的列表
    d_mat = sp.diags(d_inv)  # 使用d_inv初始化对角矩阵d_mat
    return d_mat.dot(adj).astype(np.float32).todense()  # 矩阵乘法 d_mat * adj

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std,device):
        self.mean = mean
        self.std = std
        self.device = device

    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * torch.tensor(self.std).to(device=self.device)) + torch.tensor(self.mean).to(device=self.device)
        #return (data * self.std) + self.mean

def create_save_path(base_path,model_name):
    model_folder = os.path.join(base_path, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)  # 如果文件夹不存在则创建
    print(f"Directory created successfully at: {model_folder}")
    return model_folder

def save_log_file(log_file_path,content):
    f = open(log_file_path, 'w')
    f.write(content + '\n')
    return f


def print_args(args,log_file_path):
    # Convert the args namespace to a dictionary
    args_dict = vars(args)
    # Create a list of tuples from the dictionary
    args_list = [(key, value) for key, value in args_dict.items()]
    # Print the arguments using tabulate for better formatting
    print(tabulate(args_list, headers=["Argument", "Value"], tablefmt="pretty"))
    format_Args = tabulate(args_list, headers=["Argument", "Value"], tablefmt="pretty")
    # Save the formatted arguments to a log file
    f=save_log_file(log_file_path, format_Args)
    return f

#恢复shuffle的索引
def restore_order(shuffled_indices,predictions):
    restored_indices = torch.argsort(torch.tensor(shuffled_indices))
    return predictions[restored_indices]

def save_experiment_result(model_name, feature_name, feature_data, base_path):
    model_folder = os.path.join(base_path, model_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)  # 如果文件夹不存在则创建

    feature_file = os.path.join(model_folder, f'{feature_name}_prediction.npy')
    np.save(feature_file, feature_data)
    print(f"Experiment Results Saved to {feature_file}")

def find_k_nearest_neighbors(era_data, cobs, cera,N,k=1):

        """
        找到每个 obs_his 站点 (N) 的 k 个近邻的 era_his 和 pan_fut 数据点。
            era_his: ndarray, (B, C, lat, lon, L) 的 ERA 历史数据
            cobs: ndarray, (N, 2) 的站点坐标 (纬度, 经度)
            cera: ndarray, (lat, lon, 2) 的 ERA 网格坐标 (纬度, 经度)
            k: int, 要找到的近邻数量

        返回:
            era_k: ndarray, (B, C, N, k, L) 的 ERA 近邻数据
            pan_k: ndarray, (B, C, N, k, L) 的 PAN 近邻数据
        """
        #将era5展平
        B,C,_,_,L=era_data.shape
        era_his=era_data.reshape(B,C,-1,L)
        # cera 和 cpan
        cera_flat = cera.reshape(-1, 2)  # (lat * lon, 2)

        # 初始化最近邻模型
        nbrs_era = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(cera_flat)

        era_k=[]
        for n in range(N):
            # 获取当前 obs_his 站点的坐标
            station_coord = np.array(cobs[n]).reshape(1,2)  # (2,)

            # 获取该站点最近的 k 个 ERA 和 PAN 网格点索引
            _, indices_era = nbrs_era.kneighbors(station_coord)
            era_his_n=era_his[:,:,indices_era,:]#era_his:(B,C,1,k,L)
            era_k.append(era_his_n)
        era_k=np.concatenate(era_k,axis=2)#era_k:(B,C,N,1,L)
        return era_k