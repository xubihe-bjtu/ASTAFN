import numpy as np
import os
from script.utils import StandardScaler
import torch


def load_dataset(dataset_dir,target,batch_size,args):
    data={}
    for category in ['train','val','test']:
        cat_data=np.load(os.path.join(dataset_dir,category+'.npz'),allow_pickle=True)
        data['obs_his_'+category]=cat_data['obs_his'][:,:,:,:args.in_dim].astype(float)
        data['obs_fut_' + category] = cat_data['obs_fut'][:,:,:,:args.in_dim].astype(float)
        data['era_his_'+category]=cat_data['era_his'][:,:,:,:,:args.in_dim].astype(float)
        data['pan_fut_'+category]=cat_data['pan_fut'][:,:,:,:,:args.in_dim].astype(float)
        data['index_' + category] = np.load(os.path.join(dataset_dir,'index_' +category+'.npz'),allow_pickle=True)['obs_his']


    #标准化scaler计算
    obs_mean=[data['obs_his_train'][:,:,:,i].mean() for i in range(args.in_dim)]
    obs_std=[data['obs_his_train'][:,:,:,i].std() for i in range(args.in_dim)]
    obs_scaler=StandardScaler(mean=obs_mean,std=obs_std,device=args.device)
    pan_mean = [data['pan_fut_train'][:, :, :, :,i].mean() for i in range(args.in_dim)]
    pan_std = [data['pan_fut_train'][:, :, :,:, i].std() for i in range(args.in_dim)]
    pan_scaler = StandardScaler(mean=pan_mean, std=pan_std,device=args.device)
    era_mean=[data['era_his_train'][:, :, :,:, i].mean() for i in range(args.in_dim)]
    era_std = [data['era_his_train'][:, :, :, :,i].std() for i in range(args.in_dim)]
    era_scaler = StandardScaler(mean=era_mean, std=era_std,device=args.device)
    target_mean=data['obs_his_train'][:,:,:,[target]].mean()
    target_std=data['obs_his_train'][:,:,:,[target]].std()
    target_scaler = StandardScaler(mean=target_mean, std=target_std,device=args.device)


    #对训练集/测试集/验证集的数据进行标准化操作
    for category in ['train','val','test']:
        data['obs_his_'+category]=obs_scaler.transform(data['obs_his_'+category])
        data['era_his_' + category] = era_scaler.transform(data['era_his_' + category])
        data['pan_fut_'+category]=pan_scaler.transform(data['pan_fut_'+category])

    #获取csta,cera,cpan
    csta=np.load(args.csta_path,allow_pickle=True)
    cera=np.load(args.cera_path,allow_pickle=True)
    cpan=np.load(args.cpan_path,allow_pickle=True)


    data['train_loader']=DataLoader(data['obs_his_train'],
                                    data['obs_fut_train'],
                                    data['era_his_train'],
                                    data['pan_fut_train'],
                                    data['index_train'],
                                    csta,cera,cpan,
                                    batch_size)
    data['val_loader'] = DataLoader(data['obs_his_val'],
                                    data['obs_fut_val'],
                                    data['era_his_val'],
                                    data['pan_fut_val'],
                                    data['index_val'],
                                    csta, cera, cpan,
                                    batch_size)
    data['test_loader'] = DataLoader(data['obs_his_test'],
                                    data['obs_fut_test'],
                                    data['era_his_test'],
                                    data['pan_fut_test'],
                                     data['index_test'],
                                    csta, cera, cpan,
                                     batch_size)

    data['obs_scaler']=obs_scaler
    data['pan_scaler']=pan_scaler
    data['era_scaler']=era_scaler
    data['target_scaler']=target_scaler
    return data

class DataLoader(object):
    def __init__(self,obs_his,obs_fut,era_his,pan_fut,index,csta,cera,cpan,batch_size,pad_with_last_sample=True):
        self.batch_size=batch_size
        self.current_ind=0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(obs_his) % batch_size)) % batch_size
            obs_his_padding = np.repeat(obs_his[-1:], num_padding, axis=0)
            obs_fut_padding = np.repeat(obs_fut[-1:], num_padding, axis=0)
            era_his_padding = np.repeat(era_his[-1:], num_padding, axis=0)
            pan_fut_padding = np.repeat(pan_fut[-1:], num_padding, axis=0)
            index_padding = np.repeat(index[-1:], num_padding, axis=0)

            obs_his = np.concatenate([obs_his, obs_his_padding], axis=0)
            obs_fut=np.concatenate([obs_fut,obs_fut_padding],axis=0)
            era_his = np.concatenate([era_his, era_his_padding], axis=0)
            pan_fut = np.concatenate([pan_fut, pan_fut_padding], axis=0)
            index = np.concatenate([index, index_padding], axis=0)


        self.size=len(obs_his)
        self.num_batch=int(self.size//self.batch_size)
        self.obs_his=obs_his
        self.obs_fut=obs_fut
        self.era_his=era_his
        self.pan_fut=pan_fut
        self.index=index
        self.obs_his_s = obs_his
        self.obs_fut_s = obs_fut
        self.era_his_s = era_his
        self.pan_fut_s = pan_fut
        self.index_s=index
        self.csta=csta
        self.cera=cera
        self.cpan=cpan


    def shuffle(self):
        self.original_indices=np.arange(self.size)
        permutation = torch.randperm(self.size)

        obs_his = self.obs_his[permutation]
        obs_fut = self.obs_fut[permutation]
        era_his = self.era_his[permutation]
        pan_fut = self.pan_fut[permutation]
        index = self.index[permutation]

        self.obs_his_s = obs_his
        self.obs_fut_s = obs_fut
        self.era_his_s = era_his
        self.pan_fut_s = pan_fut
        self.index_s=index
        self.shuffled_indices=permutation

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                obs_his_i = self.obs_his_s[start_ind: end_ind, ...]
                obs_fut_i = self.obs_fut_s[start_ind: end_ind, ...]
                era_his_i = self.era_his_s[start_ind: end_ind, ...]
                pan_fut_i = self.pan_fut_s[start_ind: end_ind, ...]
                index_i = self.index_s[start_ind: end_ind, ...]

                csta_i=self.csta
                cera_i=self.cera
                cpan_i=self.cpan

                yield (obs_his_i,obs_fut_i,era_his_i,pan_fut_i,index_i,csta_i,cera_i,cpan_i)
                self.current_ind += 1
        return _wrapper()