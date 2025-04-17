import torch.optim as optim
from script.metric import *
from models import ASTAFN, HimNet,DUQ,HimNet_S,MegaCRN_S,MegaCRN,MPNN


class Trainer():
    def __init__(self,args,predefined_A,target_scaler):
        self.model_dict={'ASTAFN':ASTAFN,'DUQ':DUQ,'HimNet':HimNet,'HimNet_S':HimNet_S,'MegaCRN_S':MegaCRN_S,'MPNN':MPNN,'MegaCRN':MegaCRN}
        self.model=self._build_model(args,predefined_A).to(args.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss = masked_mae
        self.res_scaler = target_scaler
        self.model_type=args.model
        self.device=args.device


    def _build_model(self,args,predefined_A):
        model=self.model_dict[args.model].Model(args,predefined_A).float()
        return model

    def train(self, obs_his,obs_fut,era_his,pan_fut,csta,cera,cpan,target,index):
        self.model.train()
        self.optimizer.zero_grad()
        if self.model_type=='HimNet_S' or self.model_type=='HimNet':
            output= self.model(obs_his,era_his,pan_fut,index,csta,cera,cpan)
        else:
            output = self.model(obs_his, era_his, pan_fut, csta, cera, cpan)
        real = obs_fut[:,[target],:,:]
        predict = self.res_scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        loss.backward()
        self.optimizer.step()
        mae, rmse, pear, r, smape, fss=metric(predict, real)
        return loss.item(), mae, rmse,pear,r,smape,fss

    def eval(self,obs_his,obs_fut,era_his,pan_fut,csta,cera,cpan,target,index):
        self.model.eval()
        if self.model_type == 'HimNet_S' or self.model_type == 'HimNet':
            output = self.model(obs_his, era_his, pan_fut, index, csta, cera, cpan)
        else:
            output = self.model(obs_his, era_his, pan_fut, csta, cera, cpan)
        real = obs_fut[:, [target], :, :]
        predict = self.res_scaler.inverse_transform(output)
        loss = self.loss(predict, real, 0.0)
        mae, rmse, pear, r, smape, fss = metric(predict, real)
        return loss.item(), mae, rmse, pear, r, smape, fss
