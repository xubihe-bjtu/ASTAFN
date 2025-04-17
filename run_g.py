import warnings
import argparse
import os
from tqdm import tqdm
from data_provider.load_dataset import load_dataset
from script.utils import load_adj,create_save_path,print_args,save_experiment_result
from Trainer.trainer import Trainer
from script.metric import *

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
warnings.filterwarnings('ignore')
parser=argparse.ArgumentParser()
parser.add_argument('--cpan_path', type=str, default='./dataset/GUANGDONG/pan_lat_lon.npy', help='Pangu Latitude and Longitude File')
parser.add_argument('--cera_path', type=str, default='./dataset/GUANGDONG/era_lat_lon.npy', help='ERA5 Latitude and Longitude File')
parser.add_argument('--csta_path', type=str, default='./dataset/GUANGDONG/obs_lat_lon.npy', help='Observation station Latitude and Longitude File')
parser.add_argument('--save_path', type=str, default='./save/GUANGDONG', help='Result Save File Path')
parser.add_argument('--data_path', type=str, default='./dataset/GUANGDONG', help='Input Data Path')
parser.add_argument('--adj_path', type=str, default='./dataset/GUANGDONG/sensor_graph/adj_mat.pkl', help='Observation Station Adjacency Matrix File Path')

parser.add_argument('--device',type=str,default='cuda:4',help='')
parser.add_argument('--runs', type=int, default=3, help='Number of Runs')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--k', type=int, default=6,help='k Nearest Neighbors for Proxy Station')
parser.add_argument('--area',type=str,default='GUANGDONG',help='region')
parser.add_argument('--target', type=int, default=0, help='[u,v,msl,tmp]')
parser.add_argument('--d_align', type=int, default=16, help='')
parser.add_argument('--num_layer', type=int, default=2, help='Number of Encoder Layers')
parser.add_argument('--seq_len', type=int, default=8, help='Input Sequence Length')
parser.add_argument('--pre_len',type=int,default=8,help='Output Sequence Length')
parser.add_argument('--d_model',type=int,default=64,help='Embedding Dimension')
parser.add_argument('--in_dim',type=int,default=4,help='inputs dimension')
parser.add_argument('--out_dim',type=int,default=1,help='outputs dimension')
parser.add_argument('--pan_in_dim',type=int,default=4,help='inputs dimension')
parser.add_argument('--model',type=str,default='MPNN',help='model type')
parser.add_argument('--lr', type=float, default=0.001, help='Training Learning Rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--num_nodes', type=int, default=24,help='Number of Observation Stations')
parser.add_argument('--epochs', type=int, default=1, help='Training Epochs')
parser.add_argument('--print_every',type=int,default=100,help='')
parser.add_argument('--mark_dim',type=int,default=5,help='')
parser.add_argument('--gcn_bool',action='store_true',default=True,help='whether to add graph convolution layer')
parser.add_argument('--addaptadj',action='store_true',default=True,help='whether add adaptive adj')
args = parser.parse_args()


mean_mae,mean_rmse,mean_pear,mean_r,mean_smape,mean_fss=[],[],[],[],[],[]
var_mae,var_rmse,var_pear,var_r,var_smape,var_fss=[],[],[],[],[],[]

device=torch.device(args.device)
# Create Model Save Path
run_dir = create_save_path(args.save_path, args.model)
# Print Parameters and Save to Work Log
log_file_path = run_dir + '/work_log.txt'
log_file=print_args(args, log_file_path)

target_list=['u','v','msl','tmp']
target_name=target_list[args.target]
#Generate Predefined Graph
predefined_A = load_adj(pkl_filename = args.adj_path)
predefined_A = [torch.tensor(adj).to(device) for adj in predefined_A]

dataloader = load_dataset(args.data_path, args.target,args.batch_size, args)
obs_scaler,era_scaler,pan_scaler,target_scaler = dataloader['obs_scaler'],dataloader['era_scaler'],dataloader['pan_scaler'],dataloader['target_scaler']

for _ in range(args.runs):
    # Initialize Training Model
    engine = Trainer(args,predefined_A,target_scaler)

    # Start Training
    print("-------------------Start Training--------------------\n")
    log_file.write("-------------------Start Training--------------------\n")


    his_loss = []
    minl = 1e5
    epoch_best = -1
    pbar = tqdm(range(args.epochs))

    for i in pbar:
        train_loss, train_mae, train_rmse, train_pear, train_r, train_smape, train_fss = [], [], [], [], [], [],[]

        #Shuffle Training Data
        dataloader['train_loader'].shuffle()

        for iter, (obs_his, obs_fut,era_his,pan_fut,index,csta,cera,cpan) in enumerate(dataloader['train_loader'].get_iterator()):
            obs_his = torch.Tensor(obs_his.astype(float)).to(device).permute(0, 3, 2, 1)  # Tensor:(B,C,N,L)
            obs_fut = torch.Tensor(obs_fut.astype(float)).to(device).permute(0, 3, 2, 1)  # Tensor:(B,C,N,L)
            era_his = torch.Tensor(era_his.astype(float)).to(device).permute(0, 4, 2, 3, 1)  # Tensor:(B,C,lat,lon,L)
            pan_fut = torch.Tensor(pan_fut.astype(float)).to(device).permute(0, 4, 2, 3, 1)  # Tensor:(B,C,lat,lon,L)
            index=torch.Tensor(index.astype(float)).to(device)
            csta=csta.astype(float)
            cera=cera.astype(float)
            cpan=cpan.astype(float)

            metrics = engine.train(obs_his,obs_fut,era_his,pan_fut,csta,cera,cpan,args.target,index)
            train_loss.append(metrics[0])
            train_mae.append(metrics[1])
            train_rmse.append(metrics[2])
            train_pear.append(metrics[3])
            train_r.append(metrics[4])
            train_smape.append(metrics[5])
            train_fss.append(metrics[6])

            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}, train_mae  {:.4f}, train_rmse  {:.4f}, train_pear  {:.4f}, train_r  {:.4f}, train_smape  {:.4f}, train_fss  {:.4f}\n'
                print(log.format(iter, train_loss[-1], train_mae[-1], train_rmse[-1], train_pear[-1], train_r[-1], train_smape[-1], train_fss[-1]), flush=True)
                log_file.write(log.format(iter, train_loss[-1], train_mae[-1], train_rmse[-1], train_pear[-1], train_r[-1], train_smape[-1], train_fss[-1]))
        pbar.set_description("Training Processing %s" % int(i + 1))

        # Start Validation
        val_loss, val_mae, val_rmse, val_pear, val_r, val_smape, val_fss = [], [], [], [], [], [],[]
        for iter, (obs_his, obs_fut,era_his,pan_fut,index,csta,cera,cpan) in enumerate(dataloader['val_loader'].get_iterator()):
            obs_his = torch.Tensor(obs_his.astype(float)).to(device).permute(0, 3, 2, 1)  # Tensor:(B,C,N,L)
            obs_fut = torch.Tensor(obs_fut.astype(float)).to(device).permute(0, 3, 2, 1)  # Tensor:(B,C,N,L)
            era_his = torch.Tensor(era_his.astype(float)).to(device).permute(0, 4, 2, 3, 1)  # Tensor:(B,C,lat,lon,L)
            pan_fut = torch.Tensor(pan_fut.astype(float)).to(device).permute(0, 4, 2, 3, 1)  # Tensor:(B,C,lat,lon,L)
            index = torch.Tensor(index.astype(float)).to(device)
            csta = csta.astype(float)
            cera = cera.astype(float)
            cpan = cpan.astype(float)

            metrics = engine.eval(obs_his,obs_fut,era_his,pan_fut,csta,cera,cpan,args.target,index)
            val_loss.append(metrics[0])
            val_mae.append(metrics[1])
            val_rmse.append(metrics[2])
            val_pear.append(metrics[3])
            val_r.append(metrics[4])
            val_smape.append(metrics[5])
            val_fss.append(metrics[6])

        # Compute Average Metrics
        mtrain_loss, mtrain_mae, mtrain_rmse, mtrain_pear, mtrain_r, mtrain_smape, mtrain_fss = (
            np.mean(train_loss), np.mean(train_mae), np.mean(train_rmse), np.mean(train_pear), np.mean(train_r), np.mean(train_smape), np.mean(train_fss))
        mval_loss, mval_mae, mval_rmse, mval_pear, mval_r, mval_smape, mval_fss = (
            np.mean(val_loss), np.mean(val_mae), np.mean(val_rmse), np.mean(val_pear), np.mean(val_r), np.mean(val_smape), np.mean(val_fss))

        his_loss.append(mval_loss)

        # Save the Model with the Lowest Validation Loss
        if mval_loss < minl:
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)  # 如果文件夹不存在则创建
            pth_path=run_dir+'/'+ target_name+'_'+ "best_epoch" + ".pth"
            torch.save(engine.model.state_dict(),pth_path)
            print('Model Saved')
            minl = mval_loss
            epoch_best = i

    # Save the Epoch with the Lowest Loss
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(
        torch.load(pth_path, map_location='cpu'))
    print("The valid loss on best model is {}, epoch:{}\n".format(str(round(his_loss[bestid], 4)), epoch_best))
    log_file.write("The valid loss on best model is {}, epoch:{}\n".format(str(round(his_loss[bestid], 4)), epoch_best))
    print("Training finished\n")

    #Start testing
    outputs = []
    realy = torch.Tensor(dataloader['obs_fut_test'].astype(float)).to(device).permute(0, 3, 2, 1)#B,C,N,L

    for iter, (obs_his, obs_fut,era_his,pan_fut,index,csta,cera,cpan) in enumerate(
            dataloader['test_loader'].get_iterator()):
        obs_his = torch.Tensor(obs_his.astype(float)).to(device).permute(0, 3, 2, 1)  # Tensor:(B,C,N,L)
        obs_fut = torch.Tensor(obs_fut.astype(float)).to(device).permute(0, 3, 2, 1)  # Tensor:(B,C,N,L)
        era_his = torch.Tensor(era_his.astype(float)).to(device).permute(0, 4, 2, 3, 1)  # Tensor:(B,C,lat,lon,L)
        pan_fut = torch.Tensor(pan_fut.astype(float)).to(device).permute(0, 4, 2, 3, 1)  # Tensor:(B,C,lat,lon,L)
        index = torch.Tensor(index.astype(float)).to(device)
        csta = csta.astype(float)
        cera = cera.astype(float)
        cpan = cpan.astype(float)

        with torch.no_grad():
            if args.model == 'HimNet_S' or args.model == 'HimNet':
                preds = engine.model(obs_his, era_his, pan_fut, index, csta, cera, cpan)
            else:
                preds = engine.model(obs_his, era_his, pan_fut, csta, cera, cpan)

        outputs.append(preds)
    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]
    yhat_inv = target_scaler.inverse_transform(yhat).permute(0, 3, 2, 1)
    yhat_data=yhat_inv.reshape(-1,args.num_nodes).cpu().numpy()
    save_experiment_result(args.model, target_name, yhat_data,args.save_path)

    test_mae, test_rmse, test_pear, test_r, test_smape, test_fss = [], [], [], [], [],[]

    for step in range(args.pre_len):
        pred = target_scaler.inverse_transform(yhat[:, :,:, step])#B,1,N
        real = realy[:, [args.target], :, step]
        metrics = metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, test_mae:{:.4f}, test_rmse:{:.4f}, test_pear:{:.4f}, test_r:{:.4f}, test_smape:{:.4f}, test_fss:{:.4f}\n'
        print(log.format(step+1, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4],metrics[5]))
        log_file.write(log.format(step + 1, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4],metrics[5]))
        test_mae.append(metrics[0])
        test_rmse.append(metrics[1])
        test_pear.append(metrics[2])
        test_r.append(metrics[3])
        test_smape.append(metrics[4])
        test_fss.append(metrics[5])

    log = 'On average over {:} horizons, test_mae:{:.4f}, test_rmse:{:.4f}, test_pear:{:.4f}, test_r:{:.4f}, test_smape:{:.4f}, test_fss:{:.4f}\n'
    print(log.format(args.pre_len, np.mean(test_mae), np.mean(test_rmse), np.mean(test_pear), np.mean(test_r), np.mean(test_smape), np.mean(test_fss)))
    log_file.write(log.format(args.pre_len, np.mean(test_mae), np.mean(test_rmse), np.mean(test_pear), np.mean(test_r), np.mean(test_smape), np.mean(test_fss)))

    mean_mae.append(np.mean(test_mae))
    mean_rmse.append(np.mean(test_rmse))
    mean_pear.append(np.mean(test_pear))
    mean_r.append(np.mean(test_r))
    mean_smape.append(np.mean(test_smape))
    mean_fss.append(np.mean(test_fss))
    var_mae.append(np.mean(test_mae))
    var_rmse.append(np.mean(test_rmse))
    var_pear.append(np.mean(test_pear))
    var_r.append(np.mean(test_r))
    var_smape.append(np.mean(test_smape))
    var_fss.append(np.mean(test_fss))

mean_mae,mean_rmse,mean_pear,mean_r,mean_smape,mean_fss = np.mean(np.array(mean_mae)), np.mean(np.array(mean_rmse)), np.mean(
    np.array(mean_pear)), np.mean(np.array(mean_r)), np.mean(np.array(mean_smape)),np.mean(np.array(mean_fss))
var_mae,var_rmse,var_pear,var_r,var_smape,var_fss = np.std(np.array(var_mae)), np.std(np.array(var_rmse)), np.std(
    np.array(var_pear)), np.std(np.array(var_r)), np.std(np.array(var_smape)),np.std(np.array(var_fss))

metric_save_path = run_dir+'/'+target_name+'_'+'metric.txt'
f = open(metric_save_path, 'w')
log = 'On average over {:} horizons Mean Values, mean_mae: {:.4f},mean_rmse: {:.4f},mean_pear: {:.4f},mean_r: {:.4f},mean_smape: {:.4f},mean_fss: {:.4f}\n'
#print(log.format(args.pre_len, mean_mae,mean_rmse,mean_pear,mean_r,mean_smape,mean_fss))
f.write(log.format(args.pre_len, mean_mae,mean_rmse,mean_pear,mean_r,mean_smape,mean_fss))
log = 'On average over {:} horizons std Values, var_mae: {:.4f},var_rmse: {:.4f},var_pear: {:.4f},var_r: {:.4f},var_smape: {:.4f},var_fss: {:.4f}\n'
#print(log.format(args.pre_len, var_mae,var_rmse,var_pear,var_r,var_smape,var_fss))
f.write(log.format(args.pre_len, var_mae,var_rmse,var_pear,var_r,var_smape,var_fss))
log_file.close()
f.close()