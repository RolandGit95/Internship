# execute with qsub main.py -epochs 16 -batch_size 2 -config config/training_two_spirals.yaml -time_steps 1 2 4 8 16 20 25 28 30 32 -depth 32
# this loops multiple trainings with different time-steps. See in the config file for further specifications
#! python

# name
#$ -N training

# execute from current directory
#$ -cwd

# Preserve environment variables
#$ -V

# Provide path to python executable, has to provide the packages from requirements.txt
#$ -S /home/stenger/smaxxhome/anaconda3/envs/gpu/bin/python

# Merge error and out
#$ -j yes

# serial queue
#$ -q taranis-gpu1.q

# Path for output
#$ -o /home/stenger/smaxxhome/outputs

import os 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import numpy as np

from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
import yaml

import wandb

from modules import STLSTM
from datasets import BarkleyDataset

import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Found', torch.cuda.device_count(), 'GPUs')

wandb.login()

os.environ["WANDB_MODE"] = "dryrun"

torch.manual_seed(42)

# %%

def make(config):  
    model = nn.DataParallel(STLSTM(1,config['num_features'])).to(device)
        
    if config['load_weights']:
        try:
            model.load_state_dict(torch.load(config['weights_file'], map_location=device), strict=True)
        except FileNotFoundError:
            print('File not found, initialize new weights')
            
    
    train_dataset = BarkleyDataset(root=config['dataset_dir'], 
                                       train=True, 
                                       depth=config['depth'], 
                                       time_steps=config['time_step'])
    
    n_train = int(len(train_dataset)*0.90+0.5)
    n_val = int(len(train_dataset)*0.10+0.5)
        
    torch.manual_seed(42)
    train_dataset, test_dataset = random_split(train_dataset, [n_train, n_val])  
    test_dataset.train = False

    train_loader = DataLoader(train_dataset, config['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, 2, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    
    if config['loss_fn'] =='MSE':
        criterion = nn.MSELoss()
    elif config['loss_fn']=='MAE':
        criterion = nn.L1Loss()
        
    if config['optimizer'] =='Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer']=='SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'])
    
    return model, train_loader, test_loader, criterion, optimizer

# %%

def train(model, train_dataloader, test_dataloader, criterion, optimizer, config, val_fn=nn.MSELoss()):
    #torch.save(model.state_dict(), 'model')
    def get_lr():
        for param_group in optimizer.param_groups:
            return param_group['lr']
    
    lrp = ReduceLROnPlateau(optimizer, patience=512, factor=0.3, min_lr=1e-7, verbose=True)
        
    test_dataloader_iter = iter(test_dataloader)
    
    min_val_loss = 10000      
    val_losses = []
    
    
    print(config['save_name'] )
    for epoch in range(config['epochs']):
        for i, (X,y) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            model.zero_grad()
            optimizer.zero_grad()
            
            X = X.to(device)
            y = y.to(device)
        
            outputs = model(X, max_depth=config['depth'])

            loss = 0.0
            loss += criterion(y, outputs) 
            
            outputs = outputs.detach()
            
            loss.backward()
            optimizer.step()
            
            try:
                X_val, y_val = next(test_dataloader_iter)
            except StopIteration:
                test_dataloader_iter = iter(test_dataloader)
                X_val, y_val = next(test_dataloader_iter)
            X_val = X_val.to(device)
            y_val = y_val.to(device)
            
            with torch.no_grad():
                val_outputs = model(X_val, max_depth=config['depth'])
                val_loss = val_fn(y_val, val_outputs)
                val_losses.append(val_loss.cpu().detach().numpy())
            lrp.step(val_loss)
            
            if config['wandb']:
                wandb.log({"loss": loss, "val_loss":val_loss})#, "val_loss": val_loss, "learning_rate":get_lr(), "last_16_mean_loss": np.mean(val_losses[-16:])})
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            
            name = config['save_name'] + '_t' + str(config['time_step']) + '_d' + str(config['depth'])
            
            try:
                os.makedirs(config['save_dir'])
            except FileExistsError:
                pass

            savename = os.path.join(config['save_dir'], name)
            print('Save model under:', savename)
            torch.save(model.state_dict(), savename)
            
        
            
# %%

def pipeline(config):   
    if config['wandb']:
        wandb.init(project=config['project_name'], 
                   name=config['name'] + '_t' + str(config['time_step']) + '_d' + str(config['depth']), dir=config['wandb_dir'], 
                   config=config, 
                   reinit=True)
        
        config = wandb.config
    model, train_dataloader, test_dataloader, criterion, optimizer = make(config)
    if config['wandb']:
        wandb.watch(model, criterion, log="all", log_freq=32)
    train(model, train_dataloader, test_dataloader, criterion, optimizer, config, val_fn=nn.L1Loss())

    #run.finish()
    return model

# %%

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Training of Neural Networks, the Barkley Diver')

    # Names
    parser.add_argument('-project_name', '--project_name', type=str, help='', default='unnamed')
    parser.add_argument('-name', '--name', type=str, help='', default='unnamed')
    
    # Model specifications
    parser.add_argument('-architecture', '--architecture', type=str, help='', default='STLSTM')
    parser.add_argument('-num_features', '--num_features', type=int, help='', default=64)
    
    # Data
    parser.add_argument('-dataset', '--dataset', type=str, help='', default='concentric')
    parser.add_argument('-time_step', '--time_step', type=int, help='Time Steps', default=3)
    parser.add_argument('-depth', '--depth', type=int, help='Depth', default=2)
    
    parser.add_argument('-time_steps', '--time_steps', type=list, nargs='+', help='list of Time Steps', default=None)
    parser.add_argument('-depths', '--depths', type=list, nargs='+', help='List of Depth', default=None)
    
    # Training process
    parser.add_argument('-epochs', '--epochs', type=int, help='', default=1)
    parser.add_argument('-batch_size', '--batch_size', type=int, help='', default=3)
    parser.add_argument('-lr', '--lr', type=float, help='Learning Rate', default=0.001)
    
    parser.add_argument('-loss_fn', '--loss_fn', type=str, help='', default='MSE')
    parser.add_argument('-optimizer', '--optimizer', type=str, help='', default='Adam')
    
    # Config files
    parser.add_argument('-config', '--config', type=str, help='Place of config file', default=None)
    parser.add_argument('-sweep', '--sweep', type=str, help='Set path to sweep file to define gridsearch parameter', default=None)
    
    # Logger
    parser.add_argument('-wandb', '--wandb', type=int, help='', default=True)
    parser.add_argument('-offline', '--offline', type=int, help='', default=False)
    parser.add_argument('-wandb_dir', '--wandb_dir', type=str, help='', default='./')
    
    # Load and save
    parser.add_argument('-load_weights', '--load_weights', type=int, help='Load previously trained weights', default=False)
    parser.add_argument('-weights_file', '--weights_file', type=str, help='If load_weights==True, set the location of the weights file', default='./model')
    parser.add_argument('-save_name', '--save_name', type=str, help='', default='model')
    parser.add_argument('-save_dir', '--save_dir', type=str, help='', default='./')
    
    parser.add_argument('-dataset_dir', '--dataset_dir', type=str, help='', default=None)
    
    args = parser.parse_args()
    time_steps = args.time_steps
    depths = args.depths
    
    args_config = vars(args)
        
    specified_config = dict()
    for key, value in vars(args).items():
        if parser.get_default(key)!=value:
            specified_config[key] = value    
            
    
    if not isinstance(args.config, type(None)):
        try:
            with open(args.config) as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
                args_config.update(config)
        except FileNotFoundError:
            print('Config-file not found, use default values')
            assert('Config-file not found, use default values')     

    args_config.update(specified_config)
    
    if int(args.offline)==True:
        print('No internet')
        os.environ['WANDB_MODE'] = 'dryrun'
        WANDB_MODE="dryrun"

    if not isinstance(args.depths, type(None)):
        for d in args.depths:
            d_int = int(''.join(d))
            args_config['depth'] = d_int
            if not isinstance(args.time_steps, type(None)):
                for t in args.time_steps:
                    t_int = int(''.join(t))
                    args_config['time_step'] = t_int  
                    
                    for key, value in args_config.items():
                        print(key + ':', value)
        
                    m = pipeline(args_config)
            else:
                for key, value in args_config.items():
                    print(key + ':', value)
                    
                m = pipeline(args_config)
    elif not isinstance(args.time_steps, type(None)):
        for t in args.time_steps:
            t_int = int(''.join(t))
            args_config['time_step'] = t_int     
            for key, value in args_config.items():
                print(key + ':', value)
            m = pipeline(args_config)
    else:
        for key, value in args_config.items():
            print(key + ':', value)
        m = pipeline(args_config)
