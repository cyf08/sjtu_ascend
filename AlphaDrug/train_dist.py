'''
Author: QHGG
Date: 2022-08-22 15:54:41
LastEditTime: 2022-08-22 16:51:51
LastEditors: QHGG
Description: 
FilePath: /AlphaDrug/train.py
'''
import torch
import json
import os
import time
import argparse
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from loguru import logger
from utils.baseline import prepareDataset
from utils.baseline import loadConfig
from utils.log import prepareFolder, trainingVis
from model.Lmser_Transformerr import MFT as DrugTransformer
# from model.Transformer import MFT as DrugTransformer
# from model.Transformer_Encoder import MFT as DrugTransformer

# For Ascend
import torch_npu
from torch_npu.contrib import transfer_to_npu
from torch_npu import optim

os.environ["LOCAL_RANK"]="4,5,6,7"
#local_rank = int(os.environ["LOCAL_RANK"])
local_rank=int('4')
#world_size = int(os.environ["WORLD_SIZE"])
print('-----local_rank------',local_rank)
#print('world_size',world_size)

def train(model, trainLoader, smiVoc, proVoc, device):
    batch = len(trainLoader)
    print('batch',batch)
    totalLoss = 0.0
    totalAcc = 0.0
    for protein, smile, label, proMask, smiMask in tqdm(trainLoader):
    
        protein = torch.as_tensor(protein).to(device)
        smile = torch.as_tensor(smile).to(device)
        proMask = torch.as_tensor(proMask).to(device)
        smiMask = torch.as_tensor(smiMask).to(device)
        label = torch.as_tensor(label).to(device)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(model,smile.shape[1]).tolist()
        #tgt_mask = [tgt_mask] * batch *len(device_ids)
        tgt_mask = [tgt_mask] 
        print (np.array(tgt_mask).shape)
        tgt_mask = torch.as_tensor(tgt_mask).to(device)
        
       
        out = model(protein, smile, smiMask, proMask, tgt_mask)
        # tgt = torch.argmax(out, dim=-1)
        cacc = ((torch.eq(torch.argmax(out, dim=-1) * smiMask, label * smiMask).sum() - (smiMask.shape[0] * smiMask.shape[1] - (smiMask).sum())) / (smiMask).sum().float()).item()
        
        totalAcc += cacc
        loss = F.nll_loss(out.permute(0, 2, 1), label, ignore_index=smiVoc.index('^')) # mask padding

        # loss = vLoss
        totalLoss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # scheduler.step()

    avgLoss = round(totalLoss / batch, 3)
    avgAcc = round(totalAcc / batch, 3)

    return [avgAcc, avgLoss]


@torch.no_grad()
def valid(model, validLoader, smiVoc, proVoc, device):
    model.eval()
    
    batch = len(validLoader)
    totalLoss = 0
    totalAcc = 0
    for protein, smile, label, proMask, smiMask in tqdm(validLoader, position=0):
        protein = torch.as_tensor(protein).to(device)
        smile = torch.as_tensor(smile).to(device)
        proMask = torch.as_tensor(proMask).to(device)
        smiMask = torch.as_tensor(smiMask).to(device)
        label = torch.as_tensor(label).to(device)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(model,smile.shape[1]).tolist()
        #tgt_mask = [tgt_mask] * batch *len(device_ids)
        tgt_mask = [tgt_mask] 
        tgt_mask = torch.as_tensor(tgt_mask).to(device)

        out = model(protein, smile, smiMask, proMask, tgt_mask)
        
        # totalAcc += ((torch.eq(torch.argmax(out, dim=2) * smiMask, label * smiMask).sum() - (smiMask.shape[0] * smiMask.shape[1] - smiMask.sum())) / smiMask.sum()).item()
        
        cacc = ((torch.eq(torch.argmax(out, dim=-1) * smiMask, label * smiMask).sum() - (smiMask.shape[0] * smiMask.shape[1] - (smiMask).sum())) / (smiMask).sum().float()).item()
        
        totalAcc += cacc

        loss = F.nll_loss(out.permute(0, 2, 1), label, ignore_index=smiVoc.index('^')) # mask padding

        # loss = vLoss
        totalLoss += loss.item()
        
    avgLoss = round(totalLoss / batch, 3)
    avgAcc = round(totalAcc / batch, 3)

    return [avgAcc, avgLoss]
    

if __name__ == '__main__':
    torch_npu.npu.set_compile_mode(jit_compile=False)
    parser = argparse.ArgumentParser(description='settings')
    
    parser.add_argument('--layers', type=int, default=4, help='transformer layers')
    parser.add_argument('-l', action="store_true", help='learning rate')
    parser.add_argument('--epoch', type=int, default=501, help='epochs')
    parser.add_argument('--device', type=str, default='4,5,6,7', help='device')
    #parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argument('--local_rank', type=int, default='4', help='DDP parameter, do not modify')
    parser.add_argument('--pretrain', type=str, default='', help='pretrain model path')
    parser.add_argument('--bs', type=int, default=32, help='bs')
    parser.add_argument('--note', type=str, default='', help='note')
    args = parser.parse_args()

    startTime = time.time()

    config = loadConfig(args)
    print('config',config)
    
    exp_folder, model_folder, vis_folder = prepareFolder()
    logger.success(args)
    local_rank = args.local_rank
    device_ids = [int(i) for i in args.device.split(',') if i!='']
    print (len(device_ids))
    if len(args.device) == 1:
        device = torch.device("npu", int(args.device))
    else:
        print("###########local_rank############",local_rank)
        device = torch.device('npu', local_rank+4)
        print("###########device############",device)
        torch_npu.npu.set_device(device)
    print(device)
    #torch.distributed.init_process_group(backend="hccl", rank=local_rank)
    torch.distributed.init_process_group(backend="hccl", rank=local_rank)

    batchSize = args.bs * len(device_ids)
    epoch = args.epoch
    lr = 1e-3

    config.batchSize = batchSize
    print('config.batchSize',config.batchSize)
    trainLoader, validLoader = prepareDataset(config)
    print('trainLoader',len(trainLoader))
    print('len_validLoader',len(validLoader))
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    # valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
    # trainLoader = DataLoader(MyDataset(train), shuffle=True, batch_size=config.batchSize, drop_last=False, sampler=train_sampler)
    # validLoader = DataLoader(MyDataset(valid), shuffle=False, batch_size=config.batchSize, drop_last=False, sampler=valid_sampler)

    settings = {
            'remark': args.note,
            'smiVoc': config.smiVoc,
            'proVoc': config.proVoc,
            'smiMaxLen': config.smiMaxLen,
            'proMaxLen': config.proMaxLen,
            'smiPaddingIdx': config.smiVoc.index('^'),
            'proPaddingIdx': config.proVoc.index('^'),
            'smi_voc_len': len(config.smiVoc),
            'pro_voc_len': len(config.proVoc),
            'batchSize': config.batchSize,
            'epoch': epoch,
            'lr': lr,
            'd_model': 96,
            'dim_feedforward': 256,
            'num_layers': args.layers, 
            'nhead': 4,
        }
    logger.info(settings)
    # 写入本次训练配置
    with open((exp_folder + 'settings.json'), 'w') as f:
        json.dump(settings, f)

    model = DrugTransformer(**settings)
    # model = torch.nn.DataParallel(model, device_ids=device_ids) # 指定要用到的设备
    model = model.to(device) # 模型加载到设备0
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True, broadcast_buffers=False)
  
    if len(args.pretrain):
        model.load_state_dict(torch.load(args.pretrain))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #optimizer = torch_npu.optim.NpuFusedAdam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.99, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,5,eta_min=0,last_epoch=-1)
    
    # optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
    # scheduler = nn.DataParallel(scheduler, device_ids=device_ids)


    propertys = ['accuracy', 'loss', ]
    prefixs = ['training', 'validation']
    columns = [' '.join([pre, pro]) for pre in prefixs for pro in propertys]
    logdf = pd.DataFrame({}, columns=columns)

    for i in range(epoch):
        logger.info('EPOCH: {} 训练'.format(i))
        d1 = train(model, trainLoader, config.smiVoc, config.proVoc, device)
        
        logger.info('EPOCH: {} 验证'.format(i))
        d2  = valid(model, validLoader, config.smiVoc, config.proVoc, device)
        
        logdf = logdf.append(pd.DataFrame([d1+d2], columns=columns), ignore_index=True)
        trainingVis(logdf, batchSize, lr, vis_folder)
        
        if args.l:
            scheduler.step()

        # if i % 10 == 0:
        if args.local_rank == 0:
            torch.save(model.state_dict(), model_folder + '{}.pt'.format(i))
        # for name,parameters in model.named_parameters():
        #     print(name,':',parameters.size(), parameters)
        logdf.to_csv(exp_folder+'logs/logdata')

    endTime = time.time()
    logger.info('time: {} h'.format((endTime - startTime) / 3600))
