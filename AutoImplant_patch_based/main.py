import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from trainer.trainer import validate, train, test
from models import vnet
from datasets.patchdataset import PatchDataset
from criterions.criterions import criterion_selector
import numpy as np
#import setproctitle
import datetime
import os


# load the config
with open('./configs/config.json',"r") as conf:
    config = json.load(conf)
# cuda for accelerate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# change process name for easier detection in tools
#setproctitle.setproctitle(config['experiment_name'])

#start!
print("build vnet")
model = vnet.VNet(elu=False, nll=True)
model = nn.parallel.DataParallel(model)

#parameter calculation
total = sum([param.nelement() for param in model.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))

# #datasetSplit
# original_root=config["Data"]["originaldataset_root"]
# gt_root=config["Data"]["gtdataset_root"]
# test_root=config["Data"]["testdataset_root"]
# result_root=config["Data"]["resultimage_root"]
# datasize =config["Data"]["InputImageSize"]
# patch_shape=config["Slicer"]["patch_shape"]
# stride_shape=config["Slicer"]["stride_shape"]
# dataset = PatchDataset(originalroot=original_root,gtroot=gt_root,patch_shape=patch_shape,stride_shape=stride_shape)
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(config["Data"]["validation_split"] * dataset_size))
# train_indices, val_indices = indices[split:], indices[:split]
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)
# #dataLoader
# trainloader = DataLoader(dataset, batch_size=config["Data"]["batch_size"], sampler= train_sampler, shuffle=False)
# validloader = DataLoader(dataset, batch_size=config["Data"]["batch_size"], sampler= valid_sampler, shuffle=False)
# #optimizer(coded to be changable in config)
# optimizer = optim.Adam(model.parameters(), lr=config["Optimizer"]["lr"], weight_decay = 0.0001)
# #cretirion
# criterion = criterion_selector(config["Criterion"]["name"])
# #load model
# model.load_state_dict(torch.load("../autoimplantdata/results/2020_09_12_10_51/model/model.pth"))
# model.eval()
# print("load model")
# #train
# for epoch in range(config["Training"]["num_epochs"]):
#     train(epoch=epoch, model=model, trainLoader=trainloader, optimizer=optimizer, device=device, criterion=criterion)
#     error=validate(model=model, valLoader=validloader, device=device, criterion=criterion)
#     if error<0.12:
#       break
#
# #create results directory
# theTime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
# result_root=config["Data"]["resultimage_root"]+'{}'.format(theTime)
# os.makedirs(result_root)
# # with open(result_root+'/config.txt', 'a') as outfile:
# #   json.dump(config,outfile)
# #   outfile.write('\nEPOCH '+str(epoch))
# #   outfile.write('\nValidation set: Average loss: {:.4f}, Error: {}/{} ({:.3f}%， Falsenegative: {}, Falsepositive: {})\n'.format(
# #         testloss, incorrect, numel, error, falsenega , falseposi))
# #save
# os.makedirs(result_root+"/model/")
# torch.save(model.state_dict(), result_root+"/model/model.pth")
# print("save model")
#
# test(model=model, testroot=test_root, resultroot= result_root, device=device, patch_shape=patch_shape,stride_shape=stride_shape)
# print('\n\nShakespeare sagte: Ende gut, Alles gut. Aber manchmal ist etwas einfach nur zu Ende.\n\n')