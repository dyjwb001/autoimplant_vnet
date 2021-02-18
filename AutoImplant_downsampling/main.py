import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from trainer.trainer import validate, train, test
from models import vnet
from datasets.datasetCrop import AortaDataset
from criterions.criterions import criterion_selector
import numpy as np
# import setproctitle


# load the config
with open('./configs/config.json',"r") as conf:
    config = json.load(conf)
# cuda for accelerate
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# change process name for easier detection in tools
# setproctitle.setproctitle(config['experiment_name'])

#start!
print("build vnet")
model = vnet.VNet(elu=False, nll=True)
model = nn.parallel.DataParallel(model)

total = sum([param.nelement() for param in model.parameters()])

print("Number of parameter: %.2fM" % (total / 1e6))

# #datasetSplit
# data_root=config["Data"]["traindataset_root"]
# test_root=config["Data"]["testdataset_root"]
# result_root=config["Data"]["resultimage_root"]
# datasize =config["Data"]["InputImageSize"]
# dataset = AortaDataset(root=data_root,datasize=datasize)
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
# optimizer = optim.Adam(model.parameters(), lr=config["Optimizer"]["lr"],weight_decay=1)
# #cretirion
# criterion = criterion_selector(config["Criterion"]["name"])
# #train
# for epoch in range(config["Training"]["num_epochs"]):
#     train(epoch=epoch, model=model, trainLoader=trainloader, optimizer=optimizer, device=device, criterion=criterion)
#     err = validate(epoch=epoch, model=model, valLoader=validloader, optimizer=optimizer, device=device, criterion=criterion)
#     #early stopping
#     if err<0.01:
#         break
# test(model=model, testroot=test_root, resultroot= result_root, device=device, datasize=datasize)
# print('\n\nShakespeare sagte: Ende gut, Alles gut. Aber manchmal ist etwas einfach nur zu Ende.\n\n')

