from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import nrrd


class ImplantDataset(Dataset):
    def __init__(self, originalroot, gtroot):
        self.originalroot = originalroot
        self.gtroot = gtroot
        self.num_samples = len([lists for lists in os.listdir(root) if os.path.isfile(os.path.join(root, lists))])
    def __getitem__(self, index):
        #image
        imagepath = self.originalroot + str(index).zfill(3)+'.nrrd'
        img = nrrd.read(imagepath)
        #groundtruth
        gtpath = self.gtroot+str(index).zfill(3)+'.nrrd'
        gt = nrrd.read(gtpath)
        #totensor
        image = torch.from_numpy(imgarray.astype(np.int32))
        groundtruth = torch.from_numpy(gtarray.astype(np.int64))
        image = torch.unsqueeze(image,0)
        groundtruth = torch.unsqueeze(groundtruth, 0)
        return image, groundtruth

    def __len__(self):
        return self.num_samples