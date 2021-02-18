from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import nrrd
from utils.createslices import SliceBuilder
import random


class PatchDataset(Dataset):
    def __init__(self, originalroot, gtroot, patch_shape, stride_shape):
        self.originalroot = originalroot
        self.gtroot = gtroot
        self.patch_shape = patch_shape
        self.stide_shape = stride_shape
        self.num_samples = len([lists for lists in os.listdir(originalroot) if os.path.isfile(os.path.join(originalroot, lists))])
        #self.num_samples = 10
    def __getitem__(self, index):
        #image
        imagepath = self.originalroot + str(index).zfill(3)+'.nrrd'
        img,_ = nrrd.read(imagepath)
        #groundtruth
        gtpath = self.gtroot+str(index).zfill(3)+'.nrrd'
        gt,_ = nrrd.read(gtpath)
        #slices
        slices=SliceBuilder(img,self.patch_shape,self.stide_shape)
        slice=slices[random.randint(0,len(slices)-1)]
        while np.sum(gt[slice]) == 0:
            slice = slices[random.randint(0, len(slices) - 1)]

        imgslice=img[slice]
        gtslice=gt[slice]
        #totensor
        image = torch.from_numpy(imgslice.astype(np.int32))
        groundtruth = torch.from_numpy(gtslice.astype(np.int64))
        image = torch.unsqueeze(image,0)
        groundtruth = torch.unsqueeze(groundtruth, 0)
        return image, groundtruth

    def __len__(self):
        return self.num_samples