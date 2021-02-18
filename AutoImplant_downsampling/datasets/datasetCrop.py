from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import SimpleITK as sitk
import numpy as np
import os
import random


def ImageResize(sitk_image, datasize):#imageresize
    dimension = sitk_image.GetDimension()
    reference_physical_size = np.zeros(dimension)
    reference_physical_size[:] = [(sz - 1) * spc if sz * spc > mx else mx for sz, spc, mx in
                                  zip(sitk_image.GetSize(), sitk_image.GetSpacing(), reference_physical_size)]

    reference_origin= np.zeros(dimension)
    reference_direction = np.identity(dimension).flatten()
    reference_size = datasize  # Arbitrary sizes, smallest size that yields desired results.
    reference_spacing = [phys_sz / (sz - 1) for sz, phys_sz in zip(reference_size, reference_physical_size)]

    reference_image = sitk.Image(reference_size, sitk_image.GetPixelIDValue())
    reference_image.SetOrigin(reference_origin)
    reference_image.SetSpacing(reference_spacing)
    reference_image.SetDirection(reference_direction)

    reference_center = np.array(
        reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize()) / 2.0))

    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(sitk_image.GetDirection())
    transform.SetTranslation(np.array(sitk_image.GetOrigin()) - reference_origin)

    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(sitk_image.TransformContinuousIndexToPhysicalPoint(np.array(sitk_image.GetSize()) / 2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    nimage = sitk.Resample(sitk_image, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    return nimage


def ImageCrop(imgarray, gtarray, datasize):#arraycrop
    lh = len(imgarray[0])
    lw = len(imgarray[0][0])
    lc = len(imgarray)
    height = datasize[0]
    width = datasize[1]
    channel = datasize[2]
    #randomcrop
    '''
    randc = random.randint(0, lc - channel)    
    randh = random.randint(0, lh-height)
    randw = random.randint(0, lw-width)
    imgarray_slice = imgarray[randc:randc + channel, randh:randh + height, randw:randw + width]
    gtarray_slice = gtarray[randc:randc + channel, randh:randh + height, randw:randw + width]
    '''
    # centercrop
    centerc = lc//2 - channel//2
    centerh = lh//2 - height//2
    centerw = lw//2 - width//2
    imgarray_slice = imgarray[centerc:centerc+channel, centerh:centerh+height, centerw:centerw+width]
    gtarray_slice = gtarray[centerc:centerc+channel, centerh:centerh+height, centerw:centerw+width]

    return imgarray_slice,gtarray_slice


def ImageNomalize(image):
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(255)
    resacleFilter.SetOutputMinimum(0)
    image = resacleFilter.Execute(image)
    return image


class AortaDataset(Dataset):
    def __init__(self, root, datasize):
        self.root = root
        self.num_samples = len([lists for lists in os.listdir(root) if os.path.isdir(os.path.join(root, lists))])
        self.datasize = datasize
    def __getitem__(self, index):
        #image
        imagepath = self.root+ 'Patient_'+str(index+1).zfill(2)+'/Patient_'+str(index+1).zfill(2)+'.nii.gz'
        img = sitk.ReadImage(imagepath)
        #groundtruth
        gtpath = self.root+'Patient_'+str(index+1).zfill(2)+'/GT.nii.gz'
        gt = sitk.ReadImage(gtpath)

        #nomalize
        img = ImageNomalize(img)

        #resize to datasize/2
        imgsize = np.array(img.GetSize())
        imgsize = (imgsize//2).tolist()
        img = ImageResize(img, imgsize)
        gt = ImageResize(gt, imgsize)

        #tonumpy
        imgarray = sitk.GetArrayFromImage(img)
        gtarray = sitk.GetArrayFromImage(gt)

        #resize
        imgarray,gtarray = ImageCrop(imgarray,gtarray,self.datasize)

        #remove other masks
        gtarray = gtarray//4
        imgarray = imgarray/255
        #totensor
        image = torch.from_numpy(imgarray.astype(np.int32))
        groundtruth = torch.from_numpy(gtarray.astype(np.int64))
        image = torch.unsqueeze(image,0)
        groundtruth = torch.unsqueeze(groundtruth, 0)
        return image, groundtruth

    def __len__(self):
        return self.num_samples