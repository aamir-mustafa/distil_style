#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 17:33:20 2021

@author: am2806
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image

def load(path):
    img = Image.open(path)
    return img

to_tensor = transforms.ToTensor()

to_pil = transforms.ToPILImage()
#B=to_pil(real_B[0,:,:,:].detach().cpu())

def torch_to_np(img_var):
    return img_var.detach().cpu().numpy()#[0]

def np_to_torch(img_np):
    return torch.from_numpy(img_np)[None, :]



def compute_Y(img):  # img is a tensor of shape [1,3,x,y]
    
    if img.shape[0]==1:
        img= img[0,:,:,:]
    img_Y= 0.212656*img[0,:,:] + 0.715158*img[1,:,:] + 0.072186*img[2,:,:]
    return img_Y


# sdr_Y_tensor and hdr_Y_tensor
    # sdr_Y and hdr_Y
    
def compute_histogram(img):  # img is a tensor of dim [1,x,y] or [x,y]
    
    img_np= torch_to_np(img)
    histogram, bin_edges = np.histogram(img_np, bins =25, range =(0,1)) 
    return histogram, bin_edges  # it is of dim int64 (50,)
#histogram_sdr, bin_edges_sdr = np.histogram(sdr_Y_tensor, bins =50, range =(0,1))      # image should be of dimension [x,y] or [1,x,y]
#histogram_hdr, bin_edges_hdr = np.histogram(hdr_Y_tensor, bins =50, range =(0,1))  

#Normalize the histogram
def normalize_histogram(hist):
    hist_normalized = (hist - hist.min())/ (hist.max() - hist.min())
    return hist_normalized
    

def concatenate_img_hist(img, histogram):
    img_sq = torch.squeeze(img)   #torch.Size([3, 540, 960])
    img_flatten=  torch.flatten(img_sq, start_dim=1)   #torch.Size([3, 518400])
    img_permuted= img_flatten.T   #torch.Size([518400, 3]) , batch_size would be 518400
    hist_ = torch.zeros((518400,50)).cuda()
    hist_[:] = torch.tensor(histogram)   # torch.float32 
    return torch.cat((img_permuted, hist_), 1)
    


def concatenate_img_feats(img, feats):
    img_sq = torch.squeeze(img)
    img_flatten=  torch.flatten(img_sq, start_dim=1)   #torch.Size([3, 518400])
    img_permuted= img_flatten.T   #torch.Size([518400, 3]) , batch_size would be 518400
    features = torch.zeros((518400,512)).cuda()
    features[:] = feats[0]
    return torch.cat((img_permuted, features),1)
     

def concatenate_features_squeezenet(img, feats):
    batch= img.shape[0]  # gives the batch size
#    img_sq = torch.squeeze(img)
    img_flatten=  torch.flatten(img, start_dim=2)   #torch.Size([batch, 3, 518400])
    images= torch.zeros((518400*batch, 3)).cuda()
    for i in range(batch):
        img_ = img_flatten[i].T   #torch.Size([518400, 3])
        images[518400*i:518400*(i+1)] = img_    # This has to be concatenated with feats
    
#    img_permuted= img_flatten.T   #torch.Size([518400, 3]) , batch_size would be 518400
    features = torch.zeros((518400*batch,100)).cuda()
    
    for ii in range(batch):
        feats_ = feats[ii]
        features[518400*ii:518400*(ii+1)] = feats_
#    features[:] = feats[0]
    return torch.cat((images, features),1)   # torch.Size([batch*518400, 103])



def concatenate_features_squeezenet_part2(img, input_feats, target_feats):
    batch= img.shape[0]  # gives the batch size
#    img_sq = torch.squeeze(img)
    img_flatten=  torch.flatten(img, start_dim=2)   #torch.Size([batch, 3, 518400])
    images= torch.zeros((518400*batch, 3)).cuda()
    for i in range(batch):
        img_ = img_flatten[i].T   #torch.Size([518400, 3])
        images[518400*i:518400*(i+1)] = img_    # This has to be concatenated with feats
    
#    img_permuted= img_flatten.T   #torch.Size([518400, 3]) , batch_size would be 518400
    if  input_feats.shape[1] == target_feats.shape[1]:    
        len_feats= int(input_feats.shape[1] *2)
    features = torch.zeros((518400*batch,len_feats)).cuda()
    
    for ii in range(batch):
        input_feats_ = input_feats[ii]
        target_feats_ = target_feats[ii]
        
        features[518400*ii:518400*(ii+1), 0:int(len_feats/2)] = input_feats_
        features[518400*ii:518400*(ii+1), int(len_feats/2):int(len_feats)] = target_feats_
#    features[:] = feats[0]
    return torch.cat((images, features),1)   # torch.Size([batch*518400, 103])





    
#def concatenate_img_histogram(img, histogram):  # img is tensor [1,3, x,y], histogram is an int64
#    W,H = img.shape[2], img.shape[3]
#    hist_ = np.zeros([1,50,W,H])
#    for i in range(50):
#        hist_[:,i,:,:] = histogram[i]
#    histogram_torch= torch.tensor(hist_).cuda()
#    return torch.cat((img, histogram_torch), 1)
    
    
    

def plot_hist(histogram, bin_edges):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixels")
    plt.xlim([0.0, 1.0])  # <- named arguments do not work here
    
    plt.plot(bin_edges[0:-1], histogram)  # <- or here
    plt.show()
