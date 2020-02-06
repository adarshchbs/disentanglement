import torch
import numpy as np
import re
import pickle
import params

batch_size = params.batch_size

with open('glove_vector','rb') as f:
    d = pickle.load(f)

glove_vector = d['glove_vector']
batch_glove_vector = d['batch_glove_vector']

if(params.gpu_flag == True):
    glove_vector = glove_vector.cuda(params.gpu_name)
    batch_glove_vector = batch_glove_vector.cuda(params.gpu_name)


def distance_loss(output, lables, alpha):
    compare_vector = glove_vector[lables].reshape(lables.shape[0],-1,params.glove_dim)
    output = output.reshape(lables.shape[0],-1,params.glove_dim)
    
    factor = -1 * alpha * torch.sum( (compare_vector - output)**2,dim=[2,1] )
    num = torch.exp( factor )
    
    temp_batch_glove_vector = batch_glove_vector[:lables.shape[0]]
    
    dem_factor = torch.exp( -1 * alpha * torch.sum((temp_batch_glove_vector-output)**2, dim =2) )
    pred_labels = torch.argmax(dem_factor,dim=1)
    dem = torch.sum(dem_factor, dim =1)
    
    prob = num/dem
    loss = torch.sum(-torch.log(prob))/lables.shape[0]
    correct = int(torch.sum( (pred_labels == lables)*1))
    
    return loss, correct, lables.shape[0]




