import numpy as np 
import random

import torch
from torch import nn
from collections import defaultdict

import params
import load_data
from networks import encoder, decoder, adv_classifier, encoder_without_dropout
from utils import chunks, last_k

def retrieve_images(sketch_query, query_label, sketch_z_encoder, image_s_enocoder, fusion_network,
                    image_feature_dataset, image_label_dataset):

    sketch_query = torch.tensor(sketch_query).reshape((-1,sketch_query.shape[0]))
    if(params.gpu_flag):
        sketch_query.cuda(params.gpu_name)

    query_z_vector = sketch_z_encoder( sketch_query )

    distance = torch.sum((image_feature_dataset - query_z_vector)**2,dim=1)
    sorted_features, sorted_arg = torch.sort(distance)

    K_closest_features = sorted_features[0:params.num_query]
    k_closest_arg = sorted_arg[0:params.num_query]

    query_z_vector = torch.cat([query_z_vector]*params.num_query)

    fake_features = fusion_network( query_z_vector, K_closest_features)
    
    normalized_distance = torch.sum((K_closest_features - fake_features)**2,dim = 1)
    _, normalized_sorted_arg = torch.sort(normalized_distance)

    retrived_arg = k_closest_arg[normalized_sorted_arg]
    predicted_label = image_label_dataset[retrived_arg]

    weight = 1/torch.tensor(np.arange(1,params.num_query+1,dtype=np.float32))
    score = torch.sum(weight * (predicted_label == query_label))/torch.sum(weight)


    return predicted_label, score