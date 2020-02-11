import torch
from torch import nn
import numpy as np 
import pickle

from image_loader import image_loader_from_file
from preprocess import preprocess_image
import network
import params


def feature_gen( model, loader, dump_location ):

    model.eval()

    # loader = image_loader_from_file(image_path_file)

    train = model_output(model = model, data_loader = loader, split_type = 'train', repeat_num = 4)
    val = model_output(model = model, data_loader = loader, split_type = 'val', repeat_num = 1)
    test = model_output(model = model, data_loader = loader, split_type = 'test', repeat_num = 1)
    
    save_dict = {
                  'train' : train,
                  'val' : val,
                  'test' : test
                }
    
    with open(dump_location, 'wb') as f:
        pickle.dump(save_dict,f)

    print("feature generated and save at location " + dump_location)

    return save_dict
    


def model_output(model, data_loader, split_type, repeat_num ):

    feature_array = []
    label_array = []

    for _, (images, labels) in zip(range(repeat_num*data_loader.size[split_type]), data_loader.image_gen(split_type = split_type)):
        images = preprocess_image( array = images,
                                    split_type = split_type,
                                    use_gpu = params.gpu_flag, gpu_name= params.gpu_name  )


        feature, _ = model(images)
        feature = feature.cpu().detach().numpy()
        
        for f,l in zip(feature, labels):
            feature_array.append(f)
            label_array.append(l)

    ret_dict = {'feature': np.array(feature_array), 'label' : np.array(label_array) }

    return ret_dict


        