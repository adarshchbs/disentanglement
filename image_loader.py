import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import random
import re
import time
import pickle
import params

class image_loader_from_file():
    
    def __init__(self, file_name ):


        with open(file_name,'rb') as f:
            name_dict = pickle.load(f)

        self.size = {
            'train' : len(name_dict['train']),
            'val' : len(name_dict['val']),
            'test' : len(name_dict['test'])
        }

        self.dir_image_paths = {}
        self.dir_image_paths['train'] = [ re.split(' ', path_label)[0] for path_label in name_dict['train'] ]
        self.dir_image_paths['val'] = [ re.split(' ', path_label)[0] for path_label in name_dict['val'] ]
        self.dir_image_paths['test'] = [ re.split(' ', path_label)[0] for path_label in name_dict['test'] ]

        self.dir_image_labels = {}
        self.dir_image_labels['train'] = [ int(re.split(' ', path_label)[1]) for path_label in name_dict['train'] ]
        self.dir_image_labels['val'] = [ int(re.split(' ', path_label)[1]) for path_label in name_dict['val'] ]
        self.dir_image_labels['test'] = [ int(re.split(' ', path_label)[1]) for path_label in name_dict['test'] ]



    def image_load(self, index, split_type):
        images, labels = [], []

        for i in index:
            images.append(Image.open(self.dir_image_paths[split_type][i]))
            labels.append(self.dir_image_labels[split_type][i])

        return images, np.array(labels)

    def image_gen( self, split_type = "train", batch_size = params.batch_size ):
        self.batch_size = batch_size
        index = np.arange(len(self.dir_image_labels[split_type]))

        if(split_type == 'train'):
            random.shuffle(index)

        iter_low = iter( range( 0, len(index) , batch_size ) )
        iter_high = iter( range( batch_size, len(index) + len(index)%batch_size , batch_size ) )
        while(True):
            try:
                
                low_lim = next(iter_low)
                high_lim = next(iter_high)

            except:
                return

            yield self.image_load(index[low_lim:high_lim], split_type )
        




# a = image_loader_from_file('v.p')
# for name in a.image_gen():
#     print(name)
#     break