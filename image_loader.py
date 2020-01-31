import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import random
import re
import time

class image_loader_from_file():
    
    def __init__(self, file_train, file_val ):

        # self.parent_folder_path = parent_folder_path

        # if folder_list is not None:
        #     self.folder_path = folder_list
        # else:
        #     self.folder_path = sorted(os.listdir(self.parent_folder_path))

        # self.folder_path = [ self.parent_folder_path + name +'/' for name in self.folder_path ]
        # self.image_paths, self.image_labels = self.image_path()

        train_set = np.loadtxt(file_train, dtype= str)
        val_set = np.loadtxt(file_val, dtype=str)


        # self.hot_vector_dim = len(set(self.image_labels))
        # self.size_total = len(self.image_labels)

        # index = np.arange(self.size_total)
        # random.shuffle(index)


        self.size = {
            'train' : len(train_set),
            'val' : len(val_set)
        }

        # size_train_val = self.size['train']+self.size['val']

        self.dir_image_paths = {}
        self.dir_image_paths['train'] = train_set[:,0]
        self.dir_image_paths['val'] = val_set[:,0]
        # self.dir_image_paths['test'] = [self.image_paths[i] for i in index[size_train_val:] ]

        self.dir_image_labels = {}
        self.dir_image_labels['train'] = [int(i[1]) for i in train_set]
        self.dir_image_labels['val'] = [int(i[1]) for i in val_set]
        # self.dir_image_labels['test'] = [self.image_labels[i] for i in index[size_train_val:] ]


    # def image_path(self):

    #     image_paths = []
    #     image_labels = []
    #     for counter,path in enumerate(self.folder_path):
    #         images_list = os.listdir(path)
    #         for image in images_list:
    #             image_paths.append( path + image )
    #             image_labels.append(counter)
    #     return image_paths, image_labels
    

    def image_load(self, index, split_type):
        images, labels = [], []

        for i in index:
            images.append(Image.open(self.dir_image_paths[split_type][i]))
            # label_hot_vector = np.zeros(self.hot_vector_dim,dtype=np.int32)
            # label_hot_vector[self.dir_image_labels[split_type][i]] = 1
            labels.append(self.dir_image_labels[split_type][i])

        return images, np.array(labels)

    def image_gen( self, split_type = "train", batch_size = 32 ):
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
        







# # 
# # path_class_list = '/home/iacv/project/adda_sketch/common_class_list.txt'
    
# # class_list = np.loadtxt(path_class_list,dtype='str')
 

# sketch = image_loader('/home/iacv/project/sketch/dataset/images/' )
# # print(sketch.folder_path

# path = sketch.dir_image_paths['val']
# labels = sketch.dir_image_labels['val']

# for p,l in zip(path,labels):
#     print("{} {}".format(p,l))
# # count = 0
# # start = time.time()
# # for i in  sketch.image_gen('val',batch_size=128):
# #     a,b = i
# #     count += 128
# #     print(count, time.time()-start)
