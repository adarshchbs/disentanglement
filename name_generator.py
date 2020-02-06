import os

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import random
import pickle
import time

class image_loader():
    
    def __init__(self, parent_folder_path, folder_list = None,split = [0.8,0.1,0.1] ):

        self.parent_folder_path = parent_folder_path

        if folder_list is not None:
            self.folder_path = folder_list
        else:
            self.folder_path = sorted(os.listdir(self.parent_folder_path))

        self.folder_path = [ self.parent_folder_path + name +'/' for name in self.folder_path ]
        self.image_paths, self.image_labels = self.image_path()

        self.hot_vector_dim = len(set(self.image_labels))
        self.size_total = len(self.image_labels)

        index = np.arange(self.size_total)
        random.shuffle(index)


        self.size = {
            'train' : int(self.size_total*split[0]),
            'val' : int(self.size_total*split[1]),
            'test' : int(self.size_total*split[2])
        }

        size_train_val = self.size['train']+self.size['val']

        self.dir_image_paths = {}
        self.dir_image_paths['train'] = [self.image_paths[i] for i in index[0:self.size['train'] ] ]
        self.dir_image_paths['val'] = [self.image_paths[i] for i in index[self.size['train']:size_train_val] ]
        self.dir_image_paths['test'] = [self.image_paths[i] for i in index[size_train_val:] ]

        self.dir_image_labels = {}
        self.dir_image_labels['train'] = [self.image_labels[i] for i in index[0:self.size['train']] ]
        self.dir_image_labels['val'] = [self.image_labels[i] for i in index[self.size['train']:size_train_val] ]
        self.dir_image_labels['test'] = [self.image_labels[i] for i in index[size_train_val:] ]

    def image_path(self):

        image_paths = []
        image_labels = []
        for counter,path in enumerate(self.folder_path):
            images_list = os.listdir(path)
            for num, image in enumerate(images_list):
                    image_paths.append( path + image )
                    image_labels.append(counter)
        return image_paths, image_labels




def create_name_file(dataset_path, path_class_list, split, dump_location):
    
    class_list = np.loadtxt(path_class_list,dtype='str')
    sketch = image_loader(dataset_path, class_list, split)

    train = []
    for p,l in zip(sketch.dir_image_paths['train'], sketch.dir_image_labels['train']):
        train.append("{} {}".format(p,l))


    val = []
    for p,l in zip(sketch.dir_image_paths['val'], sketch.dir_image_labels['val']):
        val.append("{} {}".format(p,l))

    test = []
    for p,l in zip(sketch.dir_image_paths['test'], sketch.dir_image_labels['test']):
        test.append("{} {}".format(p,l))
    
    train, val, test = np.array(train), np.array(val), np.array(test)

    name_file = {'train': train,
                  'val' : val,
                  'test' : test }

    # np.savetxt('hello.txt',train,fmt='%s')    
    with open(dump_location,'wb') as f:
        pickle.dump(name_file, f )
    

    print('name file saved save at location ' + dump_location )


# parent_path = "/home/iacv/project/adda_sketch/dataset/sketches/"
# path_class_list = "/home/iacv/cluster_backup/project/CDAN/pytorch/common_class_list.txt"

# create_name_file(parent_path,path_class_list,[0.1,0.1,0.1],'v.p')