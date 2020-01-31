import torch
from torch import nn
import numpy as np 
from image_loader import image_loader_from_file
from preprocess import preprocess_image_1, preprocess_image
import network


gpu_name = 'cuda:1'
gpu_flag = True

path_model = '/home/adarsh/project/disentanglement/resnet_50_da.pt'

model = torch.load(path_model)
# model = nn.Sequential(model)
model.cuda(gpu_name)
model.eval()
print(model)

grey = False
path_quick_draw = '/home/adarsh/project/CDAN/pytorch/dataset/sketches/'

path_class_list = '/home/adarsh/project/CDAN/pytorch/common_class_list.txt'

class_list = np.loadtxt(path_class_list,dtype='str')

file_name = 'da_sketchy_'
file_quickdraw_train = '/home/adarsh/project/CDAN/pytorch/sketchy_train.txt'
file_quickdraw_val = '/home/adarsh/project/CDAN/pytorch/sketchy_val.txt'

quick_draw = image_loader_from_file(file_quickdraw_train,file_quickdraw_val)

feature_array_train = []
label_array_train = []

feature_array_val = []
label_array_val = []

for (images, labels) in quick_draw.image_gen(split_type='val'):

    if(grey == True):
        images = preprocess_image_1( array = images,
                                    split_type = 'val',
                                    use_gpu = True, gpu_name= gpu_name  )
    else:
        images = preprocess_image( array = images,
                                    split_type = 'val',
                                    use_gpu = True, gpu_name= gpu_name  )
    # labels = torch.tensor(labels,dtype=torch.long)

    # if(gpu_flag == True):
    #     labels = labels.cuda(gpu_name)

    feature, preds = model(images)
    feature = feature.cpu().detach().numpy()
    
    for f,l in zip(feature, labels):
        feature_array_val.append(f)
        label_array_val.append(l)


np.save('./saved_features/'+file_name+'feature_val.npy',np.array(feature_array_val))
np.save('./saved_features/'+file_name+'label_val.npy',np.array(label_array_val))
print('validation features prepared')


for _, (images, labels) in zip(range(4*quick_draw.size['train']), quick_draw.image_gen(split_type='train')):

    if(grey==True):
        images = preprocess_image_1( array = images,
                                    split_type = 'train',
                                    use_gpu = True, gpu_name= gpu_name  )
    else:
        images = preprocess_image( array = images,
                                    split_type = 'train',
                                    use_gpu = True, gpu_name= gpu_name  )        

    # labels = torch.tensor(labels,dtype=torch.long)

    # if(gpu_flag == True):
    #     labels = labels.cuda(gpu_name)

    feature, preds = model(images)
    feature = feature.cpu().detach().numpy()
    
    for f,l in zip(feature, labels):
        feature_array_train.append(f)
        label_array_train.append(l)


np.save('./saved_features/'+file_name+'feature_train.npy',np.array(feature_array_train))
np.save('./saved_features/'+file_name+'label_train.npy',np.array(label_array_train))
print('train features prepared')

