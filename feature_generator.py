import torch
import numpy as np 
from image_loader import image_loader
from preprocess import preprocess_image_1


gpu_name = 'cuda:1'
gpu_flag = True

path_model = '/home/adarsh/project/CDAN/pytorch/snapshot/san/best_model.pth.tar'

model = torch.load(path_model)
# model = nn.Sequential(model)
model.cuda(gpu_name)
model.eval()

path_quick_draw = '/home/adarsh/project/CDAN/pytorch/dataset/QuickDraw_sketches_final/'

path_class_list = '/home/adarsh/project/CDAN/pytorch/common_class_list.txt'

class_list = np.loadtxt(path_class_list,dtype='str')

quick_draw = image_loader(path_quick_draw,folder_list=class_list, split=[0,1,0])

feature_array = []
label_array = []
for (images, labels) in quick_draw.image_gen(split_type='val'):

    images = preprocess_image_1( array = images,
                                split_type = 'val',
                                use_gpu = True, gpu_name= gpu_name  )

    labels = torch.tensor(labels,dtype=torch.long)

    if(gpu_flag == True):
        labels = labels.cuda(gpu_name)

    feature, preds = model(images)
    feature = feature.cpu().detach().numpy()
    
    for f,l in zip(feature, labels):
        feature_array.append(f)
        label_array.append(l)


np.save('feature_array.npy',np.array(feature_array))
np.save('label_array.npy',np.array(label_array))

