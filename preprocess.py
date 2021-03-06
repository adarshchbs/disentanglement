from torchvision import transforms
import torch 
import time

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_transforms_1 = {
    'train': transforms.Compose([
        transforms.Grayscale(3),
        transforms.Scale(256),
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(3),
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def preprocess_image(array, split_type, use_gpu = True, gpu_name = 'cuda:0'):
    if(split_type == 'test'):
        split_type = 'val'
    array_preprocess = []
    # time_counter = 0
    for i in array:
        # start = time.time()
        if(i.mode == 'L'):
            array_preprocess.append( data_transforms_1[split_type](i) )
        else:
            array_preprocess.append( data_transforms[split_type](i) )
        # end = time.time()
        # time_counter += (end-start)
    # print(time_counter)
    if( use_gpu == True ):
        array_preprocess = torch.stack(array_preprocess).cuda(gpu_name)
    else:
        array_preprocess = torch.stack(array_preprocess)
    return array_preprocess

