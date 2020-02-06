import numpy as np 
import torch
from torch import nn 

import params
from network import ResNetFc
from preprocess import preprocess_image_new

def train_network( data_loader, dump_location ):

    model = ResNetFc( resnet_name = 'ResNet50', use_bottleneck = False,
                      bottleneck_dim= 256, new_cls= True, class_num = params.num_class)

    optimizer = torch.optim.Adam(  model.parameters() ,
                            lr = params.pretrain_lr,
                            )
    
 
    
    criterion = nn.CrossEntropyLoss()

    for epoch in range( params.num_epochs_pretrain ):
    
        for step, ( images, labels ) in enumerate( data_loader.image_gen('train') ):

            images = preprocess_image_new( array = images,
                                           split_type = 'train',
                                           use_gpu = params.gpu_flag,
                                           gpu_name = params.gpu_name )

            labels = torch.tensor(labels,dtype=torch.long)

            if(params.gpu_flag == True):
                labels = labels.cuda(params.gpu_name)


            optimizer.zero_grad()
            
            _, preds = model(images) 
            loss = criterion( preds, labels )

            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pretrain,
                              step + 1,
                              int(data_loader.size['train']/data_loader.batch_size),
                              loss.data.item()))
                # print(list(source_classifier.parameters()))
        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src( model, data_loader)

    
    # save model parameters
    torch.save(model,dump_location)
    print( 'model saved at location ' + dump_location )

    return model

def eval_src( model, data_loader ):

    loss = 0
    accuracy = 0 

    model.eval()

    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0

    for (images, labels) in data_loader.image_gen(split_type='val'):

        images = preprocess_image_new( array = images,
                                           split_type = 'train',
                                           use_gpu = params.gpu_flag,
                                           gpu_name = params.gpu_name )

        labels = torch.tensor(labels,dtype=torch.long)

        if( params.gpu_flag == True):
            labels = labels.cuda(params.gpu_name)
        _, preds = model(images)
        loss += criterion( preds, labels ).item()

        _, predicted = torch.max(preds.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    loss /= data_loader.size['val']
    # accuracy /= len( data_loader )
    accuracy = correct/total

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, accuracy))

