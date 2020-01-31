import torch
import numpy as np 
import random   
import pickle
from torch import nn

from loss import distance_loss 
from networks import encoder
import params
from utils import chunks



sketch_encoder = encoder(in_dim=2048,z_dim = params.glove_dim)
sketch_encoder.cuda(params.gpu_name)

sketch_x_train = np.load('/home/adarsh/project/disentanglement/saved_features/da_sketchy_feature_train.npy',allow_pickle=True)
sketch_y_train = np.load('/home/adarsh/project/disentanglement/saved_features/da_sketchy_label_train.npy',allow_pickle=True)

sketch_x_val = np.load('/home/adarsh/project/disentanglement/saved_features/da_sketchy_feature_val.npy',allow_pickle=True)
sketch_y_val = np.load('/home/adarsh/project/disentanglement/saved_features/da_sketchy_label_val.npy',allow_pickle=True)

sketch_y_train = np.array(list(map(int,sketch_y_train)))

def train(sketch_x_train,sketch_y_train, x_val, y_val):
    optimizer = torch.optim.Adam( sketch_encoder.parameters(), lr = 0.00002, betas=[0.8,0.99], weight_decay=0.05 )

    for epoch in range( params.num_epochs_pretrain ):
        sketch_encoder.train()

        index = np.arange(len(sketch_y_train))
        random.shuffle(index)
        sketch_x_train = sketch_x_train[index]
        sketch_y_train = sketch_y_train[index]
        total_correct = 0
        total_count = 0
        for step, (features, labels) in enumerate( zip(chunks(sketch_x_train), chunks(sketch_y_train)) ):

            features = torch.tensor(features)
            labels = torch.tensor(labels,dtype=torch.long)
        
            if(params.gpu_flag == True):
                labels = labels.cuda(params.gpu_name)
                features = features.cuda(params.gpu_name)

            optimizer.zero_grad()
            
            preds = sketch_encoder(features)
            loss, correct, count = distance_loss( preds, labels, .5 )

            loss.backward()
            optimizer.step()

            total_correct += correct
            total_count += count
            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                    .format(epoch + 1,
                            params.num_epochs_pretrain,
                            step + 1,
                            int(len(sketch_x_train)/params.batch_size),
                            loss.data.item()))

        print('accuracy after {} epoch is {}'.format(epoch,total_correct/total_count))
        # eval model on test set
        
        if(epoch %10 == 9):
            validation(sketch_encoder, x_val, y_val)

        # # save model parameters
        # if ((epoch + 1) % params.save_step_pre == 0):
        #     save_model(source_encoder, "ADDA-source-encoder-{}.pt".format(epoch + 1))
        #     save_model(
        #         source_classifier, "ADDA-source-classifier-{}.pt".format(epoch + 1))
        # if ((epoch + 1) % params.eval_step_pre == 0):
        #     eval_src( source_encoder,source_classifier,data_loader,
        #                 split_type='val',gpu_flag=gpu_flag, gpu_name=gpu_name)
        #     eval_tgt( source_encoder,source_classifier,target_data_loader,
        #                 split_type='val',gpu_flag=gpu_flag, gpu_name=gpu_name)




    # # save final model
    torch.save(sketch_encoder, "sketch_encoder.pt")

    return sketch_encoder


def validation(model, x_val, y_val):
    model.eval()
    with open('glove_vector','rb') as f:
        d = pickle.load(f)

    glove_vector = d['glove_vector']
    batch_glove_vector = d['batch_glove_vector']

    if(params.gpu_flag == True):
        glove_vector = glove_vector.cuda(params.gpu_name)
        batch_glove_vector = batch_glove_vector.cuda(params.gpu_name)

    correct = 0
    total = 0  
    for features, labels in zip(chunks(x_val),chunks(y_val)):
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        if(params.gpu_flag == True):
            features = features.cuda(params.gpu_name)
            labels = labels.cuda(params.gpu_name)

        preds = model(features)
        preds = preds.reshape((preds.shape[0],-1,params.glove_dim))

        distance = torch.sum((batch_glove_vector[:preds.shape[0]] - preds)**2,dim=2)

        pred_label = torch.argmin(distance, dim=1)
        correct += int(torch.sum((pred_label == labels)*1))
        # print("correct", correct)
        total += labels.shape[0]

    print("accuracy : ", correct/total)
    print("================================================")



sketch_encoder = train(sketch_x_train,sketch_y_train, sketch_x_val, sketch_y_val)
# sketch_encoder = torch.load('sketch_encoder.pt')

quick_draw_x_val = np.load('/home/adarsh/project/disentanglement/saved_features/da_quick_draw_feature_val.npy',allow_pickle=True)
quick_draw_y_val = np.load('/home/adarsh/project/disentanglement/saved_features/da_quick_draw_label_val.npy',allow_pickle=True)

validation(sketch_encoder,quick_draw_x_val, quick_draw_y_val)