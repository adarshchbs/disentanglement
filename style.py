import numpy as np 
import random

import torch
from torch import nn

from networks import encoder, decoder, adv_classifier, encoder_without_dropout
import params
from utils import chunks
import load_data

style_encoder = encoder_without_dropout(in_dim = 2048, z_dim = params.glove_dim)
decoder = decoder(params.glove_dim)
adv_classifier = adv_classifier(feat_dim = params.glove_dim, num_classes = 87)
z_encoder = torch.load(params.path_sketch_z_encoder)

style_encoder.cuda(params.gpu_name)
decoder.cuda(params.gpu_name)
adv_classifier.cuda(params.gpu_name)
z_encoder.cuda(params.gpu_name)



def train_s_encoder(z_encoder, s_encoder, decoder, adv_classifier, train_x, train_y):

    z_encoder.eval()
    optimizer = torch.optim.Adam( list(s_encoder.parameters())+
                                list(decoder.parameters())+
                                list(adv_classifier.parameters()),
                                lr = 0.0001, betas=[0.8,0.99], weight_decay=0.01 )

    criterion = nn.CrossEntropyLoss()

    for epoch in range( params.num_epochs_style ):
        s_encoder.train()
        decoder.train()
        adv_classifier.train()

        index = np.arange(len(train_y))
        random.shuffle(index)
        train_x = train_x[index]
        train_y = train_y[index]
        total_correct = 0
        total_count = 0
        for step, (features, labels) in enumerate( zip(chunks(train_x), chunks(train_y)) ):

            features = torch.tensor(features)
            labels = torch.tensor(labels,dtype=torch.long)
        
            if(params.gpu_flag == True):
                labels = labels.cuda(params.gpu_name)
                features = features.cuda(params.gpu_name)

            optimizer.zero_grad()
            
            z_vector = z_encoder(features)
            s_vector = s_encoder(features)
            r_vector = decoder(z_vector,s_vector)

            pred_labels = adv_classifier(s_vector)

            _, predicted = torch.max(pred_labels.data,1)
            total_count += labels.size(0)
            total_correct += (predicted == labels).sum().item()

            r_loss = torch.sum( (features - r_vector)**2)/labels.shape[0]/100
            adv_loss = criterion(pred_labels,labels)

            total_loss = r_loss + 0.5 * adv_loss

            total_loss.backward()

            optimizer.step()

            # total_correct += correct
            # total_count += count
            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: r_loss={:.4f} : adv_loss={:.4f} : loss={:.4f}"
                    .format(epoch + 1,
                            params.num_epochs_style,
                            step + 1,
                            int(len(train_x)/params.batch_size),
                            r_loss.data.item(),
                            adv_loss.data.item(),
                            total_loss.data.item()))

        print('accuracy after {} epoch is {}'.format(epoch+1,total_correct/total_count))
        validation_loss(z_encoder,s_encoder,decoder)
        # eval model on test set
        
        # if(epoch %10 == 9):
            # validation(sketch_encoder, x_val, y_val)

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
    # torch.save(sketch_encoder, "sketch_encoder.pt")

    return s_encoder, decoder, adv_classifier


def validation_loss(z_encoder, s_encoder, decoder):
    r_loss = 0
    count = 0
    x_val = load_data.sketch_x_val
    y_val = load_data.sketch_y_val
    z_encoder.eval()
    s_encoder.eval()
    decoder.eval()
    for step, (features, labels) in enumerate( zip(chunks(x_val), chunks(y_val)) ):

        features = torch.tensor(features)
        labels = torch.tensor(labels,dtype=torch.long)
    
        if(params.gpu_flag == True):
            labels = labels.cuda(params.gpu_name)
            features = features.cuda(params.gpu_name)

        
        z_vector = z_encoder(features)
        s_vector = s_encoder(features)
        r_vector = decoder(z_vector,s_vector)

        r_loss = torch.sum( (features - r_vector)**2)/100
        count += labels.shape[0]

    print("validation loss {:.5f}".format(r_loss/count))

a,b,c = train_s_encoder(z_encoder,style_encoder, decoder, adv_classifier, load_data.sketch_x_train, load_data.sketch_y_train)

