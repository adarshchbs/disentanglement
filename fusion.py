import numpy as np 
import random

import torch
from torch import nn
from collections import defaultdict

import params
# import load_data
from networks import encoder, decoder, adv_classifier, encoder_without_dropout
from utils import chunks
from retrieval import retrieve_images

style_encoder = encoder_without_dropout(in_dim = 2048, z_dim = params.glove_dim)
z_encoder = torch.load(params.path_sketch_z_encoder)
fusion_network = decoder(params.glove_dim)

style_encoder.cuda(params.gpu_name)
z_encoder.cuda(params.gpu_name)
fusion_network.cuda(params.gpu_name)

def next_el( a, b, index ):
    try:
        return next(b[index])
    except StopIteration:
        random.shuffle(a[index])
        b[index] = iter(a[index])
        return next(b[index])
        
def index_init(labels):
    labels_dict = defaultdict(list)
    iter_labels_dict = {}
    for index, l in enumerate(labels):
        labels_dict[l].append(index)

    for keys in labels_dict:
        iter_labels_dict[keys] = iter(labels_dict[keys])
    
    return labels_dict, iter_labels_dict


def sample_triplet(sketch_array, image_array, sketch_labels_dict, sketch_iter_labels_dict,
                    image_labels_dict, image_iter_labels_dict ):
    sample_class = np.random.choice( params.num_class,(params.batch_size,2) )
    sketch_sample = []
    pos_sample = []
    neg_sample = []
    for pos_class, neg_clas in sample_class:
        sketch_sample.append( next_el( sketch_labels_dict,sketch_iter_labels_dict, pos_class ) )
        pos_sample.append( next_el( image_labels_dict, image_iter_labels_dict, pos_class ) )
        neg_sample.append( next_el( image_labels_dict, image_iter_labels_dict, neg_clas ) )

    return sketch_sample, pos_sample, neg_sample


    





def train_triplet(z_encoder_sketch, s_encoder_sketch, 
                 z_encoder_image, s_encoder_image, fusion_network,
                 sketch_features, sketch_labels, image_features, image_labels ):

    z_encoder_sketch.eval()
    s_encoder_sketch.eval()
    z_encoder_image.eval()
    s_encoder_image.eval()
    
    optimizer = torch.optim.Adam( fusion_network.parameters(),
                                lr = 0.0001, betas=[0.8,0.99], weight_decay=0.01 )

    dict_sketch, iter_dict_sketch = index_init(sketch_labels)
    dict_image, iter_dict_image = index_init(image_labels)


    for epoch in range( params.num_epochs_fusion ):
        fusion_network.train()

        for step, ( sketch_sample, pos_sample, neg_sample) in enumerate( sample_triplet( sketch_features, image_features,
                                                                        dict_sketch, iter_dict_sketch,
                                                                        dict_image, iter_dict_image ) ):

            sketch_sample = torch.tensor(sketch_sample)
            pos_sample = torch.tensor(pos_sample)
            neg_sample = torch.tensor(neg_sample)

            current_batch_size = pos_sample.shape[0]

            if(params.gpu_flag == True):
                sketch_sample = sketch_sample.cuda(params.gpu_name) 
                pos_sample = pos_sample.cuda(params.gpu_name)
                neg_sample = neg_sample.cuda(params.gpu_name)

            optimizer.zero_grad()
            
            z_vector_sketch = z_encoder_sketch(sketch_sample)
            s_vector_pos = s_encoder_sketch(pos_sample)
            s_vector_neg = s_encoder_sketch(neg_sample)

            reconst_postive = fusion_network( z_vector_sketch, s_vector_pos )
            reconst_negative = fusion_network( z_vector_sketch, s_vector_neg )

            loss_pos = torch.sum( ( pos_sample - reconst_postive )**2 )/current_batch_size
            loss_neg = torch.sum( ( neg_sample - reconst_negative )**2 )/current_batch_size

            total_loss = loss_pos - loss_neg + .1

            if(total_loss.data.item() > 0):
                total_loss.backward()
                optimizer.step()            

            # total_correct += correct
            # total_count += count
            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: : loss={:.4f}"
                    .format(epoch + 1,
                            params.num_epochs_style,
                            step + 1,
                            int(params.fusion_iter_len/params.batch_size),
                            total_loss.data.item()))

        
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

    return fusion_network

def fusion_validation( query_feature_arr, query_label_arr, sketch_z_encoder, image_s_enocoder, fusion_network,
                      image_feature_dataset, image_label_dataset):
        total_score = 0
        for query, label in zip(query_feature_arr, query_label_arr):
            _, scores = retrieve_images(query, label, sketch_z_encoder, image_s_enocoder,
                                         fusion_network,image_feature_dataset, image_label_dataset )
            total_score += scores
        print("validation scores :",total_score/len(query_feature_arr))