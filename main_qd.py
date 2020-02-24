import os
import numpy as np 
import torch

from name_generator import create_name_file
from image_loader import image_loader_from_file

from pre_train import train_network
from feature_generator import feature_gen
from networks import encoder, decoder, adv_classifier

from z_encoder_train import train_z_encoder
from style import train_s_encoder
from fusion import train_triplet, fusion_validation

from utils import pickle_load, cuda
import params
# import load_data


from generate_glove_vector import generate_glove_vector


if __name__ == "__main__":

    ## Generate a file containing image/sketch location and corresponding label
    ## Each line of the file contain image/sketch's absolute path and space separated label 
    ## It is divided into train, val and test unit.
    if( not os.path.isfile(params.path_image_file_list) ):
        create_name_file( dataset_path = params.path_image_dataset,
                          path_class_list = params.path_class_list,
                          split = [0.8,0.05, 0.15],
                          dump_location = params.path_image_file_list )
    
    if( not os.path.isfile(params.path_sketchy_file_list) ):
        create_name_file( dataset_path = params.path_sketchy_dataset,
                          path_class_list = params.path_class_list,
                          split = [0.8,0.05, 0.15],
                          dump_location = params.path_sketchy_file_list )
    
    if( not os.path.isfile(params.path_quickdraw_file_list) ):
        create_name_file( dataset_path = params.path_quickdraw_dataset,
                          path_class_list = params.path_class_list,
                          split = [0.5,0.02, 0.48],
                          dump_location = params.path_quickdraw_file_list )

    ## Generate a image/sketch loader from the file generated above.
    loader_image = image_loader_from_file( file_name = params.path_image_file_list )
    loader_sketchy = image_loader_from_file( file_name = params.path_sketchy_file_list )
    loader_quick_draw = image_loader_from_file( file_name = params.path_quickdraw_file_list )
 
    ## Load the saved classification model for image or sketch. If saved model is not found,
    ## train the corresponding model and save it to the location.
    if( not os.path.isfile(params.path_model_image) ):
        print('image classification model not found. Training started.')
        model_image = train_network( data_loader = loader_image,
                                     dump_location = params.path_model_image )
    else:
        image_model = torch.load( params.path_model_image, map_location = torch.device(params.gpu_name) )
        cuda(image_model)
        print('image classification model found. Loading completed')


    if( not os.path.isfile(params.path_model_sketchy) ):
        print('sketchy classification model not found. Training started.')
        model_sketchy = train_network( data_loader = loader_quick_draw,
                       dump_location = params.path_model_sketchy )
    else:
        model_sketchy = torch.load( params.path_model_sketchy, map_location = torch.device(params.gpu_name) )
        cuda(model_sketchy)
        print('sketchy classification model found. Loading completed')


    

    ## Load the saved features[last layer output of classification model] and label for 
    ## image/sketchy/quick_draw. If files are not found generate using the model trained above.
    if( not os.path.isfile( params.path_image_features ) ):
        print('image features file not found. creating it now')
        features_image_dict = feature_gen( model = image_model, loader = loader_image,
                                      dump_location = params.path_image_features ) 
    
    else:
        features_image_dict = pickle_load( params.path_image_features )
        print('image features file found. Loading completed')

    
    if( not os.path.isfile( params.path_sketchy_features ) ):
        print('sketchy features file not found. creating it now')
        features_sketchy_dict = feature_gen( model = model_sketchy, loader = loader_sketchy,
                                      dump_location = params.path_sketchy_features ) 
    
    else:
        features_sketchy_dict = pickle_load( params.path_sketchy_features )
        print('sketchy features file found. Loading completed')

    if( not os.path.isfile( params.path_quickdraw_features ) ):
        print('quickdraw features file not found. creating it now')
        features_quickdraw_dict = feature_gen( model = model_sketchy, loader = loader_quick_draw,
                                      dump_location = params.path_quickdraw_features ) 
    
    else:
        features_quickdraw_dict = pickle_load( params.path_quickdraw_features )
        print('quickdraw features file found. Loading completed')
    

    ## Generate a file containing glove vector corresponding to the classes being used.
    if(os.path.isfile(params.path_glove_vector) == False):
        generate_glove_vector()

    ## Load the z_encoder for both image and sketch. If saved model are not found, train and saved the model.
    if( not os.path.isfile( params.path_z_encoder_image ) ):
        z_encoder_image = encoder( in_dim = params.x_dim, z_dim = params.glove_dim )
        cuda(z_encoder_image)
        z_encoder_image = train_z_encoder( encoder_model = z_encoder_image,
                                           feature_dict = features_image_dict,
                                           dump_location = params.path_z_encoder_image )

    else:
        z_encoder_image = torch.load( params.path_z_encoder_image, map_location = torch.device(params.gpu_name) )
        cuda(z_encoder_image)

    if( not os.path.isfile( params.path_z_encoder_sketchy ) ):
        z_encoder_sketchy = encoder( in_dim = params.x_dim, z_dim = params.glove_dim )
        cuda(z_encoder_sketchy)
        z_encoder_sketchy = train_z_encoder( encoder_model = z_encoder_sketchy,
                                           feature_dict = features_quickdraw_dict,
                                           dump_location = params.path_z_encoder_sketchy )

    else:
        z_encoder_sketchy = torch.load( params.path_z_encoder_sketchy, map_location = torch.device(params.gpu_name) )
        cuda(z_encoder_sketchy)

    
    if( not os.path.isfile( params.path_s_encoder_sketchy ) ):
        s_encoder_sketchy = encoder( in_dim = params.x_dim, z_dim = params.glove_dim )
        decoder_sketchy = decoder(params.glove_dim)
        adv_sketchy = adv_classifier(feat_dim = params.glove_dim, num_classes = params.num_class)
        cuda(s_encoder_sketchy)
        cuda(adv_sketchy)
        cuda(decoder_sketchy)
        # s_encoder_sketchy = train_s_encoder( z_encoder =  z_encoder_sketchy,
        #                                      s_encoder =  s_encoder_sketchy,
        #                                      decoder = decoder_sketchy,
        #                                      adv_classifier =  adv_sketchy,
        #                                      feature_dict =  features_sketchy_dict,
        #                                       dump_location = params.path_s_encoder_sketchy )

    else:
        s_encoder_sketchy = torch.load( params.path_s_encoder_sketchy, map_location = torch.device(params.gpu_name) )
        cuda(s_encoder_sketchy)


    if( not os.path.isfile( params.path_s_encoder_image ) ):
        s_encoder_image = encoder( in_dim = params.x_dim, z_dim = params.glove_dim )
        decoder_image = decoder(params.glove_dim)
        adv_image = adv_classifier(feat_dim = params.glove_dim, num_classes = params.num_class)
        cuda(s_encoder_image)
        cuda(adv_image)
        cuda(decoder_image)
        # s_encoder_image = train_s_encoder( z_encoder =  z_encoder_image,
        #                                      s_encoder =  s_encoder_image,
        #                                      decoder = decoder_image,
        #                                      adv_classifier =  adv_image,
        #                                      feature_dict =  features_image_dict,
        #                                       dump_location = params.path_s_encoder_image )

    else:
        s_encoder_image = torch.load( params.path_s_encoder_image, map_location = torch.device(params.gpu_name) )
        cuda(s_encoder_image)

    if( not os.path.isfile( params.path_fusion_model ) ):
        fusion_model = decoder(params.glove_dim)
        cuda(fusion_model)
        # fusion_model = train_triplet(z_encoder_sketch = z_encoder_sketchy,
        #                              s_encoder_sketch = s_encoder_sketchy,
        #                              z_encoder_image = z_encoder_image,
        #                              s_encoder_image = s_encoder_image,
        #                              fusion_network = fusion_model,
        #                              feature_image_dict = features_image_dict,
        #                              feature_sketch_dict = features_sketchy_dict,
        #                              dump_location = params.path_fusion_model)

    else:
        fusion_model = torch.load(params.path_fusion_model, map_location = torch.device(params.gpu_name))
        cuda(fusion_model)


    fusion_validation(query_feature_arr = features_sketchy_dict['val']['feature'],
                      query_label_arr = features_sketchy_dict['val']['label'],
                      sketch_z_encoder = z_encoder_sketchy,
                      image_z_encoder = z_encoder_image,
                      image_s_encoder = s_encoder_image,
                      fusion_network = fusion_model,
                      image_feature_dataset = features_image_dict['val']['feature'],
                      image_label_dataset = features_image_dict['val']['label'])
    

    fusion_validation(query_feature_arr = features_quickdraw_dict['val']['feature'],
                      query_label_arr = features_quickdraw_dict['val']['label'],
                      sketch_z_encoder = z_encoder_sketchy,
                      image_z_encoder = z_encoder_image,
                      image_s_encoder = s_encoder_image,
                      fusion_network = fusion_model,
                      image_feature_dataset = features_image_dict['val']['feature'],
                      image_label_dataset = features_image_dict['val']['label'])
    