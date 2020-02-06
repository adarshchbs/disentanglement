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

from utils import pickle_load
import params
import load_data


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
        model_image = train_network( data_loader = loader_image,
                                     dump_location = params.path_model_image )
    else:
        image_model = torch.load( params.path_model_image )


    if( not os.path.isfile(params.path_model_sketchy) ):
        model_sketchy = train_network( data_loader = loader_sketchy,
                       dump_location = params.path_model_sketchy )
    else:
        model_sketchy = torch.load( params.path_model_sketchy )
    

    ## Load the saved features[last layer output of classification model] and label for 
    ## image/sketchy/quick_draw. If files are not found generate using the model trained above.
    if( not os.path.isfile( params.path_image_features ) ):
        features_image_dict = feature_gen( model = image_model, loader = loader_image,
                                      dump_location = params.path_image_features ) 
    
    else:
        features_image_dict = pickle_load( params.path_image_features )

    
    if( not os.path.isfile( params.path_sketchy_features ) ):
        features_sketchy_dict = feature_gen( model = model_sketchy, loader = loader_sketchy,
                                      dump_location = params.path_sketchy_features ) 
    
    else:
        features_sketchy_dict = pickle_load( params.path_sketchy_features )
    

    ## Generate a file containing glove vector corresponding to the classes being used.
    if(os.path.isfile(params.path_glove_vector) == False):
        generate_glove_vector()

    ## Load the s_encoder for both image and sketch. If saved model are not found, train and saved the model.
    if( not os.path.isfile( params.path_z_encoder_image ) ):
        z_encoder_image = encoder( in_dim = params.x_dim, z_dim = params.glove_dim )
        z_encoder_image = train_z_encoder( encoder_model = z_encoder_image,
                                           feature_dict = features_image_dict,
                                           dump_location = params.path_z_encoder_image )

    else:
        z_encoder_image = torch.load( params.path_z_encoder_image )


    if( not os.path.isfile( params.path_z_encoder_sketchy ) ):
        z_encoder_sketchy = encoder( in_dim = params.x_dim, z_dim = params.glove_dim )
        z_encoder_sketchy = train_z_encoder( encoder_model = z_encoder_sketchy,
                                           feature_dict = features_sketchy_dict,
                                           dump_location = params.path_z_encoder_sketchy )

    else:
        z_encoder_image = torch.load( params.path_z_encoder_image )


