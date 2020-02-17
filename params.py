import os

gpu_flag = True
gpu_name = 'cuda:0'

x_dim = 2048
num_class = 87
num_query = 5

batch_size = 128
eval_batch_size = 512
glove_dim = 200

pretrain_lr = 1e-4
num_epochs_pretrain = 20
eval_step_pre = 1

fusion_iter_len = 100000

num_epochs_pretrain = 30
num_epochs_style = 30
num_epochs_fusion = 50
log_step_pre = 60

path_class_list = '/home/adarsh/project/disentanglement/extra/common_class_list.txt'

dir_saved_model = '/home/adarsh/project/disentanglement/saved_model/'
dir_saved_feature = '/home/adarsh/project/disentanglement/saved_features/'
dir_dataset = '/home/adarsh/project/disentanglement/dataset/'
dir_extra = '/home/adarsh/project/disentanglement/extra/'

os.makedirs( dir_saved_model, exist_ok = True)
os.makedirs( dir_saved_feature, exist_ok = True )
os.makedirs( dir_extra, exist_ok = True )


path_model_image = dir_saved_model + 'resnet_50_image.pt'
path_model_sketchy = '/home/adarsh/project/disentanglement/saved_model/resnet_50_sketchy.pt'

path_z_encoder_sketchy = dir_saved_model + 'z_encoder_sketch.pt'
path_s_encoder_sketchy = dir_saved_model + 's_encoder_sketch.pt'
path_adv_model_sketchy = dir_saved_model + 'adv_sketch.pt'
path_recon_model_sketchy = dir_saved_model + 'reconstruck_sketch.pt'

path_z_encoder_image = dir_saved_model + 'z_encoder_image.pt'
path_s_encoder_image = dir_saved_model + 's_encoder_image.pt'
path_adv_model_image = dir_saved_model + 'adv_image.pt'
path_recon_model_image = dir_saved_model + 'reconstruck_image.pt'

path_fusion_model = dir_saved_model + 'fusion_model.pt'

path_image_dataset = dir_dataset + 'images/'
path_sketchy_dataset = dir_dataset + 'sketchy/'
path_quickdraw_dataset = dir_dataset + 'quick_draw/'


path_image_features = dir_saved_feature + 'image_features.p'
path_sketchy_features = dir_saved_feature + 'sketchy_features.p'
path_quickdraw_features = dir_saved_feature + 'quick_draw_features.p'

path_image_file_list = dir_extra + 'images_file_list.p'
path_sketchy_file_list = dir_extra + 'sketchy_file_list.p'
path_quickdraw_file_list = dir_extra + 'quick_draw_file_list.p'


path_model = '/home/adarsh/project/disentanglement/resnet_50_da.pt'

path_sketch_z_encoder = '/home/adarsh/project/disentanglement/sketch_encoder.pt'

path_glove_vector = '/home/adarsh/project/disentanglement/glove_vector'