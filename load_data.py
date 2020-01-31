import numpy as np

sketch_x_train = np.load('/home/adarsh/project/disentanglement/saved_features/da_sketchy_feature_train.npy',allow_pickle=True)
sketch_y_train = np.load('/home/adarsh/project/disentanglement/saved_features/da_sketchy_label_train.npy',allow_pickle=True)

sketch_x_val = np.load('/home/adarsh/project/disentanglement/saved_features/da_sketchy_feature_val.npy',allow_pickle=True)
sketch_y_val = np.load('/home/adarsh/project/disentanglement/saved_features/da_sketchy_label_val.npy',allow_pickle=True)

quick_draw_x_val = np.load('/home/adarsh/project/disentanglement/saved_features/da_quick_draw_feature_val.npy',allow_pickle=True)
quick_draw_y_val = np.load('/home/adarsh/project/disentanglement/saved_features/da_quick_draw_label_val.npy',allow_pickle=True)

# sketch_y_train = np.array(list(map(int,sketch_y_train)))
# np.save('/home/adarsh/project/disentanglement/saved_features/da_sketchy_label_train.npy',sketch_y_train)

# print(np.average(np.abs(sketch_x_train[0:1000])))