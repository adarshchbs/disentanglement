import numpy as np 
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import load_data
import random
from utils import chunks
import params

# classifier_path = '/home/iacv/project/disentanglement/saved_model/classifier_weights'

# classifier = np.load(classifier_path, allow_pickle = True)

# for i in classifier:
#     print(classifier[i].shape)

model_path = '/home/iacv/project/disentanglement/resnet_50_da.pt'

model = torch.load(model_path,map_location=torch.device('cpu'))

weight = model[0].fc.weight.detach().numpy()

weight_np = np.zeros((2048,87))

for i, w in enumerate(weight):
    weight_np[:,i] = w

# print(weight_np.shape)


bias = model[0].fc.bias.detach().numpy()

def classifier(features):
    pred =  features @ weight_np + bias
    exp = np.exp(pred)
    prob = exp/np.sum(exp,axis=1).reshape(features.shape[0],-1)
    arg_max = np.argmax(prob, axis=1)
    # prob_max = np.max(prob,axis = 1)
    return arg_max, prob


def entropy(prob):
    ret = -1 * prob * np.log2(prob)
    return np.sum(ret,axis=1)

# sketch_feature = load_data.sketch_x_train
# sketch_label = load_data.sketch_y_train

# correct_classified = []
# correct_label = []
# false_classified = []
# false_label = []

# for f,l in zip(chunks(sketch_feature),chunks(sketch_label)):
#     pred, p = classifier(f)
#     correct = (pred == l) * 1
    
#     for c,f_i, l_i in zip(correct,f, l):
#         if(c==1):
#             correct_classified.append(f_i)
#             correct_label.append(l_i)

#         else:
#             false_classified.append(f_i)
#             false_label.append(l_i)

# correct_classified = np.array(correct_classified  )
# correct_label = np.array(correct_label)
# false_classified = np.array(false_classified)
# false_label = np.array(false_label)

# print('features saved')

class confidence_network(torch.nn.Module):
     
    def __init__(self):
        super(confidence_network, self).__init__()

        self.linear_1 = torch.nn.Linear(params.x_dim,128)
        self.bn_1= torch.nn.BatchNorm1d(128)
        self.linear_2 = torch.nn.Linear(128,128)
        self.bn_2 = torch.nn.BatchNorm1d(128)
        self.linear_3 = torch.nn.Linear(128,params.num_class)
        self.relu = torch.nn.ReLU()

    def forward(self, input_1, input_2,alpha):
        ret_1 = self.linear_1(input_1)
        ret_2 = self.linear_1(input_2)
        ret_1 = self.bn_1(ret_1)
        ret_2 = self.bn_1(ret_2)
        ret_1 = self.relu(ret_1)
        ret_2 = self.relu(ret_2)
        ret_1 = self.linear_2(ret_1)
        ret_2 = self.linear_2(ret_2)
        ret_1 = self.bn_2(ret_1)
        ret_2 = self.bn_2(ret_2)
        ret = alpha * ret_1 + (1-alpha)*ret_2
        # ret = self.relu(ret)
        ret = self.linear_3(ret)

        return ret

def loss_norm(output,label, alpha, step):
    size = output.shape[0]
    exp = torch.exp(output)
    prob = exp/torch.sum(exp,dim = 1).reshape(size,-1)

    mask_1 = torch.zeros((size,params.num_class),dtype=torch.bool)
    mask_2 = torch.zeros((size,params.num_class),dtype=torch.bool) 

    for b,(i,j) in enumerate(zip(label[0],label[1])):
        mask_1[b,i] = True
        mask_2[b,j] = True

    prob_1 = prob[mask_1]
    prob_2 = prob[mask_2]

    if( (step+1) % 1400 ==0):
        print("==============prob=========")
        print(prob_1)
        print(prob_2)
        print("===========================")

    # ret = torch.sum( (prob_1-alpha)**2 + 100*(prob_2-1+alpha)**2 )/size
    ret = torch.sum( torch.abs(prob_1-alpha) + 10 * torch.abs(prob_2-1+alpha) )/size

    return ret

feature_dict = defaultdict(list)

for label, feature in zip(load_data.sketch_y_train, load_data.sketch_x_train):
    feature_dict[label].append(feature)

feature_dict_1 = {}
for i in feature_dict:
    feature_dict_1[i] = np.array(feature_dict[i])



def same_class_sample(l = None):
    if(l is None):
        labels = np.random.choice(np.arange(params.num_class), params.batch_size)
    else:
        labels = l
    f = []
    for i in labels:
        index = np.random.randint(0,len(feature_dict[i]),2)
        f.append(feature_dict_1[i][index])
    f = np.array(f)
    f_0 = f[:,0,:]
    f_1 = f[:,1,:]
    return f_0, f_1, labels





def train_network( features, labels, alpha, dump_location ):

    features_1 = features
    labels_1 = labels
    features_2 = features.copy()
    labels_2 = labels.copy()


    model = confidence_network()
    model.train()
    optimizer = torch.optim.Adam(  model.parameters() ,
                            lr = 0.0001
                            )
    criterion = torch.nn.CrossEntropyLoss()
    
 
    for epoch in range( 10 ):
        index_1 = np.arange(len(features_1))
        random.shuffle(index_1)
        index_2 = np.arange(len(features_2))
        random.shuffle(index_2)

        features_1 = features_1[index_1]
        labels_1 = labels_1[index_1]

        features_2 = features_2[index_2]
        labels_2 = labels_2[index_2]

        total = 0
        correct_pred = 0
        for step, ( f1, l1, f2, l2 ) in enumerate( zip(chunks(features_1),chunks(labels_1),
                                                            chunks(features_2), chunks(labels_2)) ):

            try:
                
                f1 = torch.tensor(f1)
                f2 = torch.tensor(f2)
                # norm_features = alpha * f1 + (1-alpha)*f2

                # norm_features = torch.tensor(norm_features)

                labels = torch.tensor(np.vstack([l1,l2]),dtype=torch.long)



                optimizer.zero_grad()
                
                preds_1 = model(f1,f2,alpha) 
                loss_1 = loss_norm( preds_1, labels, alpha, step )
                
                f_s_1, f_s_2, l_s = same_class_sample()
                f_s_1, f_s_2 = torch.tensor(f_s_1), torch.tensor(f_s_2)
                preds_2 = model(f_s_1,f_s_2,alpha)
                l_s = torch.tensor(l_s, dtype=torch.long)
                loss_2 = criterion(preds_2,l_s)

                if(epoch <2):
                    loss = loss_1
                else:
                    loss = loss_1 + loss_2/10
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(preds_2.data,1)
                total += l_s.size(0)
                correct_pred += (predicted == l_s).sum().item()
            
            except ValueError:
                pass

            if(step % 200 == 0):
                print("epoch {}, step {}, loss_1 {:.4f}, loss_2 {:.4f}".format(epoch,step, loss.data.item(), 0))
        print("++++++++++++++++")
        print("accuracy after {} epochs is {}".format(epoch,correct_pred/total))

    torch.save(model,dump_location)

    return model


conf_model = train_network(load_data.sketch_x_train,load_data.sketch_y_train, 0.8,'/home/iacv/project/disentanglement/saved_model/conf_model.pt')
conf_model = model = torch.load('/home/iacv/project/disentanglement/saved_model/conf_model.pt')



l = 2048
m = 0.8

x_val = load_data.quick_draw_x_val
y_val = load_data.quick_draw_y_val

index = np.arange(len(x_val))
random.shuffle(index)

x_val = x_val[index][:l]
y_val = y_val[index][:l]



# predicted_conf = []
# for i in temp:
#     if(i[1] > 0.7):
#         predicted_conf.append(1)

#     else:
#         predicted_conf.append(0)

# predicted_conf = np.array(predicted_conf)

pred_label , _ = classifier(x_val)

pred_correct = (pred_label == y_val)
print(np.sum(pred_correct)/len(pred_correct))


generated_feature,_,_ = same_class_sample(pred_label)
generated_feature = torch.tensor(generated_feature)
conf_model.eval()
conf = conf_model(torch.tensor(x_val),generated_feature,0.8)
temp = torch.exp(conf)
temp = temp/torch.sum(temp,dim=1).reshape(l,-1)

pred_label_new = torch.argmax(temp,dim=1).detach().numpy()
temp = (pred_label_new == pred_label)
print(np.sum(temp))
pred_correct = (pred_label_new[temp] == y_val[temp])
print(np.sum(pred_correct)/len(pred_correct))

# arr = np.zeros((2,2))

# for con, cf in zip(pred_correct,predicted_conf):
#     if(con and (cf == 1) ):
#         arr[0][0] += 1
#     elif(con and (cf == 0) ):
#         arr[0][1] += 1
    
#     elif( not con and (cf ==1) ):
#         arr[1][0] += 1

#     elif( not con and (cf ==0) ):
#         arr[1][1] += 1

# print(arr)






# def eval(mod_feat):
#     array = []
#     for i in mod_feat:
#         _, prob = classifier(i)
#         temp = np.diag(prob)
#         array.append(temp/np.sum(temp))
#     array = np.array(array)
#     print('array_shape ',array.shape)
#     ret = entropy(array)
#     return ret

# print(len(x_val))

# pred_label, pred_prob = classifier(x_val)

# # ent = eval(modified_features)
# ent = entropy(pred_prob)

# correct = (pred_label == y_val)*1

# correct_ent = []
# false_ent = []

# for c,e in zip(correct,ent):
#     if(c==1):
#         correct_ent.append(e)

#     else:
#         false_ent.append(e)


# correct_ent = np.array(correct_ent)
# false_ent = np.array(false_ent)

# print(len(correct_ent)/(len(correct_ent)+len(false_ent)))

# plt.hist(correct_ent,np.linspace(0,7,10))
# plt.show()
# plt.hist(false_ent,np.linspace(0,7,10))
# plt.show()




# feature_dict = defaultdict(list)

# for l,f in zip(sketch_label,sketch_feature):
#     feature_dict[l].append(f)

# feature_from_each_class = []
# for i in range(87):
#     temp = np.array(feature_dict[i])
#     temp = np.mean(temp,axis=0)
#     feature_from_each_class.append(temp)

# feature_from_each_class = np.array( feature_from_each_class )

# modified_features = []

# for i in x_val:
#     temp = m * i + (1-m) * feature_from_each_class
#     modified_features.append( temp )
