import torch
import numpy as np
import re
import pickle
import params


def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def generate_glove_vector():
    batch_size = params.batch_size
    gloveFile = 'glove.6B.200d.txt'

    glove_model = loadGloveModel(gloveFile)

    path_class_list = '/home/adarsh/project/CDAN/pytorch/common_class_list.txt'

    class_list = np.loadtxt(path_class_list,dtype='str')
    # print(class_list)

    glove_vector = []
    for i in class_list:
        try:
            glove_vector.append(glove_model[i])
        except:
            word_split = re.split('_',i)
            temp = np.zeros(params.glove_dim)
            for s in word_split:
                temp += glove_model[s]
            glove_vector.append(temp/len(word_split))

    glove_vector = torch.tensor(glove_vector)




    batch_glove_vector = torch.stack([glove_vector for i in range(batch_size)])

    with open('glove_vector','wb') as f:
        pickle.dump({'glove_vector':glove_vector,'batch_glove_vector':batch_glove_vector}, f)