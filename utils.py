import params
import pickle
import torch
import numpy as np

def chunks(lst, n=params.batch_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

from collections import deque

class last_k:
    def __init__( self, k):
        self.deque = deque()
        self.size = k

    def append(self, value):
        if(len(self.deque) < self.size):
            self.deque.appendleft(value)
        else:
            self.deque.pop()
            self.deque.appendleft(value)


def pickle_load( file_path : str ) -> dict:
    with open( file_path, 'rb' ) as f:
        file = pickle.load( f )

    return file

def make_tensor(array : np.ndarray) -> torch.Tensor:
    array = torch.tensor(array)
    if(params.gpu_flag):
        array = array.cuda(params.gpu_name)

    return array

def cuda(model):
    if(params.gpu_flag):
        model.cuda(params.gpu_name)
# import numpy as np 
# from time import time

# s = time()
# a = np.random.random((1000,1000))
# a = a*a
# end = time()
# print('time taken')
