import numpy as np 
# import torch
# from scipy.spatial import KDTree

# import load_data

# tree = KDTree(load_data.sketch_x_val[11:], leafsize=10)

# y_val = load_data.sketch_x_val
# label = load_data.sketch_y_val


# for counter,q in enumerate(y_val):
#     [d,i] = tree.query(q)
#     print(d,i)
#     print(label[counter],label[i])
#     if(counter ==10):
#         break
# d_min = 10000
# for i, vec in enumerate(y_val[11:]):
#     d = np.sum(np.abs(y_val[0]-vec))
#     if(d < d_min):
#         d_min = d
#         arg = i 

# print(d_min,i)
# print(np.sum(np.abs(y_val[0]-y_val[2154])))

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

    # def __repr__(self):
    #     return self.deque

a = last_k(3)

for i in range(10):
    a.append(i)
    print(a.deque)
