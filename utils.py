import params
import pickle

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


def pickle_load(file_path):
    with open( file_path, 'rb' ) as f:
        file = pickle.load( f )

    return file