import torch


def pairwise_distance(A, B):
    square_difference = (A-B)**2
    return torch.sum(square_difference)


