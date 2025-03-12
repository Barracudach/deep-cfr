import torch.nn as nn
import torch


class Encoder(nn.Module):
     def __init__(self, features_size, internal_features_size):
        super(Encoder, self).__init__()