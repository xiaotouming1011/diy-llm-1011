import torch
import torch.nn as nn

class Linear(nn.Modele):
    def __init__(self,in_features:int,out_features:int,device: None,dtype:None):
        super().__init__()
        self.in_features = in_features
        self.out_features= out_features
        #准备工厂参数
        #