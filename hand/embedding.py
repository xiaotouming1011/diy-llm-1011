import torch
import torch.nn as nn

#embedding层
class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device=None,dtype=None):
        super().__init__()
        #分配内存并包装为参数
        factory_kwargs = {'device':device,'dtype':dtype}
        #nn.Parameter:被标记的张量，被标记才能通过优化器更新，随模型保存
        #权重纬度为vocab_size,d_model;torch.empty:申请内存空间，比zeros高效
        self.weight = nn.Parameter(torch.empty((num_embeddings,embedding_dim),**factory_kwargs))
        #初始化，mean=0,std = 1.0,截断在[-3,3]
        nn.init.trunc_normal_(self.weight,mean=0.0,std=1.0,a=-3.0,b=3.0)
    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
