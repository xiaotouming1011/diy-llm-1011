import torch
import math
from torch.nn import softmax

def scaled_dot_product_attention(
        Q: torch.Tensor,
        K:torch.Tensor,
        V:torch.Tensor,
        mask: torch.Tensor = None

) -> torch.Tensor:
#mask是布尔矩阵，True和false组成
    
    #获取 Query 张量最后一个维度的长度，并赋值给变量 d_k。
    d_k =Q.size(-1)

    #计算相似度分数
    scores = torch.einsum('...nk,...mk ->...nm',Q,K)/ math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == False,float('-inf'))
    probs = softmax(scores,dim=-1)
    output = torch.einsum('...nm,...mk ->...nk',probs,V)


    return output