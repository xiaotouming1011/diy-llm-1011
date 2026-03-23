import torch
import torch.nn as nn

class Linear(nn.Modele):
    def __init__(self,in_features:int,out_features:int,device: None,dtype:None):
        super().__init__()
        self.in_features = in_features
        self.out_features= out_features
        #准备工厂参数
        factory_kwargs = {'device':device,'dtype':type}

        #占位置，torch.empty申请内存空间，定义权重参数
        #**factory_kwargs	解包关键字参数，传给torch.empty（定制张量的设备、数据类型等属性）
        self.weight = nn.Parameter(torch.empty((out_features,in_features),**factory_kwargs))
        
        std = (2.0/(in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight,mean = 0.0,std = std,a=-3*std,b=3*std)

    def forward(self,x:torch.Tensor) -> torch.Tensor:
    
        #x 形状：[..., in_features]，使用 einsum 表达："输入的最后一位 i 与 权重的最后一位 i 相乘，输出 o"，这种写法比 x @ self.weight.T 更具可读性，且支持任意 Batch 维度
        #einsum 的公式字符串格式是：输入维度描述 -> 输出维度描述，比如你看到的 ...i, oi -> ...o：
        #不用转置了
        return torch.einsum('...i, oi -> ...o', x, self.weight)









