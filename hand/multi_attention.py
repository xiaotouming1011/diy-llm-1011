import torch
import torch.nn as nn
from einops import rearrange #强烈推荐：使纬度交换语义清晰

class CausalSelfAttention(nn.module):
    def __init__(self,d_model:int,num_heads:int,max_seq_len=None,theta=None,device=None,dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads头数整除"
        self.d_model = d_model
        self.num_heads=num_heads
        self.d_k = d_model // num_heads

        #Q,K,V投影
        self.q_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self_k_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self_v_proj = Linear(d_model,d_model,device=device,dtype=dtype)

        #输出投影层,多头注意力拼接
        self.output_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        
        #RoPE

        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(theta=theta,max_seq_len=max_seq_len,device=device,dtype=dtype)
        else:
            self.rope = None
    
    
    
    
    def forward(self,x:torch.Tensor, token_positions:torch.Tensor = None)-> torch.Tensor:
        b,s,d = x.shape
        q= rearrange(self.q_proj(x), 'b s (h d_k) -> b h s d_k', h=self.num_heads)
        k= rearrange(self.k_proj(x), 'b s (h d_k) -> b h s d_k', h=self.num_heads)
        v= rearrange(self.v_proj(x), 'b s (h d_k) -> b h s d_k', h=self.num_heads)
        #RoPE
        if self.rope is not None:
                # 生成 [0,1,2,...,s-1] 的位置序列 → 形状 (s)
        # 增加 batch 维度 → (1, s)
        # 扩展到当前 batch 大小 → (b, s)
                if token_positions is None:
                    token_positions = torch.arange(s, device=x.device).unsqueeze(0).expand(b, s)  # (b, s)
                
                q = self.rope(q, token_positions)
                k = self.rope(k, token_positions)
        #掩码
        mask = torch.tril(torch.ones(s, s, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, s, s)

        attn_out = scaled_dot_product_attention(q, k, v, mask)  # (b, h, s, d_k)
        attn_out = rearrange(q, 'b h s d_k -> b s (h d_k)') 
                

        return self.output_proj(attn_out)