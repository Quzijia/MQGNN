import torch
import torch.nn as nn

class QuatLinear(nn.Module):
    def __init__(self, in_q, out_q, bias=True):
        super().__init__()
        self.A = nn.Parameter(torch.randn(out_q, in_q))
        self.B = nn.Parameter(torch.randn(out_q, in_q))
        self.C = nn.Parameter(torch.randn(out_q, in_q))
        self.D = nn.Parameter(torch.randn(out_q, in_q))
        self.bias = nn.Parameter(torch.zeros(out_q, 4)) if bias else None
    #     self.reset_parameters()        

    # def reset_parameters(self):
    #     for param in (self.A, self.B, self.C, self.D):
    #         nn.init.kaiming_uniform_(param, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in = self.A.size(1)
    #         bound = 1 / math.sqrt(fan_in)
    #         nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):                 # x: (N,V,in_q,4)
        a,b,c,d = x.unbind(-1)            # each (N,V,in_q)
        A,B,C,D = self.A, self.B, self.C, self.D
        # -- 正确的 Hamilton 公式，einsum 接收两个张量 --
        u = torch.einsum('nvi,oi->nvo', a, A) - torch.einsum('nvi,oi->nvo', b, B) \
            - torch.einsum('nvi,oi->nvo', c, C) - torch.einsum('nvi,oi->nvo', d, D)
        v = torch.einsum('nvi,oi->nvo', a, B) + torch.einsum('nvi,oi->nvo', b, A) \
            + torch.einsum('nvi,oi->nvo', c, D) - torch.einsum('nvi,oi->nvo', d, C)
        w = torch.einsum('nvi,oi->nvo', a, C) - torch.einsum('nvi,oi->nvo', b, D) \
            + torch.einsum('nvi,oi->nvo', c, A) + torch.einsum('nvi,oi->nvo', d, B)
        z = torch.einsum('nvi,oi->nvo', a, D) + torch.einsum('nvi,oi->nvo', b, C) \
            - torch.einsum('nvi,oi->nvo', c, B) + torch.einsum('nvi,oi->nvo', d, A)
        out = torch.stack([u, v, w, z], dim=-1)             # (N,V,out_q,4)
        if self.bias is not None:
            out = out + self.bias
        return out

