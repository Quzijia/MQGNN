import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import QuatLinear


class QuatNorm(nn.Module):
    """对每个 quaternion 做安全归一化。"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, q):
        if q.shape[-1] != 4:
            print(f"Warning: Input tensor shape {q.shape} does not have 4 channels for quaternion normalization.")
            raise ValueError("Input tensor must have 4 channels for quaternion normalization.")
        norm = q.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        return q / norm

class QGCN(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, k_num: int = 1,
                 dropout: float = 0.5, residual: bool = True):
        super().__init__()
        assert in_channels % 4 == 0 and out_channels % 4 == 0, \
            "Channels must be multiples of 4 (w,x,y,z)."
        self.k_num = k_num
        self.in_q  = in_channels // 4
        self.out_q = out_channels // 4

        # quaternion 1×1 linear (per node)
        self.qlinear = QuatLinear(self.in_q, self.out_q)
        self.dropout = nn.Dropout(dropout)
        self.bn_pre  = nn.BatchNorm1d(in_channels)
        self.qnorm   = QuatNorm()

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        # x: (N,C_in,V) ,  A: (k_num,V,V)
        res = self.residual(x)
        N, Cin, V = x.shape
        # --- Pre‑activation ---
        out = self.bn_pre(x)
        # out = F.relu(out, inplace=True)
        out = F.silu(out)  # 预激活
        # --- Quaternion linear ---
        out = out.permute(0,2,1).contiguous().view(N, V, self.in_q, 4)  # (N,V,in_q,4)
        out = self.qlinear(out)  
        out = self.qnorm(out)                                        # (N,V,out_q,4)
        out = out.view(N, V, self.out_q*4).permute(0,2,1).contiguous()   # (N,C_out,V)
        out = self.dropout(out)
        # --- Message passing ---
        out = out.view(N, self.k_num, out.shape[1]//self.k_num, V)
        out = torch.einsum('nkcv,kvw->ncw', out, A)                      # (N,C_out,V)
        # return F.relu(out + res, inplace=True)
        return F.silu(out + res)

class Gcn(nn.Module):

    def __init__(self, in_channels, out_channels,k_num=1, # 对应原来的 A.size(0)
                 dropout=0.5,
                  residual=True):
        super().__init__()
        self.k_num = k_num
        # 相当于对输入的每个节点做线性映射(1x1卷积)，并为每个邻接通道输出 out_channels
        # 总共有 k_num 个邻接通道，需要 out_channels*k_num 大小
        self.conv = nn.Conv1d(in_channels,
                              out_channels * k_num,
                              kernel_size=1)

        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.silu = nn.SiLU()
        self.bn_pre  = nn.BatchNorm1d(in_channels)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.silu = nn.SiLU()

    def forward(self, x, A):
        res = self.residual(x)
        # --- 预激活 ---
        out = self.bn_pre(x)
        out = F.silu(out)
        # 主路与原版相同
        out = self.conv(out)
        out = self.dropout(out)
        N, kc, V = out.shape
        out = out.view(N, self.k_num, kc//self.k_num, V)
        out = torch.einsum('nkcv,kvw->ncw', (out, A))
        # 不再额外 BN
        # return F.relu(out + res, inplace=True)
        return F.silu(out + res)

class DecoderGcn(nn.Module):

    
    def __init__(self, in_channels, out_channels, k_num, dropout=0.5):
        super().__init__()
        
        # Ensure channels are multiples of 4 for quaternion operations
        assert in_channels % 4 == 0, "in_channels must be multiple of 4 for quaternion features"
        assert out_channels % 4 == 0, "out_channels must be multiple of 4 for quaternion features"
        
        self.k_num = k_num
        self.in_q = in_channels // 4
        self.out_q = out_channels // 4
        
        # Use quaternion linear operations instead of regular Conv1d
        self.quat_linear = QuatLinear(self.in_q, self.out_q * k_num)
        self.dropout = nn.Dropout(dropout)
        self.quat_bn = nn.BatchNorm1d(out_channels)
        self.silu = nn.SiLU()
        
    def forward(self, x, A_skl):  
        """
        Forward pass of quaternion graph decoder.
        
        Args:
            x: [N, in_channels, V] - Input quaternion features
            A_skl: [k_num, V, V] - Adjacency matrices
            
        Returns:
            Normalized quaternion features [N, out_channels, V]
        """
        N, C, V = x.size()
        actual_in_q = C // 4
        
        # Apply quaternion linear transformation
        # First reshape to expose quaternion structure
        x_reshaped = x.permute(0, 2, 1).contiguous()  # [N, V, C]
        x_quat = x_reshaped.view(N, V, actual_in_q, 4)  # [N, V, in_q, 4]
        
        # Apply quaternion linear layer
        x_transformed = self.quat_linear(x_quat)  # [N, V, out_q*k_num, 4]
        
        # Reshape for graph operations
        x_graph = x_transformed.view(N, V, self.out_q * self.k_num * 4)
        x_graph = x_graph.permute(0, 2, 1).contiguous()  # [N, out_channels*k_num, V]
        x_graph = self.dropout(x_graph)
        
        # Message passing with adjacency matrices
        x_graph = x_graph.view(N, self.k_num, x_graph.size(1) // self.k_num, V)  # [N, k_num, out_channels, V]
        x_out = torch.einsum('nkcv,kvw->ncw', (x_graph, A_skl))  # [N, out_channels, V]
        
        x_out = self.quat_bn(x_out)
        x_out = self.silu(x_out)
        return x_out
    
class DecodeGcn(nn.Module):
    """
    与原先 decode 类似，这里保留，但去掉了时间卷积只做图上的消息传递。
    x: [N, d, V], A_skl: [k, V, V]
    """
    
    def __init__(self, in_channels, out_channels, k_num,
                dropout=0.5):
        super().__init__()

        self.k_num = k_num
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels*(k_num), 
                              kernel_size=1,
                              stride=1, 
                              padding=0, 
                            )
        self.dropout = nn.Dropout(dropout)
        self.qnorm   = QuatNorm()

    def forward(self, x, A_skl):  
        # x: [N, in_channels, V]
        x = self.conv(x)                  # [N, out_channels*k_num, V]
        x = self.dropout(x)
        N, kc, V = x.size()
        x = x.view(N, self.k_num, kc // self.k_num, V)  # [N, k_num, out_channels, V]
        x = torch.einsum('nkcv,kvw->ncw', (x, A_skl))    # [N, out_channels, V]
        N, C, V = x.shape
        if C % 4 == 0:
            x = x.view(N, C//4, 4, V)               # (N,Q,4,V)
            # x = QuatNorm()(x.permute(0,3,1,2))      # (N,V,Q,4)
            x = x.permute(0,2,3,1).contiguous()     # (N,Q,4,V)
            x = x.view(N, C, V) 
        return x                   # 还原形状
