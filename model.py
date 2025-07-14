import torch
import torch.nn as nn
import torch.nn.functional as F

from graph import  Graph_J
from module import *
from utils import quat_mul_batch
import math
    
class HierarchicalJointModule(nn.Module):
    """
    Enhanced joint relationship module that models different levels of joint dependencies:
    1. Kinematic chain (parent-child relationships)
    2. Functional relationships (e.g., symmetric joints like left-right)
    3. Global coordination patterns
    """
    def __init__(self, feature_dim, num_joints=7):
        super().__init__()
        self.num_joints = num_joints
        self.feature_dim = feature_dim
        
        # Define key joints with dependencies
        
        # Define kinematic chain for joints (parent-child relationships)
        self.kinematic_parents = {
            1: 0,  # Left upper leg -> pelvis
            2: 1,  # Left lower leg -> left upper leg
            3: 2,  # Left foot -> left lower leg
            4: 0,  # Right upper leg -> pelvis
            5: 4,  # Right lower leg -> right upper leg
            6: 5   # Right foot -> right lower leg
        }
        
        # Define symmetric pairs (functional relationships)
        self.symmetric_pairs = [
            (1, 4),  # Left/right upper leg
            (2, 5),  # Left/right lower leg
            (3, 6)   # Left/right foot
        ]
        
        # 1. Kinematic parent-child processors
        self.parent_transform = nn.ModuleList([
            nn.Linear(feature_dim * 2, feature_dim) 
            for _ in range(num_joints)
        ])
        
        # 2. Symmetric joint processors
        self.symmetric_transform = nn.ModuleList([
            nn.Linear(feature_dim * 2, feature_dim)
            for _ in range(len(self.symmetric_pairs))
        ])
        
        # 3. Global coordination - lightweight attention
        # (Using a more efficient implementation than full attention)
        self.global_key = nn.Linear(feature_dim, feature_dim // 4)
        self.global_query = nn.Linear(feature_dim, feature_dim // 4)
        self.global_value = nn.Linear(feature_dim, feature_dim)
        self.global_output = nn.Linear(feature_dim, feature_dim)
        
        # # Additional processing for ankle joints (more dependencies)
        # self.ankle_processors = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(feature_dim * 4, feature_dim * 2),
        #         nn.SiLU(),
        #         nn.Linear(feature_dim * 2, feature_dim)
        #     ) for _ in self.ankle_indices
        # ])
        
        # Final layer normalization and gate
        self.norm = nn.LayerNorm(feature_dim)
        self.gate = nn.Parameter(torch.ones(1, 1, num_joints) * 0.8)
        
    def forward(self, x):
        """
        Args:
            x: Joint features [B, C, V]
        Returns:
            Enhanced features with joint dependencies [B, C, V]
        """
        B, C, V = x.shape
        x_trans = x.permute(0, 2, 1)  # [B, V, C]
        
        # Create output tensor
        output = x_trans.clone()
        output_list = [output[:, j].clone() for j in range(self.num_joints)]
        
        # 1. Process kinematic dependencies (parent-child)
        for joint_idx in range(self.num_joints):
            if joint_idx == 0:  # Root joint (pelvis) has no parent
                continue
                
            # Get parent joint
            parent_idx = self.kinematic_parents[joint_idx]
            
            # Combine joint and parent features
            joint_features = x_trans[:, joint_idx]
            parent_features = x_trans[:, parent_idx]
            combined = torch.cat([joint_features, parent_features], dim=1)
            
            # Transform and update with residual connection
            transformed = self.parent_transform[joint_idx](combined)
            output_list[joint_idx] = output_list[joint_idx] + transformed
        
        # 2. Process symmetric joint relationships
        for i, (left_idx, right_idx) in enumerate(self.symmetric_pairs):
            # Process left-to-right influence
            left_features = x_trans[:, left_idx]
            right_features = x_trans[:, right_idx]
            combined_lr = torch.cat([left_features, right_features], dim=1)
            transformed_lr = self.symmetric_transform[i](combined_lr)
            
            # Apply with a small residual connection
            output_list[right_idx] = output_list[right_idx] + 0.2 * transformed_lr
            
            # Process right-to-left influence (using same weights for symmetry)
            combined_rl = torch.cat([right_features, left_features], dim=1)
            transformed_rl = self.symmetric_transform[i](combined_rl)
            
            # Apply with a small residual connection
            output_list[left_idx] = output_list[left_idx] + 0.2 * transformed_rl
        
        # 3. Global coordination patterns (efficient attention)
        # Compute keys and queries for all joints
        keys = self.global_key(x_trans)  # [B, V, C//4]
        queries = self.global_query(x_trans)  # [B, V, C//4]
        values = self.global_value(x_trans)  # [B, V, C]
        
        # Compute attention scores
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))  # [B, V, V]
        attn_scores = attn_scores / (self.feature_dim ** 0.25)  # Scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply attention
        global_context = torch.matmul(attn_weights, values)  # [B, V, C]
        global_output = self.global_output(global_context)  # [B, V, C]
        
        # Apply global context with gating
        for j in range(self.num_joints):
            # Dynamic gate based on joint position in hierarchy
            gate_value = self.gate[:, :, j]
            output_list[j] = output_list[j] * (1 - gate_value) + global_output[:, j] * gate_value
        
        
        # Apply normalization
        for j in range(self.num_joints):
            output_list[j] = self.norm(output_list[j])
        output = torch.stack(output_list, dim=1)
            
        return output.permute(0, 2, 1)  # Back to [B, C, V]
    

class HighFrequencyQuaternionFilter(nn.Module):
    def __init__(self, channels=4, num_joints=7, fps=30.0):
        super().__init__()
        # Learnable adaptive filtering parameters
        self.base_cutoff = nn.Parameter(torch.ones(num_joints) * 2.0)  # Lower base cutoff
        self.beta = nn.Parameter(torch.ones(num_joints) * 0.015)  # Higher adaptation
        self.adaptive_strength = nn.Parameter(torch.ones(num_joints) * 0.8)  # Filter strength
        
        # Outlier detection parameters
        self.outlier_threshold = nn.Parameter(torch.ones(num_joints) * 3.0)
        self.outlier_correction = nn.Parameter(torch.ones(num_joints) * 0.9)
        
        self.dt = 1.0 / fps
        
        # Component influence with learnable weights
        self.component_influence = nn.Parameter(torch.ones(channels))
        
        # Multi-scale filtering - enhanced state tracking
        self.register_buffer('prev_filtered', torch.zeros(1, channels, num_joints))
        self.register_buffer('prev_raw', torch.zeros(1, channels, num_joints))
        self.register_buffer('velocity_history', torch.zeros(1, 3, num_joints))  # Store velocity history
        
    def detect_outliers(self, curr, prev, velocity_std):
        """Detect and correct sudden outliers"""
        # Calculate frame-to-frame difference
        diff = torch.norm(curr - prev, p=2, dim=1)  # [B, V]
        
        # Dynamic threshold based on historical velocity
        dynamic_threshold = self.outlier_threshold.unsqueeze(0) * (1.0 + velocity_std)
        
        # Detect outliers
        outlier_mask = diff > dynamic_threshold
        
        # Correction factor for outliers
        correction_factor = torch.where(
            outlier_mask,
            self.outlier_correction.unsqueeze(0),
            torch.ones_like(diff)
        )
        
        return correction_factor.unsqueeze(1)  # [B, 1, V]
    
    def forward(self, x):
        """
        Apply quaternion filtering with enhanced outlier detection
        
        Args:
            x: [B, T, C, V] - batch, time, quaternion components, joints
        """
        B, T, C, V = x.shape
        filtered = torch.zeros_like(x)
        
        # Ensure first frame is normalized
        filtered[:, 0] = F.normalize(x[:, 0], p=2, dim=1)
        
        # Make sure buffers have the right batch size
        if self.prev_raw.shape[0] != B:
            self.prev_raw = torch.zeros(B, C, V, device=x.device)
            self.prev_filtered = torch.zeros(B, C, V, device=x.device)
            self.velocity_history = torch.zeros(B, 3, V, device=x.device)
        
        # Calculate velocity statistics for adaptive filtering
        velocities = []
        for t in range(1, min(T, 4)):  # Use recent frames for velocity estimation
            vel = torch.norm(x[:, t] - x[:, t-1], p=2, dim=1)
            velocities.append(vel)
        
        if velocities:
            velocity_tensor = torch.stack(velocities, dim=1)  # [B, T-1, V]
            velocity_std = torch.std(velocity_tensor, dim=1)   # [B, V]
        else:
            velocity_std = torch.zeros(B, V, device=x.device)
        
        # Update velocity history
        if T > 1:
            current_vel = torch.norm(x[:, -1] - x[:, -2], p=2, dim=1)  # [B, V]
            self.velocity_history = torch.cat([
                self.velocity_history[:, 1:], 
                current_vel.unsqueeze(1)
            ], dim=1)
        
        # Progressive filtering through time
        for t in range(1, T):
            curr = x[:, t]  # [B, C, V]
            prev = x[:, t-1] if t > 1 else self.prev_raw
            prev_filtered = filtered[:, t-1]
            
            # Detect and handle outliers
            outlier_correction = self.detect_outliers(curr, prev, velocity_std)
            
            # Calculate angular velocity for adaptive filtering
            q_curr = F.normalize(curr, p=2, dim=1)
            q_prev = F.normalize(prev, p=2, dim=1)
            
            # Calculate relative quaternion
            q_prev_conj = torch.cat([q_prev[:, 0:1], -q_prev[:, 1:]], dim=1)
            q_rel = quat_mul_batch(q_curr, q_prev_conj)
            
            # Extract angle (angular velocity) with better numerical stability
            angle = 2.0 * torch.acos(torch.clamp(torch.abs(q_rel[:, 0]), 0.0, 0.9999))  # [B, V]
            velocity = angle / self.dt  # rad/sec
            
            # Enhanced adaptive cutoff with outlier consideration
            base_cutoff = torch.clamp(self.base_cutoff.unsqueeze(0), 0.5, 15.0)
            adaptive_cutoff = base_cutoff + velocity * self.beta.unsqueeze(0)
            
            # Apply outlier correction to cutoff
            adaptive_cutoff = adaptive_cutoff * outlier_correction.squeeze(1)
            
            # Convert to alpha parameter with better numerical stability
            rc = 1.0 / (2.0 * math.pi * torch.clamp(adaptive_cutoff, min=0.1, max=20.0))
            alpha_base = self.dt / (rc + self.dt)
            
            # Apply adaptive strength
            alpha = alpha_base * self.adaptive_strength.unsqueeze(0)
            alpha = torch.clamp(alpha, 0.0, 0.95)
            
            # Progressive filtering: stronger for high velocities
            velocity_factor = torch.clamp(velocity / 10.0, 0.0, 1.0)
            alpha = alpha * (1.0 + velocity_factor)
            alpha = torch.clamp(alpha, 0.0, 0.98)
            
            # Apply component influence if provided
            comp_influence = F.softmax(self.component_influence, dim=0)
            
            # Perform quaternion filtering using SLERP
            q_filtered = self.quaternion_slerp_filter(
                q_curr, prev_filtered, alpha, comp_influence
            )
            
            # Apply outlier correction to the final result
            q_filtered = q_filtered * outlier_correction.squeeze(1).unsqueeze(1)
            
            # Update filtered sequence
            filtered_new = filtered.clone()
            filtered_new[:, t] = q_filtered
            filtered = filtered_new
        
        # Update state for next batch
        if self.training:
            self.prev_filtered = filtered[:, -1].detach().clone()
            self.prev_raw = x[:, -1].detach().clone()
            
        return filtered
    
    def quaternion_slerp_filter(self, q_curr, q_prev, alpha, comp_influence=None):
        """
        Apply quaternion filtering using SLERP with improved numerical stability
        
        Args:
            q_curr: Current quaternion [B, C, V]
            q_prev: Previous filtered quaternion [B, C, V] 
            alpha: Filter strength [B, V]
            comp_influence: Component influence weights [C]
        
        Returns:
            Filtered quaternion [B, C, V]
        """
        # Ensure unit quaternions
        q_curr = F.normalize(q_curr, p=2, dim=1)
        q_prev = F.normalize(q_prev, p=2, dim=1)
        
        # Ensure shortest path (handle double cover)
        dot = torch.sum(q_prev * q_curr, dim=1, keepdim=True)  # [B, 1, V]
        sign = torch.sign(dot)
        q_curr_adj = q_curr * sign
        
        # Calculate SLERP parameters with better numerical stability
        dot_abs = torch.abs(dot)
        omega = torch.acos(torch.clamp(dot_abs, 0, 0.9999))
        mask = omega < 1e-5
        sin_omega = torch.sin(omega)
        
        # Apply component influence if provided
        if comp_influence is not None:
            alpha = alpha.unsqueeze(1) * comp_influence.view(1, -1, 1)  # [B, C, V]
        else:
            alpha = alpha.unsqueeze(1)  # [B, 1, V]
        
        # SLERP weights with numerical stability
        k0 = torch.where(mask, 1-alpha, torch.sin((1-alpha)*omega)/torch.clamp(sin_omega, 1e-6))
        k1 = torch.where(mask, alpha, torch.sin(alpha*omega)/torch.clamp(sin_omega, 1e-6))
        
        # Ensure interpolation weights sum to 1 for better stability
        weight_sum = k0 + k1
        k0 = k0 / torch.clamp(weight_sum, min=1e-6)
        k1 = k1 / torch.clamp(weight_sum, min=1e-6)
        
        # Handle small angles case with linear interpolation
        small_angle_mask = sin_omega.abs() < 1e-6
        k0 = torch.where(small_angle_mask, 1 - alpha, k0)
        k1 = torch.where(small_angle_mask, alpha, k1)
        
        # SLERP quaternions
        q_filtered = k0 * q_prev + k1 * q_curr_adj
        
        # Ensure unit quaternion result
        q_filtered = F.normalize(q_filtered, p=2, dim=1)
        
        return q_filtered
        
    def reset_state(self, batch_size=None, device=None):
        """Reset filter state with proper device handling"""
        if device is None:
            device = self.prev_filtered.device
            
        if batch_size is None:
            batch_size = 1
            
        self.prev_filtered = torch.zeros(batch_size, 4, 7, device=device)
        self.prev_raw = torch.zeros(batch_size, 4, 7, device=device)
        self.velocity_history = torch.zeros(batch_size, 3, 7, device=device)

    
    
class ModelEncoder(nn.Module):
    """
    独立的编码器类，用于将四元数编码为高维特征
    """
    def __init__(self, n_in_enc=4, n_hid_enc=24, n_out_enc=24, graph_args_j={}, 
                edge_weighting=True, dropout=0.3, **kwargs):
        super().__init__()

        self.graph_j = Graph_J(**graph_args_j)
        A_j = torch.tensor(self.graph_j.A_j, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A_j", A_j)

         # 构造 relrec/relsend 矩阵，假设关节、部位和身体节点数分别为 7, 5, 3
        # self.build_relrec_relsend_for_joints(V=7)  # 类似的函数也在 Encoder 内部构造并注册

        
        self.qgcn = QGCN(n_in_enc, n_hid_enc, 1, dropout=dropout, residual=True)
        self.gcn = Gcn(n_hid_enc, n_out_enc, 1, dropout=dropout, residual=True)
        self.silu=  nn.SiLU()

        
        # 可学习的边权重 - 每层都有独立的权重
        if edge_weighting:
            # 每个邻接矩阵只有一组乘法和加法权重
            self.emul = nn.Parameter(torch.ones(self.A_j.size()))
            self.eadd = nn.Parameter(torch.zeros(self.A_j.size()))
        else:
            self.register_buffer('emul', torch.ones(self.A_j.size()))
            self.register_buffer('eadd', torch.zeros(self.A_j.size()))
    
    def forward(self, x):
        """
        参数:
        - x: 输入四元数 [B, 4, V]
        
        返回:
        - encoded: 编码后的特征 [B, n_out_enc, V]
        - skip_connections: 跳跃连接列表
        """

        # ---------- Stage-0  基础 GCN ----------
        x = self.qgcn(x, self.A_j*self.emul + self.eadd)   # (B,C,V=7)
        x= self.silu(x)  # 激活函数
        x=self.gcn(x, self.A_j*self.emul + self.eadd)  
        return x
    
    def build_relrec_relsend_for_joints(self, V):
        """
        示例：生成关节级别 relrec_s1, relsend_s1 => shape [E, V]
        E = V*(V-1) (有向边，不含自环) or 参考你实际在 code 里使用的 one-hot encode
        """
        off_diag = torch.ones(V, V) - torch.eye(V)
        # E.g. E = 42 for V=7 (7*6)
        # row idx => edge, col idx => node
        relrec = []
        relsend = []
        edge_idx = 0
        for i in range(V):
            for j in range(V):
                if i != j:
                    # edge: i->j
                    one_hot_rec = torch.zeros(V)
                    one_hot_send = torch.zeros(V)
                    one_hot_rec[j] = 1.
                    one_hot_send[i] = 1.
                    relrec.append(one_hot_rec)
                    relsend.append(one_hot_send)
                    edge_idx += 1
        relrec_s1 = torch.stack(relrec, dim=0)  # [E, V]
        relsend_s1 = torch.stack(relsend, dim=0)
        self.register_buffer('relrec_s1', relrec_s1)
        self.register_buffer('relsend_s1', relsend_s1)

    def build_relrec_relsend_for_parts(self, V):
        # 和上面同理, V=5
        off_diag = torch.ones(V, V) - torch.eye(V)
        relrec = []
        relsend = []
        for i in range(V):
            for j in range(V):
                if i != j:
                    one_hot_rec = torch.zeros(V)
                    one_hot_send = torch.zeros(V)
                    one_hot_rec[j] = 1.
                    one_hot_send[i] = 1.
                    relrec.append(one_hot_rec)
                    relsend.append(one_hot_send)
        relrec_s2 = torch.stack(relrec, dim=0)
        relsend_s2 = torch.stack(relsend, dim=0)
        self.register_buffer('relrec_s2', relrec_s2)
        self.register_buffer('relsend_s2', relsend_s2)

    def build_relrec_relsend_for_body(self, V):
        # 同理, V=3
        off_diag = torch.ones(V, V) - torch.eye(V)
        relrec = []
        relsend = []
        for i in range(V):
            for j in range(V):
                if i != j:
                    one_hot_rec = torch.zeros(V)
                    one_hot_send = torch.zeros(V)
                    one_hot_rec[j] = 1.
                    one_hot_send[i] = 1.
                    relrec.append(one_hot_rec)
                    relsend.append(one_hot_send)
        relrec_s3 = torch.stack(relrec, dim=0)
        relsend_s3 = torch.stack(relsend, dim=0)
        self.register_buffer('relrec_s3', relrec_s3)
        self.register_buffer('relsend_s3', relsend_s3)


class ModelDecoder(nn.Module):
    """
    独立的解码器类，用于将高维特征解码为四元数
    """
    def __init__(self, n_in_dec=32, n_hid_dec=24, 
                 graph_args_j={}, edge_weighting=True, dropout=0.3, **kwargs):
        super().__init__()
        
        # 构造并注册邻接矩阵
        self.graph_j = Graph_J(**graph_args_j)
        A_j = torch.tensor(self.graph_j.A_j, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A_j", A_j)
        
        self.norm    = nn.LayerNorm(n_in_dec)
        self.msg_in  = DecodeGcn(n_in_dec, n_in_dec, k_num=1)
 
        self.gcn   = QGCN(n_in_dec, n_hid_dec, 1, dropout=dropout, residual=True)
        
        # 可学习的边权重 - 每层都有独立的权重
        if edge_weighting:
            # 每个邻接矩阵只有一组乘法和加法权重
            self.emul = nn.Parameter(torch.ones(self.A_j.size()))
            self.eadd = nn.Parameter(torch.zeros(self.A_j.size()))
        else:
            self.register_buffer('emul', torch.ones(self.A_j.size()))
            self.register_buffer('eadd', torch.zeros(self.A_j.size()))
        
        # 四元数归一化层
        self.quat_norm = QuatNorm()
        self.head = nn.Sequential(
            nn.Linear(n_hid_dec, n_hid_dec*2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(n_hid_dec*2, 4),
        )
        self.dropout = nn.Dropout(dropout)
        self.skip_gate = nn.Parameter(torch.zeros(1)) 
    
    def forward(self, x):
        """
        参数:
        - x: 输入特征 [B, n_in_dec, V]
        - skip_connections: 跳跃连接列表 (可选)
        
        返回:
        - decoded: 解码后的四元数 [B, 4, V]
        """
       
        hidden = self.norm(x.permute(0,2,1)).permute(0,2,1)     # (B,C,V)
        res = hidden
        # 2) 消息传递
        x = self.msg_in(hidden, self.A_j*self.emul + self.eadd)    # (B,C,V)

        x=x+res
        x = self.gcn(x, self.A_j*self.emul + self.eadd)           # (B,C,V)
        x = self.dropout(x)

        x=x.permute(0,2,1)                                       # (B,V,C)
        pred = self.head(x)    
        pred = self.quat_norm(pred).permute(0,2,1)                            
        
        return pred

class Model(nn.Module):

        
    def __init__(self,
        n_in_enc=4,
        n_hid_dec=32,
        n_out_enc=64,  
        n_mid_temporal=48,  
        n_hid_enc = 32,
        graph_args_j={}, # 省略细节
        graph_args_p={}, # 省略细节
        graph_args_b={}, # 省略细节
        max_frames=5,   
        device='cuda',
        **kwargs):
        super().__init__()
        self.device = device
        self.max_frames = max_frames
        dropout= 0.3
        self.n_out_enc = n_out_enc

     
        self.model_encoder = ModelEncoder(
            n_in_enc=n_in_enc,
            n_hid_enc=n_hid_enc,
            n_out_enc=n_out_enc,
            graph_args_j=graph_args_j,
            edge_weighting=True,
            dropout=dropout,
            **kwargs
        )
        self.joint_dependency = HierarchicalJointModule(
            feature_dim=n_out_enc,
            num_joints=7
        )

        self.quat_filter = HighFrequencyQuaternionFilter(channels=4, num_joints=7, fps=30.0)
        self._init_filter_parameters()

        

        self.temporal_encoder = nn.GRU(
            input_size=n_out_enc,
            hidden_size=n_out_enc,
            num_layers=1,
            batch_first=True
        )
        
        # Attention for focusing on relevant temporal features
        self.temporal_attention = nn.Sequential(
            nn.Linear(n_out_enc, n_out_enc // 2),
            nn.SiLU(),
            nn.Linear(n_out_enc // 2, 1)
        )

                # 3. Feature projection after temporal processing
        self.temporal_projection = nn.Sequential(
            nn.Conv1d(n_out_enc, n_mid_temporal, 1),
            nn.BatchNorm1d(n_mid_temporal),
            nn.SiLU()
        )
        

        self.delta_scale = nn.Parameter(torch.ones(1) * 0.1)  # Initial small correction
        # 四元数归一化层
        self.quat_norm = QuatNorm()
        self.layer_norm = nn.LayerNorm(n_mid_temporal)

        self.model_decoder = ModelDecoder(
            n_in_dec=n_mid_temporal,
            n_hid_dec=n_hid_dec,
            n_out_dec=n_in_enc,
            graph_args_j=graph_args_j,
            edge_weighting=True,
            dropout=dropout,
            **kwargs
        )



    def _init_filter_parameters(self):
        cutoffs = torch.tensor([1.5, 1.0, 1.5, 0.6,1.0, 1.5, 0.6])  # Lower for better filtering
        betas = torch.tensor([0.02,0.02, 0.015, 0.012, 0.015, 0.012, 0.018])  # Higher adaptation
        strengths = torch.tensor([0.9,0.9, 0.85, 0.8, 0.85, 0.8, 0.87])  # Adaptive strength
        
        # Outlier detection thresholds
        outlier_thresholds = torch.tensor([4.0,4.0, 3.5, 3.0, 3.5, 3.0, 3.8])
        outlier_corrections = torch.tensor([0.95,0.95, 0.9, 0.85, 0.9, 0.85, 0.92])
        
        self.quat_filter.base_cutoff.data = cutoffs
        self.quat_filter.beta.data = betas
        self.quat_filter.adaptive_strength.data = strengths
        self.quat_filter.outlier_threshold.data = outlier_thresholds
        self.quat_filter.outlier_correction.data = outlier_corrections
        
        # Component influence - emphasize w component less for better filtering
        comp_weights = torch.tensor([0.8, 1.2, 1.2, 1.2])
        self.quat_filter.component_influence.data = comp_weights

    def forward(self, x):
        batch_size, T, quat_dim, num_joints = x.shape

        # Enhanced filtering with adaptive strength
        x_smooth = self.quat_filter(x)
        
        # Multi-scale temporal processing
        encoded_frames = []
        temporal_weights = torch.softmax(torch.linspace(0.5, 2.0, T, device=x.device), dim=0)
        
        for t in range(T):
            encoded = self.model_encoder(x_smooth[:, t])
            # Apply temporal weighting (recent frames get higher weights)
            encoded = encoded * temporal_weights[t]
            encoded_frames.append(encoded)

        # Stack and process with enhanced temporal modeling
        encoded_sequence = torch.stack(encoded_frames, dim=1)
        
        # Multi-head temporal attention for better sequence modeling
        temporal_features = []
        for v in range(num_joints):
            joint_seq = encoded_sequence[:, :, :, v]
            
            # Enhanced GRU with residual connections
            gru_out, hidden = self.temporal_encoder(joint_seq)
            
            # Multi-head attention for temporal dependencies
            attn_weights = self.temporal_attention(gru_out)
            attn_weights = F.softmax(attn_weights, dim=1)
            
            # Exponential weighting for recent frames
            recent_weights = torch.exp(torch.linspace(-2, 0, T, device=x.device)).unsqueeze(0).unsqueeze(-1)
            combined_weights = attn_weights * recent_weights
            combined_weights = combined_weights / combined_weights.sum(dim=1, keepdim=True)
            
            # Weighted temporal fusion
            joint_feature = torch.sum(gru_out * combined_weights, dim=1)
            temporal_features.append(joint_feature)
        
        temporal_features = torch.stack(temporal_features, dim=2)
        temporal_features = self.joint_dependency(temporal_features)
        temporal_features = self.temporal_projection(temporal_features)
        temporal_features = self.layer_norm(temporal_features.permute(0, 2, 1)).permute(0, 2, 1)
        
        # Enhanced residual prediction
        last_frame = x_smooth[:, -1]
        delta_q = self.model_decoder(temporal_features)
        
        # Apply residual with adaptive weighting
        residual_weight = torch.sigmoid(self.delta_scale)
        delta_q = delta_q * residual_weight
        
        denoised = quat_mul_batch(delta_q, last_frame)
        denoised = self.quat_norm(denoised.permute(0, 2, 1)).permute(0, 2, 1)
        
        return denoised
    
    def reset_filter_state(self, batch_size=None):
        """Reset filter state between sequences"""
        if hasattr(self, 'quat_filter'):
            device = next(self.parameters()).device
            self.quat_filter.reset_state(batch_size=batch_size, device=device)
    
    def motion_velocity(self, x):
        """Calculate motion velocity to adapt filtering strength"""
        if x.shape[1] < 2:
            return torch.zeros(x.shape[0], 1, x.shape[3], device=x.device)
        
        # Calculate frame-to-frame difference
        curr = x[:, -1]  # (B, C, V)
        prev = x[:, -2]  # (B, C, V)
        
        # Quaternion difference as proxy for motion speed
        diff = torch.abs(curr - prev).mean(dim=1, keepdim=True)  # (B, 1, V)
        
        return diff