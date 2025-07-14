# processor.py

import sys
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
from model import Model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import symmetrical_quaternion_loss,align_quaternion_signs,quaternion_log_loss,relative_quat,quat_geodesic_angle,quaternion_velocity_loss,compute_adduction_loss
import torch.nn.functional as F



def align_quat_signs(pred_q: np.ndarray, tgt_q: np.ndarray) -> np.ndarray:
    """
    pred_q, tgt_q: (..., 4) or (..., 4, J)
    把 pred_q 的符号调成和 tgt_q 同方向（点积为正）。
    """
    # 将 component 维提前，方便广播 (N,4,J) → (N,1,J)
    dot = np.sum(pred_q * tgt_q, axis=-2, keepdims=True)
    sign = np.where(dot < 0.0, -1.0, 1.0)
    return pred_q * sign


def geodesic_error_deg(pred_q: np.ndarray, tgt_q: np.ndarray) -> np.ndarray:
    """
    返回 shape (..., J) 的旋转测地角度误差（°）
    """
    dot = np.sum(pred_q * tgt_q, axis=-2)
    dot = np.clip(np.abs(dot), -1.0, 1.0)        # 取绝对值去掉 ±
    rad = 2.0 * np.arccos(dot)
    return np.degrees(rad)


def create_temporal_windows(data, window_size, boundaries):
    """
    从连续数据中创建时间窗口样本
    
    参数:
    - data: 原始数据 [N, D] (N个样本, D维特征)
    - window_size: 窗口大小 (帧数)
    - boundaries: 数据集边界列表，每个元素为(start, end)对
    
    返回:
    - windows: 时间窗口数据 [M, T, D] (M个窗口, T个时间步, D维特征)
    - targets: 对应的目标帧 [M, D]
    """
    windows = []
    targets = []
    
    for start, end in boundaries:
        # 确保序列长度足够
        if end - start + 1 >= window_size:
            # 对每个可能的窗口起点
            for i in range(start, end - window_size + 2):
                # 提取窗口
                window = data[i:i+window_size]
                
                # 获取中间帧作为目标
                mid_idx = window_size // 2
                target = data[i + mid_idx]
                
                windows.append(window)
                targets.append(target)
    
    if windows:
        windows = np.stack(windows)
        targets = np.stack(targets)
        return windows, targets
    
    return np.empty((0, window_size, data.shape[1])), np.empty((0, data.shape[1]))

def load_single_frame_data(
    base_dir="DataSet",
    exercise_types=["Squat"],
    sets=["set1", "set2", "set3"],
    test_ratio=0.15,
    val_ratio=0.15,
    random_seed=42,
    split_by_set=False
):
    """
    读取 Kinect & Xsens CSV，并将所有单帧拼接到一起；
    只保留 Kinect 需要的 7 个关节(各 4 维 quat => 28 维输入)。
    假设 Xsens 也提取相同关节(28 维)作为输出(可自行修改)。
    **修改说明：**
    1. 针对每个 set 按时序划分 train/val/test（先按时间顺序分割，不打乱顺序）。
    2. 对训练数据再做一次 shuffle 用于训练，但同时返回原始时序的训练数据用于后续绘图。
    
    返回: (shuffled_train, val, test, sequential_train)
    """
    # --- 保持原有代码：构造列名等 ---
    kinect_joints = [0,18,19,20,22,23,24]
    xsens_joint_names = [
        "Plevis",        
        "Left Upper Leg",   
        "Left Lower Leg",     
        "Left Foot",    
        "Right Upper Leg",  
        "Right Lower Leg",    
        "Right Foot"    
    ]
    kinect_joint_columns = []
    for j in kinect_joints:
        kinect_joint_columns.extend([
            f"Joint_{j}_quat_w",
            f"Joint_{j}_quat_x",
            f"Joint_{j}_quat_y",
            f"Joint_{j}_quat_z"
        ])
    xsens_joint_columns = []
    for name in xsens_joint_names:
        xsens_joint_columns.extend([
            f"{name}_quat_0",
            f"{name}_quat_1",
            f"{name}_quat_2",
            f"{name}_quat_3"
        ])

    pairs = [(ex, st) for ex in exercise_types for st in sets]
    if split_by_set:

        pairs_by_ex = {ex: [] for ex in exercise_types}
        for ex, st in pairs:
            pairs_by_ex[ex].append((ex, st))

        train_pairs, val_pairs, test_pairs = [], [], []
        for ex, ex_pairs in pairs_by_ex.items():
            # rng.shuffle(ex_pairs)
            n_total = len(ex_pairs)
            n_test = max(1, int(round(n_total * test_ratio)))
            n_val  = max(1, int(round(n_total * val_ratio)))
            # 保障留给 train 的至少 1 条
            if n_total - n_test - n_val < 1:
                # 把 val 压缩到 1，再保证 train≥1
                n_val  = max(1, n_total - n_test - 1)

            test_pairs += ex_pairs[:n_test]
            val_pairs  += ex_pairs[n_test : n_test + n_val]
            train_pairs+= ex_pairs[n_test + n_val :]
                
                
            def pair2set(pair_list):
                """把 (ex,set) 对转换成 'ex/set' 方便看"""
                return [f"{ex}/{st}" for ex, st in pair_list]

        print("======= Data split by set =======")
        print("Train sets:", pair2set(train_pairs))
        print("Val   sets:", pair2set(val_pairs))
        print("Test  sets:", pair2set(test_pairs))
        print("=================================")

        # 最终再整体打乱一次顺序（非必须，仅增加随机性）
        # rng.shuffle(train_pairs)
        # rng.shuffle(val_pairs)
        # rng.shuffle(test_pairs)
    else:
        test_pairs = val_pairs = train_pairs = None  # 占位

    # ---------- 2. 初始化收集列表 ----------
    train_k, val_k, test_k = [], [], []
    train_x, val_x, test_x = [], [], []
    train_l, val_l, test_l = [], [], []

    # ---------- 3. 遍历所有文件 ----------
    for ex, st in pairs:
        # 3‑a. 目标集合归属
        if split_by_set:
            if   (ex, st) in test_pairs:  target = "test"
            elif (ex, st) in val_pairs:   target = "val"
            else:                         target = "train"
        else:
            target = None  # 帧级切分稍后处理

        # 3‑b. 读取 CSV
        k_file = os.path.join(base_dir, ex, st, "slerped_fused_kinect_data.csv")
        x_file = os.path.join(base_dir, ex, st, "aligned_xsens_data_with_frame.csv")
        if not (os.path.exists(k_file) and os.path.exists(x_file)):
            print(f"[警告] 缺少文件，跳过: {k_file} / {x_file}")
            continue

        k_df = pd.read_csv(k_file)
        x_df = pd.read_csv(x_file)
        if len(k_df) != len(x_df):
            print(f"[错误] 行数不一致，跳过: {k_file} ({len(k_df)}) vs {x_file} ({len(x_df)})")
            raise ValueError(f"行数不一致: {k_file} vs {x_file}")

        k_data = k_df[kinect_joint_columns].values.astype(np.float32)
        x_data = x_df[xsens_joint_columns].values.astype(np.float32)
        labels = np.full(len(k_data), exercise_types.index(ex), np.int64)

        # 3‑c. 根据 split_by_set 选择加入方式
        if split_by_set:
            if target == "train":
                train_k.append(k_data); train_x.append(x_data); train_l.append(labels)
            elif target == "val":
                val_k.append(k_data);   val_x.append(x_data);   val_l.append(labels)
            else:  # test
                test_k.append(k_data);  test_x.append(x_data);  test_l.append(labels)
        else:
            n = len(k_data)
            n_test = int(n * test_ratio)
            n_val  = int(n * val_ratio)
            n_train= n - n_test - n_val
            train_slice = slice(0, n_train)
            val_slice   = slice(n_train, n_train + n_val)
            test_slice  = slice(n_train + n_val, n)

            train_k.append(k_data[train_slice]); train_x.append(x_data[train_slice]); train_l.append(labels[train_slice])
            val_k.append(k_data[val_slice]);     val_x.append(x_data[val_slice]);     val_l.append(labels[val_slice])
            test_k.append(k_data[test_slice]);   test_x.append(x_data[test_slice]);   test_l.append(labels[test_slice])

    # ---------- 4. 拼接 训练 ----------
    if not train_k:
        raise RuntimeError("未加载到任何训练数据，请检查路径或参数。")

    X_train_seq = np.concatenate(train_k, axis=0)
    Y_train_seq = np.concatenate(train_x, axis=0)
    L_train_seq = np.concatenate(train_l, axis=0)

    X_val   = np.concatenate(val_k,  axis=0) if val_k else np.empty((0, X_train_seq.shape[1]), np.float32)
    Y_val   = np.concatenate(val_x,  axis=0) if val_x else np.empty_like(X_val)
    L_val   = np.concatenate(val_l,  axis=0) if val_l else np.empty(0, np.int64)

    X_test  = np.concatenate(test_k, axis=0) if test_k else np.empty((0, X_train_seq.shape[1]), np.float32)
    Y_test  = np.concatenate(test_x, axis=0) if test_x else np.empty_like(X_test)
    L_test  = np.concatenate(test_l, axis=0) if test_l else np.empty(0, np.int64)

    # rng = np.random.RandomState(random_seed)
    # shuf_idx = rng.permutation(len(X_train_seq))
    shuf_idx =  np.arange(len(X_train_seq))
    X_train = X_train_seq[shuf_idx]
    Y_train = Y_train_seq[shuf_idx]
    L_train = L_train_seq[shuf_idx]


    return ((X_train, Y_train, L_train),
            (X_val,   Y_val,   L_val),
            (X_test,  Y_test,  L_test),
            (X_train_seq, Y_train_seq, L_train_seq))

def weights_init(m):
    """
    对卷积或 BN等层进行自定义初始化
    """
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def set_seed():
    """
    设置随机种子, 以便实验可复现。
    """
    seed = 3407
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def encode_onehot(indices):
    """
    将一系列索引转为 one-hot 矩阵
    """
    n = indices.max() + 1
    out = np.zeros((len(indices), n))
    for i, idx in enumerate(indices):
        out[i, idx] = 1
    return out

class Processor(object):
    """
    单帧预测示例的核心流程，去掉所有时间序列相关操作
    """

    def __init__(self,
                 base_dir="DataSet",
                 exercise_types=["Squat2-raw"],
                 sets=["set1","set2","set3","set4"],
                 split_by_set = True,
                 test_ratio=0.15,
                 val_ratio=0.15,
                 random_seed=42,
                 use_gpu=True,
                 device_id=0,
                 batch_size=32,
                 base_lr=1e-3,
                 weight_decay=1e-4,
                 num_epoch=20,
                 optimizer_type='Adam',
                 fusion_layer=0,
                 device='cuda',
                 n_in_enc=4,
                 n_hid_dec=32,
                 n_out_enc=64,  
                 n_mid_temporal=48,  
                 n_hid_enc = 32,
                 lamda_p=1,
                 frame_window=5, 
                 cross_w=0.4):
        set_seed()

        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.base_lr = base_lr
        self.weight_decay = weight_decay
        self.num_epoch = num_epoch
        self.optimizer_type = optimizer_type

        self.base_dir = base_dir
        self.exercise_types = exercise_types
        self.sets = sets
        self.test_ratio =test_ratio
        self.val_ratio = val_ratio
        self.random_seed = random_seed
        self.fusion_layer= fusion_layer
        self.cross_w = cross_w
        self.split_by_set = split_by_set
        self.device = device

        self.lamda_p = lamda_p

        self.frame_window = frame_window
        self.set_lengths = {}

        self.n_out_enc = n_out_enc
        self.n_hid_enc = n_hid_enc
        self.n_hid_dec = n_hid_dec
        self.n_in_enc =  n_in_enc
        self.n_mid_temporal = n_mid_temporal  # 中间 temporal 层的维度





        self.log_file = 'log.csv'   # 日志文件



        # 3) 构造模型并移动到 GPU/CPU
        self.load_model()
        self.gpu()


        self.load_data()

        # 5) 初始化优化器
        self.optimizer = optimizer_type
        self.load_optimizer(self.optimizer)

        self.loss_sym= symmetrical_quaternion_loss
        self.loss_log = quaternion_log_loss

        self.train_losses = []
        self.val_losses   = []

        self.patience = 15  # 多少轮不改善就停
        self.no_improve_count = 0


    def compute_set_boundaries(self):
        """计算正确的set边界"""
        self.train_boundaries = []
        self.val_boundaries = []
        self.test_boundaries = []
        
        # 使用与load_single_frame_data相同的顺序计算训练集边界
        start_idx = 0
        for ex, st in self.original_train_pairs:
            if (ex, st) in self.set_lengths:
                set_length = self.set_lengths[(ex, st)]
                if set_length > 0:
                    self.train_boundaries.append((start_idx, start_idx + set_length - 1))
                    start_idx += set_length
        
        # 计算验证集边界
        start_idx = 0
        for ex, st in self.original_val_pairs:
            if (ex, st) in self.set_lengths:
                set_length = self.set_lengths[(ex, st)]
                if set_length > 0:
                    self.val_boundaries.append((start_idx, start_idx + set_length - 1))
                    start_idx += set_length
        
        # 计算测试集边界
        start_idx = 0
        for ex, st in self.original_test_pairs:
            if (ex, st) in self.set_lengths:
                set_length = self.set_lengths[(ex, st)]
                if set_length > 0:
                    self.test_boundaries.append((start_idx, start_idx + set_length - 1))
                    start_idx += set_length
        
        print("=== Set Boundaries Information ===")
        print(f"Training set boundaries: {self.train_boundaries}")
        print(f"Validation set boundaries: {self.val_boundaries}")
        print(f"Testing set boundaries: {self.test_boundaries}")
        
        # # 验证边界计算的正确性
        # self.verify_boundaries()

    def verify_boundaries(self):
        """验证边界计算的正确性"""
        # 1. 验证训练集总长度
        total_train_frames = sum(end - start + 1 for start, end in self.train_boundaries)
        if total_train_frames != len(self.train_x):
            print(f"WARNING: Training set boundary total ({total_train_frames}) does not match actual data length ({len(self.train_x)})")
        else:
            print(f"✓ Training set boundary total matches data length: {total_train_frames}")
        
        # 2. 验证验证集总长度
        total_val_frames = sum(end - start + 1 for start, end in self.val_boundaries)
        if total_val_frames != len(self.val_x):
            print(f"WARNING: Validation set boundary total ({total_val_frames}) does not match actual data length ({len(self.val_x)})")
        else:
            print(f"✓ Validation set boundary total matches data length: {total_val_frames}")
        
        # 3. 验证测试集总长度
        total_test_frames = sum(end - start + 1 for start, end in self.test_boundaries)
        if total_test_frames != len(self.test_x):
            print(f"WARNING: Test set boundary total ({total_test_frames}) does not match actual data length ({len(self.test_x)})")
        else:
            print(f"✓ Test set boundary total matches data length: {total_test_frames}")
        
        # 4. 验证边界不重叠且连续
        for dataset_name, boundaries in [("Training", self.train_boundaries), 
                                        ("Validation", self.val_boundaries), 
                                        ("Testing", self.test_boundaries)]:
            if not boundaries:
                continue
                
            prev_end = -1
            for i, (start, end) in enumerate(boundaries):
                # 检查边界的有效性
                if start > end:
                    print(f"ERROR: Invalid boundary in {dataset_name} set: ({start}, {end})")
                
                # 检查边界的连续性
                if i > 0 and start != prev_end + 1:
                    print(f"WARNING: Non-contiguous boundaries in {dataset_name} set: {boundaries[i-1]} -> {boundaries[i]}")
                
                prev_end = end
            
    def load_model(self):
        
        self.model = Model(
        n_in_enc=self.n_in_enc,
        n_hid_dec=self.n_hid_dec,
        n_out_enc=self.n_out_enc,
        n_hid_enc=self.n_hid_enc,
        n_mid_temporal=self.n_mid_temporal,

        graph_args_j={'strategy': 'uniform','max_hop': 1,'dilation': 1},
        fusion_layer=self.fusion_layer,
        cross_w=self.cross_w,
        )
        self.model.apply(weights_init)



        print("Model created, with adjacency info for joints/parts/body")

    
    def load_weights(self):
        """
        如果指定了权重文件，则加载它
        """
        if self.arg.weights is not None:
            ckpt = torch.load(self.arg.weights, map_location='cpu')
            self.model.load_state_dict(ckpt)
            print(f'Loaded weights from {self.arg.weights}')

    def gpu(self):
        """
        将模型移动到 GPU 或 CPU
        """
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.model.to(self.device)    
            # self.model.cuda(self.device)
            print(f"Use GPU: cuda:{self.device}")
        else:
            self.device = torch.device("cpu")
            self.model.to(self.device)    
            print("Use CPU.")


    def load_data(self):
        """加载数据并构建时间窗口"""
        # 首先加载原始数据
        (self.train_x, self.train_y, self.train_label), (self.val_x, self.val_y, self.val_label), (self.test_x, self.test_y, self.test_label), _ = load_single_frame_data(
            self.base_dir, self.exercise_types, self.sets,
            self.test_ratio, self.val_ratio, self.random_seed, self.split_by_set
        )
        
        # 处理set边界
        self.preprocess_set_lengths()
        self.compute_set_boundaries()
        
            # 创建时间窗口序列 - 输入是多帧，目标是对应的ground truth
        # 修改 create_windows_with_gt 函数，让目标帧为当前帧（t帧）而非窗口中间帧
        def create_windows_with_gt(input_data, gt_data, boundaries, window_size):
            """
            创建时间窗口，使用ground truth作为目标
            
            参数:
            - input_data: 输入数据 (Kinect数据)
            - gt_data: ground truth数据 (Xsens数据)
            - boundaries: 数据集边界
            - window_size: 窗口大小
            
            返回:
            - windows: 窗口数据 [M, T, D]，其中T是窗口大小
            - targets: 目标数据 [M, D]，对应当前帧(t帧)的ground truth
            """
            windows = []
            targets = []
            
            for start, end in boundaries:
                # 确保序列长度足够
                if end - start + 1 >= window_size:
                    # 窗口起点从 start 到 end-window_size+1
                    for i in range(start, end - window_size + 2):
                        # 提取输入窗口，包含 t-(window_size-1) 到 t 帧
                        window = input_data[i:i+window_size]
                        
                        # 重要修改：目标帧是窗口中的最后一帧(t帧)对应的ground truth
                        # 窗口排序为 [t-(window_size-1), ..., t-2, t-1, t]
                        target_idx = i + window_size - 1  # t帧索引
                        target = gt_data[target_idx]
                        
                        windows.append(window)
                        targets.append(target)
            
            if windows:
                return np.array(windows), np.array(targets)
            return np.empty((0, window_size, input_data.shape[1])), np.empty((0, input_data.shape[1]))
        
        # 为训练、验证和测试集创建窗口，使用正确的ground truth
        self.train_windows, self.train_targets = create_windows_with_gt(
            self.train_x, self.train_y, self.train_boundaries, self.frame_window
        )
        self.val_windows, self.val_targets = create_windows_with_gt(
            self.val_x, self.val_y, self.val_boundaries, self.frame_window
        )
        self.test_windows, self.test_targets = create_windows_with_gt(
            self.test_x, self.test_y, self.test_boundaries, self.frame_window
        )
        
        # 保存原始目标用于比较
        self.train_y_orig = self.train_y
        self.val_y_orig = self.val_y
        self.test_y_orig = self.test_y
        
        # 重塑数据以适合模型输入 [B, T, 4, V]
        def reshape_data(windows, targets):
            B, T, C = windows.shape
            V = 7  # 关节数
            
            # 将窗口重塑为 [B, T, V, 4]
            reshaped_windows = windows.reshape(B, T, V, 4)
            # 转置为 [B, T, 4, V]
            reshaped_windows = reshaped_windows.transpose(0, 1, 3, 2)
            
            # 将目标重塑为 [B, V, 4]
            reshaped_targets = targets.reshape(B, V, 4)
            # 转置为 [B, 4, V]
            reshaped_targets = reshaped_targets.transpose(0, 2, 1)
            
            return reshaped_windows, reshaped_targets
        
        self.train_windows, self.train_targets = reshape_data(self.train_windows, self.train_targets)
        self.val_windows, self.val_targets = reshape_data(self.val_windows, self.val_targets)
        self.test_windows, self.test_targets = reshape_data(self.test_windows, self.test_targets)
        
        print(f"Data loaded and shaped: {self.train_windows.shape[0]} training samples, "
            f"{self.val_windows.shape[0]} validation samples, "
            f"{self.test_windows.shape[0]} test samples")
        
        # 验证数据形状
        print(f"Input window shape: {self.train_windows.shape}")  # [B, T, 4, V]
        print(f"Target shape: {self.train_targets.shape}")        # [B, 4, V]
        
        # 验证目标数据确实是ground truth而不是输入数据
        if len(self.train_windows) > 0:
            # 比较第一个样本的输入最后一帧和目标
            sample_input_last = self.train_windows[0, -1]  # 输入窗口的最后一帧
            sample_target = self.train_targets[0]           # 对应的目标
            
            # 计算差异
            diff = np.mean(np.abs(sample_input_last - sample_target))
            print(f"Sample difference between input last frame and target: {diff:.6f}")
            
            if diff < 1e-6:
                print("WARNING: Target and input appear to be identical! Check ground truth data.")
            else:
                print("✓ Target and input are different, ground truth is being used correctly.")

    def preprocess_set_lengths(self):
        """预处理所有set的长度信息"""
        # 遍历所有可能的set组合
        pairs = [(ex, st) for ex in self.exercise_types for st in self.sets]
        
        # 首先确定每个set的分配（训练/验证/测试）
        # 这部分逻辑必须与load_single_frame_data中的分配逻辑完全一致
        pairs_by_ex = {ex: [] for ex in self.exercise_types}
        for ex, st in pairs:
            pairs_by_ex[ex].append((ex, st))

        train_pairs, val_pairs, test_pairs = [], [], []
        for ex, ex_pairs in pairs_by_ex.items():
            n_total = len(ex_pairs)
            n_test = max(1, int(round(n_total * self.test_ratio)))
            n_val = max(1, int(round(n_total * self.val_ratio)))
            if n_total - n_test - n_val < 1:
                n_val = max(1, n_total - n_test - 1)

            test_pairs += ex_pairs[:n_test]
            val_pairs += ex_pairs[n_test : n_test + n_val]
            train_pairs += ex_pairs[n_test + n_val :]
        
        self.original_train_pairs = train_pairs
        self.original_val_pairs = val_pairs
        self.original_test_pairs = test_pairs
        
        print("Training sets:", self.original_train_pairs)
        print("Validation sets:", self.original_val_pairs)
        print("Testing sets:", self.original_test_pairs)
        
        # 读取每个set的数据长度
        for ex, st in pairs:
            k_file = os.path.join(self.base_dir, ex, st, "slerped_fused_kinect_data.csv")
            if not os.path.exists(k_file):
                print(f"Warning: File not found: {k_file}")
                continue
            
            # 读取CSV文件并获取帧数
            try:
                k_df = pd.read_csv(k_file)
                self.set_lengths[(ex, st)] = len(k_df)
                print(f"Set {ex}/{st} has {len(k_df)} frames")
            except Exception as e:
                print(f"Error reading {k_file}: {e}")
                self.set_lengths[(ex, st)] = 0

            
    def load_optimizer(self,optimizer):
        if optimizer == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.arg.base_lr,
                                       momentum=0.9,
                                       weight_decay=self.arg.weight_decay)
        elif optimizer == 'Adam':
            decay, no_decay = [], []
            for n,p in self.model.named_parameters():
                if 'bias' in n or 'bn' in n.lower() or 'QuatLinear' in n:
                    no_decay.append(p)
                else:
                    decay.append(p)
            # self.optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
            self.optimizer = optim.AdamW(
                    [{'params':decay,    'weight_decay':5e-4},
                    {'params':no_decay, 'weight_decay':0. }],
                    lr=self.base_lr, betas=(0.9,0.999))
        else:
            raise ValueError('No such Optimizer')
        # self.scheduler = StepLR(
        #     self.optimizer,
        #     step_size=5,   # 例如每 10 epoch 学习率衰减一次
        #     gamma=0.5       # 衰减系数，可按需调整
        # )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #            T_max=self.num_epoch, eta_min=1e-5)
        


            # Calculate steps for scheduler
        train_samples = len(self.train_windows)
        steps_per_epoch = math.ceil(train_samples / self.batch_size)
        total_steps = steps_per_epoch * self.num_epoch

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
        self.optimizer,
        max_lr=self.base_lr,  # Use your base learning rate
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        div_factor=20,  # initial_lr = max_lr/25
        final_div_factor=100,  # final_lr = initial_lr/1000
        anneal_strategy='cos'
        )

        print("Scheduler created: StepLR with step_size=10, gamma=0.5")


    def adjust_lr(self):
        """
        对应 recognition.py 的 adjust_lr
        """
        if self.arg.optimizer == 'SGD' and len(self.arg.step) > 0:
            # 例如每到达 step 里某个iter，就衰减 lr
            # 这里把 self.meta_info['iter'] 换成内置计数 iteration
            # 也可以每个 epoch 用 epoch 逻辑
            iteration = self.meta_info['iter'] if hasattr(self, 'meta_info') else 0
            # 统计 iteration >= step 的个数
            n = np.sum(iteration >= np.array(self.arg.step))
            new_lr = self.arg.base_lr * (0.5 ** n)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.lr = new_lr

        elif self.arg.optimizer == 'Adam' and len(self.arg.step) > 0:
            iteration = self.meta_info['iter'] if hasattr(self, 'meta_info') else 0
            n = np.sum(iteration >= np.array(self.arg.step))
            new_lr = self.arg.base_lr * (0.98 ** n)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
            self.lr = new_lr

        else:
            # 不做衰减
            self.lr = self.arg.base_lr

    def compute_loss(self, pred, target, window_data=None):
        """
        计算综合损失函数
        
        参数:
        - pred: 预测的四元数 [B, 4, V]
        - target: 目标四元数 [B, 4, V]
        - window_data: 输入窗口数据 [B, T, 4, V] (可选)
        
        返回:
        - 总损失值
        - 各损失分量的字典
        """
        SKELETON_PAIRS = [     # parent -> child
            (0,1), (1,2), (2,3),    # left leg
            (0,4), (4,5), (5,6)     # right leg
        ]

        
        # 1. 四元数重建损失 - 基本的重建目标
        quat_loss  = self.loss_sym(pred, target)
        
        # 2. 四元数对数损失 - 捕获旋转空间误差
        log_loss = self.loss_log(pred, target)

        adduction_loss_val = 0.02*compute_adduction_loss(pred, target)
        
        # 3. 关节间结构一致性损失
        joint_consistency_loss = 0.0
        for (parent, child) in SKELETON_PAIRS:
            # 计算父子关节间的相对旋转
            rel_pred = relative_quat(pred[:, :, child], pred[:, :, parent])
            rel_target = relative_quat(target[:, :, child], target[:, :, parent]) 
            
            pair_loss = self.loss_log(rel_pred, rel_target)
            joint_consistency_loss += pair_loss
        
        joint_consistency_loss /= len(SKELETON_PAIRS)


        ankle_loss = 0.0
        ankle_indices = [3, 6]  # Left and right foot
        for ankle_idx in ankle_indices:
            ankle_loss += 0.2 *self.loss_sym(pred[:, :, ankle_idx], target[:, :, ankle_idx])
    
        
        # 4. 时间平滑损失 - 当前帧与前一帧之间的平滑性
    
        diff1_Loss = diff2_Loss = 0.0
        if window_data is not None and window_data.shape[1] >= 3:
            prev_frame = window_data[:, -2]      # t-1
            prev2_frame = window_data[:, -3]     # t-2
            
            # Directly use tensor operations instead of scalar functions
            # Calculate velocities for each joint (first derivative)
            pred_vel = quat_geodesic_angle(pred, prev_frame)  # [B, V]
            target_vel = quat_geodesic_angle(target, prev_frame)  # [B, V]
            
            # Calculate accelerations (second derivative)
            prev_vel = quat_geodesic_angle(prev_frame, prev2_frame)  # [B, V]
            pred_accel = pred_vel - prev_vel  # [B, V]
            target_accel = target_vel - prev_vel  # [B, V]
            
            # Compute MSE for velocity and acceleration
            diff1_Loss = F.mse_loss(pred_vel, target_vel)
            diff2_Loss = F.mse_loss(pred_accel, target_accel)
            
            diff_loss = diff1_Loss + 2.0 * diff2_Loss


        if window_data is not None and window_data.shape[1] > 1:

            prev_frame = window_data[:, -2]  # t-1 frame
        
            # Calculate geodesic angle between current and previous frame
            angle_diff = quaternion_velocity_loss(pred, prev_frame)  # [B, V]
            
            # Encourage small angular changes
            smoothness_loss = 0.1*angle_diff.pow(2).mean()

            
        
        # 总损失计算
        # 重要修改：调整各损失分量权重
        total_loss = quat_loss + 1.5*log_loss+0.02*joint_consistency_loss+0.01*adduction_loss_val
        
        # # 加入平滑性损失，权重适中
        if isinstance(smoothness_loss, torch.Tensor) and smoothness_loss > 0:
            total_loss += 0.7 * smoothness_loss
        
        # 记录各损失分量
        losses = {
            'quat': quat_loss.item(),
            'log': log_loss.item(),
            'joint': joint_consistency_loss.item(),
            'smooth': smoothness_loss.item() if isinstance(smoothness_loss, torch.Tensor) else 0
        }
        
        return total_loss, losses
    

    def train_one_epoch(self, epoch):
        """
        单帧训练: 每个 epoch 随机打乱训练数据 => 分batch => forward/backward
        """
        # torch.autograd.set_detect_anomaly(True)
        self.model.train()

        # for param in self.model.quat_filter.parameters():
        #             param.requires_grad = False

        if epoch < 4:  # First 10 epochs
            for param in self.model.quat_filter.parameters():
                    param.requires_grad = False
        else:
            for param in self.model.quat_filter.parameters():
                    param.requires_grad = True

        num_samples = len(self.train_windows)
        num_batch = math.ceil(num_samples / self.batch_size)
        indices = np.random.permutation(num_samples)
        # indices = np.random.permutation(len(self.train_x))
        # num_batch = len(self.train_x) // self.batch_size
        epoch_loss = 0.0
        loss_log = 0.0
        loss_sym = 0.0
        pred_prev = None
        num_samples = len(self.train_windows)
        loss_components = {'quat': 0.0, 'log': 0.0, 'smooth': 0.0, 'joint': 0.0}
        
        for i in range(num_batch):
            self.model.reset_filter_state(batch_size=self.batch_size)

            start = i * self.batch_size
            end   = min(start + self.batch_size, num_samples)
            # batch_indices = indices[i * self.batch_size:(i + 1) * self.batch_size]
            batch_indices = indices[start:end]


            batch_windows = torch.from_numpy(
                self.train_windows[batch_indices]
            ).float().to(self.device)
            
            batch_targets = torch.from_numpy(
                self.train_targets[batch_indices]
            ).float().to(self.device)

            denoised = self.model(batch_windows)

            



            
            loss, components = self.compute_loss(denoised, batch_targets, batch_windows)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            epoch_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v
           

        avg_loss = epoch_loss / num_batch
        avg_components = {k: v / num_batch for k, v in loss_components.items()}


        print(f"[Train] Epoch {epoch}, Loss: {avg_loss:.6f}, "
              f"Quat: {avg_components['quat']:.6f}, "
              f"Log: {avg_components['log']:.6f}, "
              f"Joint: {avg_components['joint']:.6f}, "
              f"Smooth: {avg_components.get('smooth', 0):.6f}")
        return avg_loss

    def val_one_epoch(self, epoch):
        """
        验证集上的评估
        """
        self.model.eval()
        indices = np.arange(len(self.val_x)-1)
        # 创建有效的帧对列表
        num_samples = len(self.val_windows)
        
        # 计算批次数
        num_batch = num_samples // self.batch_size
        total_loss = 0.0
        loss_log = 0.0
        loss_sym = 0.0
        pred_prev = None  # 用于平滑损失计算
        loss_components = {'quat': 0.0, 'log': 0.0, 'smooth': 0.0, 'joint': 0.0}
        num_samples = len(self.val_windows)

        with torch.no_grad():
            for i in range(num_batch):
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_samples)
                self.model.reset_filter_state(batch_size=self.batch_size)
                
                
                # 获取批次数据
                batch_windows = torch.from_numpy(
                    self.val_windows[start_idx:end_idx]
                ).float().to(self.device)
                
                batch_targets = torch.from_numpy(
                    self.val_targets[start_idx:end_idx]
                ).float().to(self.device)
                
                # 模型预测
                denoised = self.model(batch_windows)
                
                # 计算损失
                loss, components = self.compute_loss(denoised, batch_targets, batch_windows)
                
                total_loss += loss.item()
                for k, v in components.items():
                    loss_components[k] += v
        
        # 计算平均损失
        avg_loss = total_loss / num_batch
        avg_components = {k: v / num_batch for k, v in loss_components.items()}
        
        
        print(f"[Val] Epoch {epoch}, Loss: {avg_loss:.6f}, "
              f"Quat: {avg_components['quat']:.6f}, "
              f"Log: {avg_components['log']:.6f}, "
              f"Joint: {avg_components['joint']:.6f}, "
              f"Smooth: {avg_components.get('smooth', 0):.6f}")
        
        return avg_loss

    def test(self):
        """
        全部epoch结束后对测试集做评估
        """
        self.model.eval()
        # indices = np.arange(len(self.test_x))
        # num_batch = len(self.test_x) // self.batch_size
        valid_frame_pairs = []
        all_preds = []
        all_targets = []
        
        # 计算批次数
        num_samples = len(self.test_windows)
        num_batch = max(1, num_samples // self.batch_size)
        
        total_loss = 0.0
        loss_log = 0.0
        loss_sym = 0.0

        loss_components = {'quat': 0.0, 'log': 0.0, 'smooth': 0.0, 'joint': 0.0}

        with torch.no_grad():
            for i in range(num_batch):
                # 获取当前批次
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, num_samples)
                
                # 获取批次数据
                batch_windows = torch.from_numpy(
                    self.test_windows[start_idx:end_idx]
                ).float().to(self.device)
                
                batch_targets = torch.from_numpy(
                    self.test_targets[start_idx:end_idx]
                ).float().to(self.device)
                
                # 模型预测
                denoised = self.model(batch_windows)
                
                # 计算损失
                loss, components = self.compute_loss(denoised, batch_targets, batch_windows)
                
                total_loss += loss.item()
                for k, v in components.items():
                    loss_components[k] += v
                
                # 存储预测和目标
                all_preds.append(denoised.cpu().numpy())
                all_targets.append(batch_targets.cpu().numpy())
        
        # 计算平均损失
        avg_loss = total_loss / num_batch
        avg_components = {k: v / num_batch for k, v in loss_components.items()}
        
        
        print(f"[Test] , Loss: {avg_loss:.6f}, "
              f"Quat: {avg_components['quat']:.6f}, "
              f"Log: {avg_components['log']:.6f}, "
              f"Joint: {avg_components['joint']:.6f}, "
              f"Smooth: {avg_components.get('smooth', 0):.6f}")
        
        # 合并所有预测和目标
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # 可视化测试结果
        self.visualize_all_results()
        
        return avg_loss
    
    def run(self):
        """
        全 epoch: Train + Val + (最后Test)
        """
        with open('log.csv', 'w') as f:
         f.write("epoch,train_loss,val_loss\n")  # 表头

        best_val_loss = float('inf')
        for epoch in range(1, self.num_epoch+1):
            train_loss = self.train_one_epoch(epoch)
            val_loss  = self.val_one_epoch(epoch)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            with open('log.csv', 'a') as f:
                f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f}\n")
            self.scheduler.step()


            
        # Early Stopping logic
            if epoch %10 == 0:
                        self.save_model(f"model_{epoch}.pt")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.no_improve_count = 0
                # 可保存最优模型
                torch.save(self.model.state_dict(), "best_model.pt")

            else:
                self.no_improve_count += 1
                print(f"No improvement for {self.no_improve_count} epoch(s).")

            if self.no_improve_count >= self.patience:
                print("Early stopping triggered!")
                break


        # epoch全跑完 => test
        # 所有 epoch 完成后，绘制整体的 loss 曲线
        self.plot_final_loss_curve(self.train_losses, self.val_losses)
        self.test()
        # self.plot_all_quaternion_sequences()

    def save_model(self, filename):
        torch.save(self.model.state_dict(), filename)
        print(f"Model saved: {filename}")

    def plot_final_loss_curve(self, train_losses, val_losses):
        plt.figure(figsize=(8,6))
        epochs = range(1, len(train_losses)+1)
        plt.plot(epochs, train_losses, label="Train Loss")
        plt.plot(epochs, val_losses, label="Validation Loss")
        plt.title("Train and Validation Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("final_loss_curve.png")
        plt.close()
        print("Final loss curve saved as final_loss_curve.png")

    def visualize_all_results(self, results_dir="visualization_results"):
        """
        可视化所有数据集的预测结果
        """
        os.makedirs(results_dir, exist_ok=True)
        
        # 处理每个数据集
        datasets = [
            ("train", self.train_windows, self.train_targets),
            ("val", self.val_windows, self.val_targets),
            ("test", self.test_windows, self.test_targets)
        ]
        
        for dataset_name, windows, targets in datasets:
            print(f"Generating visualizations for {dataset_name} dataset...")
            
            # 分批处理以避免内存问题
            batch_size = 64
            num_samples = len(windows)
            all_preds = []
            all_inputs = []  # 存储当前帧(t帧)的输入数据
            
            self.model.eval()
            with torch.no_grad():
                for i in range(0, num_samples, batch_size):
                    end_idx = min(i + batch_size, num_samples)
                    
                    # 获取批次数据
                    batch_windows = torch.from_numpy(
                        windows[i:end_idx]
                    ).float().to(self.device)
                    
                    # 存储当前帧(t帧)的输入数据
                    input_current = batch_windows[:, -1].cpu().numpy()  # t帧
                    all_inputs.append(input_current)
                    
                    # 模型预测
                    preds = self.model(batch_windows)
                    all_preds.append(preds.cpu().numpy())
            
            # 合并所有预测和输入
            all_preds = np.concatenate(all_preds, axis=0)
            all_inputs = np.concatenate(all_inputs, axis=0)
            
            # 绘制每个关节的时间序列
            components = ['w', 'x', 'y', 'z']
            
            for j in range(7):  # 7个关节
                fig, axes = plt.subplots(2, 2, figsize=(15, 15))
                axes = axes.flatten()
                
                for c, comp in enumerate(components):
                    ax = axes[c]
                    
                    # 提取当前关节和分量的预测值、目标值和输入值
                    pred_values = all_preds[:, c, j]
                    target_values = targets[:, c, j]
                    input_values = all_inputs[:, c, j]
                    
                    # 创建时间索引
                    time_indices = np.arange(len(pred_values))
                    
                    # 绘制预测值、目标值和输入值
                    ax.plot(time_indices, pred_values, 'blue', alpha=0.7, label='pred')
                    ax.plot(time_indices, target_values, 'orange', alpha=0.7, label='target')
                    ax.plot(time_indices, input_values, 'green', alpha=0.5, label='input')
                    
                    # 计算误差
                    error = np.abs(pred_values - target_values)
                    mean_error = np.mean(error)
                    input_error = np.mean(np.abs(input_values - target_values))
                    
                    ax.set_title(f"component {comp} (Pred Error: {mean_error:.4f}, Input Error: {input_error:.4f})")
                    ax.set_xlabel("frames")
                    ax.set_ylabel("Unit")
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                
                plt.suptitle(f"{dataset_name.capitalize()} dataset - joint {j} quaternions")
                plt.tight_layout()
                plt.savefig(f"{results_dir}/{dataset_name}_joint{j}_quaternions.png")
                plt.close()
                
            # 创建误差汇总热图
            plt.figure(figsize=(15, 10))
            
            # 计算三种误差：预测vs目标、输入vs目标、预测vs输入
            error_types = ["pred VS. target", "input VS, target", "pred VS. input"]
            error_data = np.zeros((7, 4, 3))  # (关节数, 分量数, 误差类型数)
            
            for j in range(7):
                for c in range(4):
                    error_data[j, c, 0] = np.mean(np.abs(all_preds[:, c, j] - targets[:, c, j]))
                    error_data[j, c, 1] = np.mean(np.abs(all_inputs[:, c, j] - targets[:, c, j]))
                    error_data[j, c, 2] = np.mean(np.abs(all_preds[:, c, j] - all_inputs[:, c, j]))
            
            # 创建三个子图
            fig, axes = plt.subplots(1, 3, figsize=(21, 7))
            
            for e, error_type in enumerate(error_types):
                im = axes[e].imshow(error_data[:, :, e], cmap='hot', aspect='auto')
                axes[e].set_title(f"{error_type} error")
                axes[e].set_xlabel('quaternion component')
                axes[e].set_ylabel('joint index')
                axes[e].set_xticks(np.arange(4))
                axes[e].set_xticklabels(components)
                axes[e].set_yticks(np.arange(7))
                axes[e].set_yticklabels([f"joint {j}" for j in range(7)])
                
                # 添加误差数值
                for j in range(7):
                    for c in range(4):
                        text_color = "w" if error_data[j, c, e] > 0.2 else "k"
                        axes[e].text(c, j, f"{error_data[j, c, e]:.4f}", 
                                ha="center", va="center", color=text_color)
                
                plt.colorbar(im, ax=axes[e], label='menean abs error')
            
            plt.suptitle(f"{dataset_name.capitalize()}  - error analysis")
            plt.tight_layout()
            plt.savefig(f"{results_dir}/{dataset_name}_error_analysis.png")
            plt.close()
            
            # 计算整体去噪效果
            pred_error = np.mean(np.abs(all_preds - targets))
            input_error = np.mean(np.abs(all_inputs - targets))
            improvement = (input_error - pred_error) / input_error * 100
            
            print(f"{dataset_name} :")
            print(f"  input error: {input_error:.6f}")
            print(f"  pred error: {pred_error:.6f}")
            print(f"  improved: {improvement:.2f}%")
            
            print(f"Completed visualization for {dataset_name} dataset")


    def plot_all_quaternion_sequences(self, results_dir="results_all"):
        """
        针对双帧模型的可视化函数：对于 train、val、test，每个阶段全部数据（各 set 数据按原始顺序拼接）都画出来，
        每个关节生成一张2x2的图，每个子图对应一个四元数分量（w,x,y,z），
        在同一子图中绘制预测值、真实值、输入值以及隐藏层状态（L2范数）的曲线。
        """
        os.makedirs(results_dir, exist_ok=True)
        phases = {
            # 使用时序数据 self.train_x_seq 而非打乱后的 self.train_x
            "train": (self.train_x_seq, self.train_y_seq),
            "val":   (self.val_x, self.val_y),
            "test":  (self.test_x, self.test_y)
        }
        
        for phase, (xdata, ydata) in phases.items():
            # 跳过第一帧，因为我们需要前一帧作为输入
            num_frames = len(xdata) - 1
            
            if num_frames <= 0:
                print(f"Warning: Not enough frames in {phase} phase to make predictions")
                continue
                
            # 初始化存储预测结果和隐藏状态的数组
            pred_all = []
            hidden_all = []
            
            # 使用小批量处理，避免内存溢出
            batch_size = 64  # 可以根据GPU内存调整
            for i in range(0, num_frames, batch_size):
                end_idx = min(i + batch_size, num_frames)
                
                # 获取当前批次的前一帧和当前帧
                bx_prev = torch.from_numpy(xdata[i:end_idx]).to(self.device).view(-1, 7, 4).transpose(1, 2)
                bx_curr = torch.from_numpy(xdata[i+1:end_idx+1]).to(self.device).view(-1, 7, 4).transpose(1, 2)
                
                with torch.no_grad():
                    # 运行模型，获取预测和隐藏状态
                    pred_batch, hidden_batch = self.model(bx_prev, bx_curr)
                    
                    # 收集结果
                    pred_all.append(pred_batch.cpu())
                    hidden_all.append(hidden_batch.cpu())
            
            # 拼接所有批次的结果
            pred = torch.cat(pred_all, dim=0)
            hidden = torch.cat(hidden_all, dim=0)
            
            # 准备绘图数据
            by = torch.from_numpy(ydata[1:num_frames+1]).to(self.device).view(-1, 7, 4).transpose(1, 2)  # 跳过第一帧的真实值
            bx_plot = torch.from_numpy(xdata[1:num_frames+1]).to(self.device).view(-1, 7, 4)  # 跳过第一帧的输入值
            
            # 对齐四元数符号
            pred = align_quaternion_signs(pred, by)
            
            # 转置为 (num_frames, 7, 4) 用于绘图
            if pred.shape[1] == 4:
                pred = pred.transpose(1, 2)
            
            # 计算隐藏状态的范数
            hidden_trans = hidden.transpose(1, 2)  # [batch, hid, joints] -> [batch, joints, hid]
            T, J, C = hidden_trans.shape
            
            # 检查隐藏层维度是否为4的倍数
            if C % 4 == 0:
                hidden_q = hidden_trans.view(T, J, 4, C // 4)   # (T,7,4,q_ch)
                hidden_norm = hidden_q.norm(dim=-1)             # (T,7,4)
            else:
                # 如果不是4的倍数，直接计算每个关节的隐藏状态范数
                hidden_norm = hidden_trans.norm(dim=-1).unsqueeze(-1).repeat(1, 1, 4)  # (T,7,4)
            
            # 绘制每个关节的四元数分量
            for j in range(7):
                # 提取每个关节的数据
                pred_joint = pred[:, j, :].cpu().numpy()      # shape: (num_frames, 4)
                true_joint = by.transpose(1, 2)[:, j, :].cpu().numpy()  # shape: (num_frames, 4)
                input_joint = bx_plot[:, j, :].cpu().numpy()  # shape: (num_frames, 4)
                
                # 生成单个关节的图，2x2子图分别对应w,x,y,z
                fig = plt.figure(figsize=(12, 10))
                components = ['w', 'x', 'y', 'z']
                
                for comp in range(4):
                    ax = plt.subplot(2, 2, comp+1)
                    
                    # 绘制预测值和真实值
                    ax.plot(range(len(pred_joint)), pred_joint[:, comp], label="pred", linewidth=2)
                    ax.plot(range(len(true_joint)), true_joint[:, comp], label="true", linewidth=2)
                    
                    # 绘制输入值曲线
                    ax.plot(range(len(input_joint)), input_joint[:, comp], label="input", alpha=0.6)
                    
                    # 绘制隐藏层状态曲线
                    # hidden_joint = hidden_norm[:, j, comp].cpu().numpy()
                    # 归一化隐藏状态到[0,1]范围，以便更好地可视化
                    # if len(hidden_joint) > 0:
                    #     h_min, h_max = hidden_joint.min(), hidden_joint.max()
                    #     if h_max > h_min:  # 避免除以零
                    #         hidden_joint = (hidden_joint - h_min) / (h_max - h_min)
                    #         ax.plot(range(len(hidden_joint)), hidden_joint, label="hidden norm", linestyle='--', alpha=0.7)
                    
                    ax.set_title(f"Component {components[comp]}")
                    ax.set_xlabel("Frame")
                    ax.set_ylabel("Value")
                    ax.legend()
                    # ax.grid(True, alpha=0.3)
                
                plt.suptitle(f"{phase.capitalize()} - Joint {j} Quaternion Comparison")
                plt.tight_layout()
                save_path = os.path.join(results_dir, f"{phase}_joint{j}_all_quaternion.png")
                plt.savefig(save_path)
                plt.close(fig)
                print(f"Saved all quaternion plot for {phase} joint {j} at {save_path}")



    


  

