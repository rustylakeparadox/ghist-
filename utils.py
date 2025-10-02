import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def prepare_avgexp_profiles(avg_exp_data, target_genes):
    """
    准备AvgExp参考矩阵，与目标基因对齐
    
    Args:
        avg_exp_data: 平均表达谱DataFrame
        target_genes: 目标基因列表（来自训练数据）
    
    Returns:
        对齐后的AvgExp参考矩阵
    """
    if avg_exp_data is None:
        return None
    
    print(f"Preparing AvgExp reference matrix...")
    print(f"Original AvgExp shape: {avg_exp_data.shape}")
    print(f"Target genes: {len(target_genes)}")
    
    # 获取AvgExp的基因列表（列名）
    avg_exp_genes = avg_exp_data.columns.tolist()
    
    # 找出共有的基因
    common_genes = set(target_genes) & set(avg_exp_genes)
    print(f"Common genes between AvgExp and target: {len(common_genes)}")
    
    if len(common_genes) == 0:
        print("Warning: No common genes between AvgExp and target data!")
        return None
    
    # 按目标基因顺序筛选和重排AvgExp
    aligned_avg_exp = avg_exp_data[list(common_genes)]
    
    # 如果目标基因中有AvgExp没有的基因，添加零列
    missing_genes = set(target_genes) - set(avg_exp_genes)
    if missing_genes:
        print(f"Adding zeros for {len(missing_genes)} missing genes in AvgExp")
        for gene in missing_genes:
            aligned_avg_exp[gene] = 0.0
    
    # 确保基因顺序与目标一致
    aligned_avg_exp = aligned_avg_exp[target_genes]
    
    print(f"Final AvgExp reference shape: {aligned_avg_exp.shape}")
    print(f"Number of reference cell types in AvgExp: {len(aligned_avg_exp)}")
    
    return torch.FloatTensor(aligned_avg_exp.values)

def collate_fn(batch):
    """批处理函数"""
    patch_tensors, cell_features_list, cell_positions_list, targets_list = zip(*batch)
    
    # 处理patch图像
    patch_tensors = torch.stack(patch_tensors)
    
    # 处理细胞特征 - 填充到最大长度
    max_cells = max([cf.shape[0] for cf in cell_features_list])
    feature_dim = cell_features_list[0].shape[1]
    num_genes = targets_list[0]['expression_gt'].shape[1]
    num_cell_types = targets_list[0]['neighborhood_gt'].shape[0]
    
    batch_size = len(batch)
    
    # 初始化填充后的张量
    cell_features_padded = torch.zeros(batch_size, max_cells, feature_dim)
    expression_gt_padded = torch.zeros(batch_size, max_cells, num_genes)
    cell_type_labels_padded = torch.zeros(batch_size, max_cells, dtype=torch.long)
    cell_mask = torch.zeros(batch_size, max_cells, dtype=torch.bool)
    
    neighborhood_gt = torch.stack([t['neighborhood_gt'] for t in targets_list])
    morph_labels_padded = torch.zeros(batch_size, max_cells, dtype=torch.long)
    
    # 填充数据
    for i, (cf, targets) in enumerate(zip(cell_features_list, targets_list)):
        num_cells = cf.shape[0]
        cell_features_padded[i, :num_cells] = cf
        expression_gt_padded[i, :num_cells] = targets['expression_gt']
        cell_type_labels_padded[i, :num_cells] = targets['cell_type_labels']
        morph_labels_padded[i, :num_cells] = targets['morph_labels']
        cell_mask[i, :num_cells] = True
    
    # 整合目标值
    targets_batch = {
        'expression_gt': expression_gt_padded,
        'cell_type_labels': cell_type_labels_padded,
        'neighborhood_gt': neighborhood_gt,
        'morph_labels': morph_labels_padded,
        'cell_mask': cell_mask
    }
    
    return patch_tensors, cell_features_padded, cell_positions_list, targets_batch

def normalize_expression_data(expression_data, method='log1p'):
    """标准化表达数据"""
    if method == 'log1p':
        # log(1 + x) 标准化
        return np.log1p(expression_data)
    elif method == 'standard':
        # Z-score 标准化
        scaler = StandardScaler()
        return scaler.fit_transform(expression_data)
    else:
        return expression_data

def setup_logging(experiment_dir):
    """设置日志"""
    import logging
    log_file = os.path.join(experiment_dir, 'logs', 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def save_config(config, save_path):
    """保存配置到文件"""
    import json
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)

def load_config(load_path):
    """从文件加载配置"""
    import json
    with open(load_path, 'r') as f:
        return json.load(f)

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device():
    """获取可用设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')