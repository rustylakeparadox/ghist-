import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from model import ExpressionHead

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import GHISTModel, GHISTDataset
from utils import get_device, collate_fn

def load_model(checkpoint_path, config, device, dataset=None):
    """加载训练好的模型"""
    print(f"Loading model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 获取模型参数 - 修复KeyError
    if 'num_genes' in checkpoint['config']:
        num_genes = checkpoint['config']['num_genes']
    else:
        # 从数据集获取基因数量
        if dataset is not None:
            num_genes = dataset.cell_gene_matrix.shape[1]
        else:
            # 如果无法从数据集获取，使用默认值或从config获取
            num_genes = config.get('num_genes', 280)  # 默认280个基因
        print(f"Using num_genes: {num_genes} (from {'dataset' if dataset else 'config'})")
    
    # 获取细胞类型数量
    if 'cell_type_mapping' in checkpoint:
        num_cell_types = len(checkpoint['cell_type_mapping'])
    
    # 准备AvgExp（如果使用）
    avg_exp_profiles = None
    if config.get('use_avgexp', False):
        # 尝试从检查点加载，否则从config准备
        if 'avg_exp_profiles' in checkpoint:
            avg_exp_profiles = checkpoint['avg_exp_profiles'].to(device)
        elif dataset is not None and hasattr(dataset, 'avg_exp') and dataset.avg_exp is not None:
            # 从数据集重新准备AvgExp
            from utils import prepare_avgexp_reference
            avg_exp_profiles = prepare_avgexp_reference(dataset.avg_exp, dataset.target_genes)
            if avg_exp_profiles is not None:
                avg_exp_profiles = avg_exp_profiles.to(device)
    
    # 初始化模型
    model = GHISTModel(config, num_genes, num_cell_types, avg_exp_profiles)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully. Epoch: {checkpoint['epoch']}, Loss: {checkpoint['loss']:.4f}")
    return model, checkpoint

def compute_gene_expression_metrics(predictions, targets, cell_mask=None):
    """专门计算基因表达预测的指标"""
    metrics = {}
    
    pred_expr = predictions['expression_pred']
    true_expr = targets['expression_gt']
    
    if cell_mask is not None:
        # 只计算有效细胞的指标
        mask_expanded = cell_mask.unsqueeze(-1).expand_as(pred_expr)
        pred_expr_valid = pred_expr[mask_expanded].view(-1, pred_expr.shape[-1])
        true_expr_valid = true_expr[mask_expanded].view(-1, true_expr.shape[-1])
    else:
        pred_expr_valid = pred_expr.reshape(-1, pred_expr.shape[-1])
        true_expr_valid = true_expr.reshape(-1, true_expr.shape[-1])
    
    # 1. 总体MSE
    mse = F.mse_loss(pred_expr_valid, true_expr_valid)
    metrics['overall_mse'] = mse.item()
    
    # 2. 基因级别的Pearson相关系数
    gene_correlations = []
    gene_mse = []
    
    for gene_idx in range(pred_expr_valid.shape[1]):
        pred_gene = pred_expr_valid[:, gene_idx].cpu().numpy()
        true_gene = true_expr_valid[:, gene_idx].cpu().numpy()
        
        # 只计算有变化的基因
        if np.std(pred_gene) > 0 and np.std(true_gene) > 0:
            corr, _ = pearsonr(pred_gene, true_gene)
            gene_correlations.append(corr)
            
            # 单个基因的MSE
            gene_mse.append(F.mse_loss(
                torch.tensor(pred_gene), 
                torch.tensor(true_gene)
            ).item())
    
    if gene_correlations:
        metrics['mean_gene_correlation'] = np.mean(gene_correlations)
        metrics['std_gene_correlation'] = np.std(gene_correlations)
        metrics['median_gene_correlation'] = np.median(gene_correlations)
        metrics['max_gene_correlation'] = np.max(gene_correlations)
        metrics['min_gene_correlation'] = np.min(gene_correlations)
        
        # 相关性分布统计
        metrics['genes_above_0.7'] = np.sum(np.array(gene_correlations) > 0.7)
        metrics['genes_above_0.5'] = np.sum(np.array(gene_correlations) > 0.5)
        metrics['genes_above_0.3'] = np.sum(np.array(gene_correlations) > 0.3)
        
        metrics['mean_gene_mse'] = np.mean(gene_mse)
    
    # 3. 细胞级别的相关性
    cell_correlations = []
    for cell_idx in range(pred_expr_valid.shape[0]):
        pred_cell = pred_expr_valid[cell_idx].cpu().numpy()
        true_cell = true_expr_valid[cell_idx].cpu().numpy()
        
        if np.std(pred_cell) > 0 and np.std(true_cell) > 0:
            corr, _ = pearsonr(pred_cell, true_cell)
            cell_correlations.append(corr)
    
    if cell_correlations:
        metrics['mean_cell_correlation'] = np.mean(cell_correlations)
        metrics['std_cell_correlation'] = np.std(cell_correlations)
    
    return metrics

def run_gene_expression_validation(model, dataloader, device, save_visualizations=False, output_dir='./validation_results'):
    """运行基因表达验证"""
    model.eval()
    all_metrics = []
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (patch_tensors, cell_features, cell_positions, targets) in enumerate(tqdm(dataloader, desc="Validating gene expression")):
            # 移动到设备
            patch_tensors = patch_tensors.to(device)
            cell_features = cell_features.to(device)
            
            # 移动目标值到设备
            targets_device = {}
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    targets_device[key] = value.to(device)
                else:
                    targets_device[key] = value
            
            # 前向传播
            predictions = model(patch_tensors, cell_features, targets=targets_device)
            
            # 只计算基因表达指标
            metrics = compute_gene_expression_metrics(predictions, targets_device, targets_device.get('cell_mask', None))
            metrics['batch_idx'] = batch_idx
            all_metrics.append(metrics)
    
    return all_metrics

def aggregate_gene_metrics(all_metrics):
    """聚合基因表达指标"""
    aggregated = {}
    
    # 收集所有batch的指标
    all_correlations = []
    all_mse = []
    
    for metrics in all_metrics:
        if 'mean_gene_correlation' in metrics:
            all_correlations.append(metrics['mean_gene_correlation'])
        if 'overall_mse' in metrics:
            all_mse.append(metrics['overall_mse'])
    
    if all_correlations:
        aggregated['final_mean_gene_correlation'] = np.mean(all_correlations)
        aggregated['final_std_gene_correlation'] = np.std(all_correlations)
        aggregated['final_median_gene_correlation'] = np.median(all_correlations)
    
    if all_mse:
        aggregated['final_mean_mse'] = np.mean(all_mse)
        aggregated['final_std_mse'] = np.std(all_mse)
    
    # 汇总基因级别的统计
    total_genes_above_07 = sum(metrics.get('genes_above_0.7', 0) for metrics in all_metrics)
    total_genes_above_05 = sum(metrics.get('genes_above_0.5', 0) for metrics in all_metrics)
    total_genes_above_03 = sum(metrics.get('genes_above_0.3', 0) for metrics in all_metrics)
    
    aggregated['total_genes_above_0.7'] = total_genes_above_07
    aggregated['total_genes_above_0.5'] = total_genes_above_05
    aggregated['total_genes_above_0.3'] = total_genes_above_03
    
    return aggregated

def save_validation_results(aggregated_metrics, all_metrics, output_dir):
    """保存验证结果"""
    
    def convert_to_serializable(obj):
        """将 NumPy 数据类型转换为 Python 原生数据类型"""
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_to_serializable(item) for item in obj)
        else:
            return obj
    
    # 转换数据类型
    aggregated_metrics_serializable = convert_to_serializable(aggregated_metrics)
    
    # 保存聚合指标
    with open(os.path.join(output_dir, 'gene_expression_metrics.json'), 'w') as f:
        json.dump(aggregated_metrics_serializable, f, indent=2)
    
    # 保存详细指标 - 确保 DataFrame 中的数据也是可序列化的
    all_metrics_serializable = convert_to_serializable(all_metrics)
    detailed_metrics_df = pd.DataFrame(all_metrics_serializable)
    detailed_metrics_df.to_csv(os.path.join(output_dir, 'detailed_gene_metrics.csv'), index=False)
    
    # 打印主要结果
    print("\n" + "="*60)
    print("GENE EXPRESSION VALIDATION RESULTS")
    print("="*60)
    
    # 使用转换后的指标进行打印
    metrics = aggregated_metrics_serializable
    
    if 'final_mean_gene_correlation' in metrics:
        print(f"Mean Gene Correlation: {metrics['final_mean_gene_correlation']:.4f} ± {metrics['final_std_gene_correlation']:.4f}")
        print(f"Median Gene Correlation: {metrics['final_median_gene_correlation']:.4f}")
    
    if 'final_mean_mse' in metrics:
        print(f"Mean MSE: {metrics['final_mean_mse']:.4f} ± {metrics['final_std_mse']:.4f}")
    
    print(f"Genes with correlation > 0.7: {metrics.get('total_genes_above_0.7', 0)}")
    print(f"Genes with correlation > 0.5: {metrics.get('total_genes_above_0.5', 0)}")
    print(f"Genes with correlation > 0.3: {metrics.get('total_genes_above_0.3', 0)}")

def main():
    parser = argparse.ArgumentParser(description='GHIST Gene Expression Validation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, default='data_demo',
                       help='Directory containing validation data')
    parser.add_argument('--output_dir', type=str, default='./gene_expression_validation',
                       help='Directory to save validation results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for validation')
    parser.add_argument('--visualize', action='store_true',
                       help='Save visualization plots')
    parser.add_argument('--use_avgexp', action='store_true',
                       help='Use average expression profiles')
    
    args = parser.parse_args()
    
    # 设置设备
    device = get_device()
    print(f"Using device: {device}")
    
    config = {
        'data_dir': args.data_dir,
        'use_avgexp': args.use_avgexp,
        'batch_size': args.batch_size,
        'patch_size': 256,
        'num_workers': 4,
        'min_cells_per_patch': 5,
        'use_celltype': True
    }
    
    # 创建验证数据集
    print("Loading validation dataset...")
    val_dataset = GHISTDataset(config, args.data_dir)
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 加载模型
    model, checkpoint = load_model(args.checkpoint, config, device, dataset=val_dataset)
    
    # 创建数据加载器
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.get('num_workers', 4)
    )
    
    print(f"Validation dataset: {len(val_dataset)} patches")
    
    # 运行基因表达验证
    print("Starting gene expression validation...")
    all_metrics = run_gene_expression_validation(
        model, 
        val_dataloader, 
        device, 
        save_visualizations=args.visualize,
        output_dir=args.output_dir
    )
    
    # 聚合指标并保存结果
    aggregated_metrics = aggregate_gene_metrics(all_metrics)
    save_validation_results(aggregated_metrics, all_metrics, args.output_dir)
    
    print(f"\nValidation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()