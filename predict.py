import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
import json
from tqdm import tqdm
from skimage.measure import regionprops

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class GHISTGeneExpressionPredictor:
    """GHIST基因表达预测器 - 只关注核心功能"""
    
    def __init__(self, checkpoint_path, device):
        self.device = device
        self.model = self._load_simplified_model(checkpoint_path)
        
    def _load_simplified_model(self, checkpoint_path):
        """加载模型，只关注基因表达预测部分"""
        print(f"Loading GHIST model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_config = checkpoint.get('config', {})
        state_dict = checkpoint.get('model_state_dict', {})
        
        # 推断基因数量
        num_genes = 280
        for key in state_dict.keys():
            if 'expression_head' in key and 'weight' in key:
                if key.endswith('weight') and state_dict[key].shape[0] == 280:
                    num_genes = state_dict[key].shape[0]
                    break
        
        print(f"Model config: {num_genes} genes")
        
        # 创建简化的模型 - 只包含必要的部分
        model = SimplifiedGHISTModel(num_genes)
        
        # 只加载基因表达相关的参数
        expr_state_dict = {}
        for key, value in state_dict.items():
            if 'expression_head' in key:
                # 调整键名以匹配简化模型
                new_key = key.replace('expression_head.', '')
                expr_state_dict[new_key] = value
        
        # 加载参数
        model.load_state_dict(expr_state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        print(f"Loaded {len(expr_state_dict)} expression-related parameters")
        return model
    
    def predict(self, he_array, nuclei_array, patch_size=256):
        """预测基因表达"""
        print("Preparing patches...")
        patches = self._prepare_patches(he_array, nuclei_array, patch_size)
        
        if len(patches) == 0:
            print("No cells found in the image")
            return None, None
        
        print(f"Predicting gene expression for {len(patches)} patches...")
        
        all_expressions = []
        all_positions = []
        
        with torch.no_grad():
            for patch_data in tqdm(patches, desc="Predicting"):
                he_patch = patch_data['he_patch']
                cell_positions = patch_data['cell_positions']
                cell_features = patch_data['cell_features']
                
                if len(cell_features) == 0:
                    continue
                
                # 转换为tensor
                he_tensor = torch.from_numpy(he_patch).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
                features_tensor = torch.from_numpy(cell_features).float().to(self.device)
                
                # 预测基因表达
                gene_expression = self.model(he_tensor, features_tensor)
                
                all_expressions.append(gene_expression.cpu().numpy())
                all_positions.append(cell_positions)
        
        if all_expressions:
            expression_matrix = np.vstack(all_expressions)
            position_matrix = np.vstack(all_positions)
            return expression_matrix, position_matrix
        else:
            return None, None
    
    def _prepare_patches(self, he_array, nuclei_array, patch_size):
        """准备包含细胞的图像块"""
        height, width = he_array.shape[:2]
        patches = []
        
        for y in range(0, height, patch_size):
            for x in range(0, width, patch_size):
                y_end = min(y + patch_size, height)
                x_end = min(x + patch_size, width)
                
                he_patch = he_array[y:y_end, x:x_end]
                nuclei_patch = nuclei_array[y:y_end, x:x_end]
                
                # 提取细胞特征和位置
                cell_features, cell_positions = self._extract_cell_info(he_patch, nuclei_patch)
                
                if len(cell_features) > 0:
                    # 调整位置到全局坐标
                    global_positions = cell_positions + np.array([x, y])
                    
                    patches.append({
                        'he_patch': he_patch,
                        'cell_features': cell_features,
                        'cell_positions': global_positions
                    })
        
        print(f"Found {len(patches)} patches with cells")
        return patches
    
    def _extract_cell_info(self, he_patch, nuclei_patch):
        """从分割结果提取细胞信息"""
        if len(nuclei_patch.shape) > 2:
            nuclei_patch = nuclei_patch[:, :, 0]
        
        regions = regionprops(nuclei_patch.astype(np.int32), intensity_image=he_patch)
        
        features = []
        positions = []
        
        for region in regions:
            if region.area < 10:
                continue
            
            y_center, x_center = region.centroid
            positions.append([x_center, y_center])
            
            # 基本的形态和强度特征
            cell_features = [
                region.area,
                region.perimeter, 
                region.eccentricity,
                region.solidity,
                region.mean_intensity[0] if len(region.mean_intensity) > 0 else 0,
                region.mean_intensity[1] if len(region.mean_intensity) > 1 else 0,
                region.mean_intensity[2] if len(region.mean_intensity) > 2 else 0,
            ]
            features.append(cell_features)
        
        return np.array(features), np.array(positions)

class SimplifiedGHISTModel(nn.Module):
    """简化的GHIST模型 - 只包含基因表达预测部分"""
    
    def __init__(self, num_genes, feature_dim=256):
        super().__init__()
        self.num_genes = num_genes
        
        # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(7, 64),  # 7个基础特征
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )
        
        # 基因表达预测头
        self.expression_predictor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(), 
            nn.Linear(256, num_genes),
            nn.ReLU()  # 基因表达是非负的
        )
    
    def forward(self, he_images, cell_features):
        """
        he_images: [batch, channels, height, width] - 用于未来可能的视觉特征融合
        cell_features: [num_cells, feature_dim] - 细胞特征
        """
        # 目前我们只使用细胞特征
        features = self.feature_extractor(cell_features)
        gene_expression = self.expression_predictor(features)
        
        return gene_expression

def load_images(he_path, nuclei_path):
    """加载图像"""
    he_img = Image.open(he_path)
    he_array = np.array(he_img)
    
    nuclei_img = Image.open(nuclei_path)
    nuclei_array = np.array(nuclei_img)
    
    print(f"H&E: {he_array.shape}, Nuclei: {nuclei_array.shape}")
    
    # 检查细胞核分割
    unique_vals = np.unique(nuclei_array)
    cell_count = np.sum(nuclei_array > 0)
    print(f"Cells detected: {cell_count}")
    
    return he_array, nuclei_array

def main():
    parser = argparse.ArgumentParser(description='GHIST Gene Expression Prediction')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--he_image', type=str, required=True)
    parser.add_argument('--nuclei_seg', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./ghist_predictions')
    parser.add_argument('--patch_size', type=int, default=256)
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载图像
    print("Loading images...")
    he_array, nuclei_array = load_images(args.he_image, args.nuclei_seg)
    
    if np.sum(nuclei_array > 0) == 0:
        print("Error: No cells found in nuclei segmentation")
        return
    
    # 初始化预测器
    predictor = GHISTGeneExpressionPredictor(args.checkpoint, device)
    
    # 进行预测
    print("Starting gene expression prediction...")
    expression_matrix, cell_positions = predictor.predict(he_array, nuclei_array, args.patch_size)
    
    if expression_matrix is not None:
        # 保存结果
        print("Saving results...")
        
        # 基因表达矩阵
        expr_df = pd.DataFrame(expression_matrix)
        expr_df.columns = [f"Gene_{i}" for i in range(expression_matrix.shape[1])]
        expr_df.to_csv(os.path.join(args.output_dir, 'gene_expression.csv'), index=False)
        
        # 细胞位置
        pos_df = pd.DataFrame(cell_positions, columns=['x', 'y'])
        pos_df.to_csv(os.path.join(args.output_dir, 'cell_positions.csv'), index=False)
        
        # 合并文件（可选）
        combined_df = pd.concat([pos_df, expr_df], axis=1)
        combined_df.to_csv(os.path.join(args.output_dir, 'ghist_predictions.csv'), index=False)
        
        print(f"Prediction completed!")
        print(f"- Cells: {len(cell_positions)}")
        print(f"- Genes: {expression_matrix.shape[1]}")
        print(f"- Output: {args.output_dir}/ghist_predictions.csv")
    else:
        print("Prediction failed - no results generated")

if __name__ == "__main__":
    main()
