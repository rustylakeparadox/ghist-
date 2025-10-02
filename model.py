import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from backbone import UNet_3Plus
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import tifffile

class GHISTDataset(Dataset):
    """基于6个输入文件的Dataset类"""
    
    def __init__(self, config, data_dir):
        self.config = config
        self.data_dir = data_dir
        self.patch_size = config.get('patch_size', 256)
        
        # 加载所有数据
        self.load_all_data()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建patch
        self.patches = self.create_patches()
        print(f"Created {len(self.patches)} patches with cells")

    def load_tiff(self, image_path):
        """安全加载TIFF图像"""
        try:
            # 方法1: 使用tifffile（对大TIFF支持更好）
            print(f"Loading TIFF with tifffile: {image_path}")
            tiff_data = tifffile.imread(image_path)
            
            # 转换为PIL图像
            if len(tiff_data.shape) == 3:
                # 如果是多通道，转换为RGB
                if tiff_data.shape[0] == 3:  # 通道在前
                    tiff_data = np.transpose(tiff_data, (1, 2, 0))
                elif tiff_data.shape[2] == 3:  # 通道在后
                    pass  # 已经是正确格式
            elif len(tiff_data.shape) == 2:
                # 如果是单通道，转换为RGB
                tiff_data = np.stack([tiff_data] * 3, axis=-1)
            
            # 确保数据类型正确
            if tiff_data.dtype != np.uint8:
                tiff_data = (tiff_data / tiff_data.max() * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(tiff_data)
            print(f"Successfully loaded TIFF with shape: {tiff_data.shape}")
            return pil_image
            
        except Exception as e:
            print(f"tifffile failed: {e}")
            # 方法2: 使用PIL并忽略截断错误
            try:
                from PIL import ImageFile
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                pil_image = Image.open(image_path)
                pil_image.load()  # 强制加载图像数据
                print(f"Successfully loaded TIFF with PIL: {pil_image.size}")
                return pil_image
            except Exception as e2:
                print(f"PIL also failed: {e2}")
                raise    
    
    def load_all_data(self):
        """加载所有6个输入文件"""
        print("Loading all data files...")
        
        # 1. 平均表达谱
        avg_exp_path = os.path.join(self.data_dir, "avgexp.csv")
        if os.path.exists(avg_exp_path):
            self.avg_exp = pd.read_csv(avg_exp_path, index_col=0)
            print(f"AvgExp reference: {self.avg_exp.shape}")
        else:
            self.avg_exp = None
            print("No AvgExp file found, proceeding without reference profiles")
        
        # 2. 细胞核分割图
        self.nuclei_seg = Image.open(os.path.join(self.data_dir, "he_image_nuclei_seg.tif"))
        print(f"Nuclei segmentation image size: {self.nuclei_seg.size}")
        
        # 3. H&E组织学图像
        he_image_path = os.path.join(self.data_dir, "he_image.tif")
        self.he_image = self.load_tiff(he_image_path)
        
        # 4. 细胞基因表达矩阵
        self.cell_gene_matrix = pd.read_csv(
            os.path.join(self.data_dir, "cell_gene_matrix_filtered.csv"), 
            index_col=0
        )
        print(f"Cell-gene matrix: {self.cell_gene_matrix.shape}")

        # 设置 target_genes 属性
        self.target_genes = self.cell_gene_matrix.columns.tolist()
        print(f"Target genes: {len(self.target_genes)}")

        # 5. 细胞类型标注
        self.cell_types = pd.read_csv(
            os.path.join(self.data_dir, "celltype_filtered.csv"), 
            index_col=0
        )
        print(f"Cell types: {self.cell_types.shape}")
        
        # 6. 匹配的细胞核大小
        self.matched_nuclei = pd.read_csv(
            os.path.join(self.data_dir, "matched_nuclei_filtered.csv")
        )
        print(f"Matched nuclei: {len(self.matched_nuclei)}")
        
        # 提取细胞位置信息
        self.cell_positions = self.extract_cell_positions()
        
        # 创建细胞类型映射
        self.cell_type_mapping = self.create_cell_type_mapping()
        
        # 验证数据一致性
        self.validate_data_consistency()
    
    def extract_cell_positions(self):
        """从细胞核分割图中提取细胞位置"""
        seg_array = np.array(self.nuclei_seg)
        cell_positions = {}
        
        # 如果是多通道，取第一个通道
        if len(seg_array.shape) == 3:
            seg_array = seg_array[:, :, 0]
        
        # 找到所有细胞标签
        cell_labels = np.unique(seg_array)
        cell_labels = cell_labels[cell_labels != 0]  # 移除背景
        
        print(f"Found {len(cell_labels)} cells in segmentation image")
        
        for cell_id in cell_labels:
            positions = np.argwhere(seg_array == cell_id)
            if len(positions) > 0:
                y_coords = positions[:, 0]
                x_coords = positions[:, 1]
                
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                
                centroid_x = int(x_coords.mean())
                centroid_y = int(y_coords.mean())
                
                cell_positions[cell_id] = {
                    'bbox': [x_min, y_min, x_max, y_max],
                    'centroid': [centroid_x, centroid_y],
                    'area': len(positions)
                }
        
        return cell_positions
    
    def create_cell_type_mapping(self):
        """创建细胞类型到整数的映射"""
        unique_cell_types = self.cell_types.iloc[:, 0].unique()
        cell_type_mapping = {cell_type: idx for idx, cell_type in enumerate(unique_cell_types)}
        print(f"Cell type mapping: {cell_type_mapping}")
        return cell_type_mapping
    
    def validate_data_consistency(self):
        """验证数据一致性"""
        # 检查细胞ID在三个文件中是否一致
        expr_cells = set(self.cell_gene_matrix.index)
        type_cells = set(self.cell_types.index)
        matched_cells = set(self.matched_nuclei['cell_id']) if 'cell_id' in self.matched_nuclei.columns else set()
        
        common_cells = expr_cells & type_cells
        if matched_cells:
            common_cells = common_cells & matched_cells
        
        print(f"Common cells across files: {len(common_cells)}")
        
        # 过滤只保留共有的细胞
        self.valid_cell_ids = list(common_cells)
        self.cell_gene_matrix = self.cell_gene_matrix.loc[self.valid_cell_ids]
        self.cell_types = self.cell_types.loc[self.valid_cell_ids]
    
    def create_patches(self):
        """将图像分成patch"""
        img_width, img_height = self.he_image.size
        patch_size = self.patch_size
        
        patches = []
        
        # 计算patch网格
        num_patches_x = (img_width + patch_size - 1) // patch_size
        num_patches_y = (img_height + patch_size - 1) // patch_size
        
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                x_start = i * patch_size
                y_start = j * patch_size
                x_end = min(x_start + patch_size, img_width)
                y_end = min(y_start + patch_size, img_height)
                
                # 找到在这个patch中的有效细胞
                patch_cells = []
                for cell_id in self.valid_cell_ids:
                    if cell_id in self.cell_positions:
                        centroid_x, centroid_y = self.cell_positions[cell_id]['centroid']
                        if (x_start <= centroid_x < x_end and 
                            y_start <= centroid_y < y_end):
                            patch_cells.append(cell_id)
                
                if len(patch_cells) >= self.config.get('min_cells_per_patch', 5):
                    patches.append({
                        'patch_coords': [x_start, y_start, x_end, y_end],
                        'cell_ids': patch_cells
                    })
        
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch_info = self.patches[idx]
        x_start, y_start, x_end, y_end = patch_info['patch_coords']
        patch_cell_ids = patch_info['cell_ids']
        
        # 1. 提取H&E图像patch
        patch_image = self.he_image.crop((x_start, y_start, x_end, y_end))
        patch_tensor = self.transform(patch_image)  # [3, H, W]
        
        # 2. 准备细胞数据
        num_cells = len(patch_cell_ids)
        num_genes = self.cell_gene_matrix.shape[1]
        num_cell_types = len(self.cell_type_mapping)
        
        # 初始化张量
        cell_features = torch.zeros(num_cells, 320)  # 将从backbone提取
        expression_gt = torch.zeros(num_cells, num_genes)
        cell_type_labels = torch.zeros(num_cells, dtype=torch.long)
        cell_positions_list = []
        
        # 填充数据
        for i, cell_id in enumerate(patch_cell_ids):
            # 细胞位置（相对于patch的坐标）
            global_centroid = self.cell_positions[cell_id]['centroid']
            local_centroid = [
                global_centroid[0] - x_start,
                global_centroid[1] - y_start
            ]
            cell_positions_list.append(local_centroid)
            
            # 基因表达目标值
            if cell_id in self.cell_gene_matrix.index:
                expr_values = self.cell_gene_matrix.loc[cell_id].values
                expression_gt[i] = torch.FloatTensor(expr_values)
            
            # 细胞类型标签
            if cell_id in self.cell_types.index:
                cell_type_str = self.cell_types.loc[cell_id].iloc[0]
                cell_type_idx = self.cell_type_mapping[cell_type_str]
                cell_type_labels[i] = cell_type_idx
        
        # 3. 计算邻居组成（细胞类型比例）
        neighborhood_gt = torch.zeros(num_cell_types)
        for cell_type_idx in cell_type_labels:
            neighborhood_gt[cell_type_idx] += 1
        neighborhood_gt = neighborhood_gt / num_cells  # 归一化为比例
        
        # 4. 形态学标签（使用细胞类型作为简化）
        morph_labels = cell_type_labels.clone()
        
        # 5. 细胞特征（在实际训练中会从backbone提取，这里先用随机值）
        cell_features = torch.randn(num_cells, 320)
        
        targets = {
            'expression_gt': expression_gt,
            'cell_type_labels': cell_type_labels,
            'neighborhood_gt': neighborhood_gt,
            'morph_labels': morph_labels
        }
        
        return patch_tensor, cell_features, cell_positions_list, targets


def prepare_avgexp_reference(avg_exp_data, target_genes):
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


class GHISTModel(nn.Module):
    def __init__(self, config, num_genes, num_cell_types, avg_exp_profiles=None):
        super().__init__()
        self.config = config
        self.num_cell_types = num_cell_types
        
        # Backbone - 使用UNet3+
        self.backbone = UNet_3Plus()
        
        # 三个主要的预测头
        self.cell_type_head = CellTypeHead(input_dim=320, num_cell_types=num_cell_types)
        self.neighborhood_head = NeighborhoodHead(input_dim=320, num_cell_types=num_cell_types)
        self.expression_head = ExpressionHead(
            input_dim=320, 
            num_genes=num_genes, 
            num_cell_types=num_cell_types,
            use_avgexp=config.get('use_avgexp', False),
            avg_exp_profiles=avg_exp_profiles
        )
        
        # 一致性模块
        self.consistency_module = ConsistencyModule(num_genes, num_cell_types)
        
    def forward(self, he_images, cell_features, targets=None):
        """
        he_images: [batch_size, 3, H, W] - H&E图像patch
        cell_features: [batch_size, num_cells, 320] - 细胞特征（从backbone提取）
        targets: 目标值字典（用于训练）
        """
        batch_size, num_cells, _ = cell_features.shape
        
        # 1. 细胞类型分类
        cell_type_logits, cell_type_embeddings = self.cell_type_head(
            cell_features, return_embedding=True
        )
        cell_type_pred = F.softmax(cell_type_logits, dim=-1)
        
        # 2. 邻居组成预测（patch级别）
        patch_features = cell_features.mean(dim=1)  # [batch_size, 320]
        neighborhood_pred = self.neighborhood_head(patch_features)  # [batch_size, num_cell_types]
        
        # 3. 基因表达预测 - 移除 cell_type_labels 参数
        expression_outputs = self.expression_head(
            cell_features, 
            neighborhood_pred
            # 移除了 cell_type_labels 参数，因为新的 ExpressionHead 不再需要它
        )
        
        # 4. 一致性模块（仅在训练时使用真实表达）
        consistency_outputs = {}
        if targets is not None and 'expression_gt' in targets:
            consistency_outputs = self.consistency_module(
                expression_outputs['expression'],
                targets['expression_gt']
            )
        
        # 整合所有预测
        predictions = {
            'cell_type_pred': cell_type_pred,
            'cell_type_logits': cell_type_logits,
            'cell_type_embeddings': cell_type_embeddings,
            'neighborhood_est': neighborhood_pred,
            'expression_pred': expression_outputs['expression'],
            'S_initial': expression_outputs['S'],
            **consistency_outputs
        }
        
        # 计算邻居组成从细胞类型预测（用于L_NC,pr损失）
        if targets and 'cell_type_labels' in targets:
            cell_type_labels = targets['cell_type_labels']  # [batch, num_cells]
            neighborhood_from_celltype = torch.zeros(
                batch_size, self.num_cell_types, 
                device=cell_type_labels.device
            )
            
            for i in range(batch_size):
                # 计算每个细胞类型的比例
                for cell_type in range(self.num_cell_types):
                    count = (cell_type_labels[i] == cell_type).sum()
                    neighborhood_from_celltype[i, cell_type] = count.float() / num_cells
            
            predictions['neighborhood_from_celltype'] = neighborhood_from_celltype
        
        return predictions

class ExpressionHead(nn.Module):
    def __init__(self, input_dim, num_genes, num_cell_types, use_avgexp=False, avg_exp_profiles=None):
        super().__init__()
        self.num_genes = num_genes
        self.num_cell_types = num_cell_types
        self.use_avgexp = use_avgexp
        
        # 基础表达预测器
        self.base_expression_predictor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_genes)
        )
        
        # AvgExp融合模块
        if use_avgexp and avg_exp_profiles is not None:
            self.num_ref_types = avg_exp_profiles.shape[0]  # 参考细胞类型数量
            self.avgexp_weight_predictor = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.num_ref_types),
                nn.Softmax(dim=-1)
            )
            # 注册为buffer（固定参考，不参与训练）
            self.register_buffer('avg_exp_profiles', avg_exp_profiles)
            print(f"AvgExp reference integrated: {self.num_ref_types} reference types")
        
        # Multi-head attention相关层
        self.query_proj = nn.Linear(num_cell_types, num_genes)
        self.output_proj = nn.Linear(num_genes, num_genes)
        
        # 论文中的multi-head cross-attention（8头）
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=num_genes,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, cell_features, neighborhood_compositions, cell_type_labels=None):
        """
        cell_features: [batch_size, num_cells, feature_dim]
        neighborhood_compositions: [batch_size, num_cell_types]
        """
        batch_size, num_cells, _ = cell_features.shape
        
        # 1. 基础表达预测
        base_expression = self.base_expression_predictor(cell_features)
        
        # 2. AvgExp融合（如果启用）
        if self.use_avgexp and hasattr(self, 'avg_exp_profiles'):
            # 预测每个细胞对参考细胞类型的权重
            weights = self.avgexp_weight_predictor(cell_features)  # [batch, num_cells, num_ref_types]
            
            # 使用权重混合参考表达谱
            ref_expression = torch.matmul(weights, self.avg_exp_profiles)  # [batch, num_cells, num_genes]
            
            # 结合基础预测和参考信息
            S = base_expression + ref_expression
        else:
            S = base_expression
        
        # 3. Multi-head Cross-Attention整合邻域信息
        # Query: 邻居组成扩展为每个细胞的查询
        query = neighborhood_compositions.unsqueeze(1)  # [batch, 1, num_cell_types]
        query = query.expand(-1, num_cells, -1)         # [batch, num_cells, num_cell_types]
        query_proj = self.query_proj(query)             # [batch, num_cells, num_genes]
        
        # Key/Value: 初始表达S
        attended, _ = self.cross_attention(
            query=query_proj,    # [batch, num_cells, num_genes]
            key=S,               # [batch, num_cells, num_genes]  
            value=S,             # [batch, num_cells, num_genes]
        )
        
        # 4. 最终预测
        b = self.output_proj(attended)
        final_expression = F.relu(S + b)
        
        return {
            'expression': final_expression,
            'S': S
        }

class NeighborhoodHead(nn.Module):
    """邻域组成预测头 - 修正版"""
    
    def __init__(self, input_dim, num_cell_types):
        super().__init__()
        self.num_cell_types = num_cell_types
        
        # 基于patch平均特征预测邻域组成（论文描述）
        self.neighborhood_predictor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_cell_types),
            nn.Softmax(dim=-1)  # 输出细胞类型比例，和为1
        )
    
    def forward(self, patch_features):
        """
        patch_features: [batch_size, feature_dim] - 整个patch的平均特征
        返回: [batch_size, num_cell_types] - 预测的细胞类型组成
        """
        return self.neighborhood_predictor(patch_features)


class CellTypeHead(nn.Module):
    """细胞类型分类头 - 您缺失的这个Head"""
    
    def __init__(self, input_dim, num_cell_types):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_cell_types)
        )
        
        # 用于一致性模块的中间层（用于L_CT,embed损失）
        self.embedding_layer = nn.Linear(input_dim, 128)
    
    def forward(self, cell_features, return_embedding=False):
        """
        cell_features: [batch_size, num_cells, input_dim]
        返回分类logits和可选的嵌入
        """
        # 分类logits
        logits = self.classifier(cell_features)  # [batch, num_cells, num_cell_types]
        
        if return_embedding:
            # 用于一致性损失的嵌入
            embeddings = self.embedding_layer(cell_features)  # [batch, num_cells, 128]
            return logits, embeddings
        else:
            return logits


class ConsistencyModule(nn.Module):
    """一致性模块 - 用于L_CT,embed和L_CT,exp损失"""
    
    def __init__(self, num_genes, num_cell_types):
        super().__init__()
        self.num_cell_types = num_cell_types
        
        # 从基因表达预测细胞类型的模块
        self.expression_to_celltype = nn.Sequential(
            nn.Linear(num_genes, 256),
            nn.ReLU(),
            nn.Linear(256, 128),  # 嵌入层
            nn.ReLU(),
            nn.Linear(128, num_cell_types)
        )
        
        # 嵌入投影层
        self.embedding_proj = nn.Linear(128, 128)
    
    def forward(self, predicted_expression, ground_truth_expression=None):
        """
        predicted_expression: [batch, num_cells, num_genes] - 模型预测的表达
        ground_truth_expression: [batch, num_cells, num_genes] - 真实表达（可选）
        """
        outputs = {}
        
        # 从预测表达得到细胞类型logits和嵌入
        pred_intermediate = self.expression_to_celltype[:-2](predicted_expression)  # 到嵌入层之前
        pred_embedding = self.embedding_proj(pred_intermediate)  # [batch, num_cells, 128]
        pred_logits = self.expression_to_celltype[-2:](pred_intermediate)  # 最后两层得到logits
        
        outputs['cell_type_logits_from_pred'] = pred_logits
        outputs['expr_embedding'] = pred_embedding
        
        # 如果提供真实表达，也计算相应的logits和嵌入
        if ground_truth_expression is not None:
            gt_intermediate = self.expression_to_celltype[:-2](ground_truth_expression)
            gt_embedding = self.embedding_proj(gt_intermediate)
            gt_logits = self.expression_to_celltype[-2:](gt_intermediate)
            
            outputs['cell_type_logits_from_gt'] = gt_logits
            outputs['gt_embedding'] = gt_embedding
        
        return outputs

class GHISTLoss(nn.Module):
    """
    GHIST多任务损失函数
    包含论文中描述的8个损失项
    """
    
    def __init__(self, num_cell_types, config=None):
        super().__init__()
        self.num_cell_types = num_cell_types
        self.config = config or {}
        
        # 论文中提到的特殊缩放因子
        self.ct_embed_scale = self.config.get('ct_embed_scale', 100.0)  # L_CT,embed缩放100倍
        self.ge_adj_scale = 1.0 / max(num_cell_types, 1)  # L_GE,adj除以细胞类型数
        
    def forward(self, predictions, targets, cell_mask=None):
        """
        predictions: 模型预测结果字典
        targets: 真实标签字典
        cell_mask: 可选，有效细胞的掩码
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 1. L_Morph: 形态学损失 - 细胞核分割和分类（公式1）
        if 'morph_logits' in predictions and 'morph_labels' in targets:
            L_Morph = self._morphology_loss(
                predictions['morph_logits'], 
                targets['morph_labels']
            )
            total_loss += L_Morph
            loss_dict['L_Morph'] = L_Morph
        
        # 2. L_CT,class: 细胞类型分类损失（公式2）
        if 'cell_type_pred' in predictions and 'cell_type_labels' in targets:
            L_CT_class = self._cell_type_classification_loss(
                predictions['cell_type_pred'], 
                targets['cell_type_labels'],
                cell_mask
            )
            total_loss += L_CT_class
            loss_dict['L_CT_class'] = L_CT_class
        
        # 3. L_CT,embed: 嵌入一致性损失（公式3）
        if all(key in predictions for key in ['expr_embedding', 'gt_embedding']):
            L_CT_embed = self._embedding_consistency_loss(
                predictions['expr_embedding'],
                predictions['gt_embedding'],
                cell_mask
            )
            total_loss += L_CT_embed * self.ct_embed_scale  # 论文缩放100倍
            loss_dict['L_CT_embed'] = L_CT_embed
        
        # 4. L_CT,exp: 表达一致性损失（公式4）
        if all(key in predictions for key in ['cell_type_logits_from_pred', 'cell_type_logits_from_gt']):
            L_CT_exp = self._expression_consistency_loss(
                predictions['cell_type_logits_from_pred'],
                predictions['cell_type_logits_from_gt'],
                cell_mask
            )
            total_loss += L_CT_exp
            loss_dict['L_CT_exp'] = L_CT_exp
        
        # 5. L_NC,est: 估计邻居组成损失（公式5）
        if all(key in predictions for key in ['neighborhood_est', 'neighborhood_gt']):
            L_NC_est = self._neighborhood_composition_loss(
                predictions['neighborhood_est'],
                targets['neighborhood_gt']
            )
            total_loss += L_NC_est
            loss_dict['L_NC_est'] = L_NC_est
        
        # 6. L_NC,pr: 预测邻居组成损失（公式6）
        if all(key in predictions for key in ['neighborhood_from_celltype', 'neighborhood_gt']):
            L_NC_pr = self._neighborhood_composition_loss(
                predictions['neighborhood_from_celltype'],
                targets['neighborhood_gt']
            )
            total_loss += L_NC_pr
            loss_dict['L_NC_pr'] = L_NC_pr
        
        # 7. L_GE: 基因表达损失（公式15 - 单细胞模式）
        if 'expression_pred' in predictions and 'expression_gt' in targets:
            L_GE = self._gene_expression_loss(
                predictions['expression_pred'],
                targets['expression_gt'],
                cell_mask
            )
            total_loss += L_GE
            loss_dict['L_GE'] = L_GE
        
        # 8. L_GE,adj: 调整后的基因表达损失（公式17）
        if 'expression_adj_pred' in predictions and 'expression_gt' in targets:
            L_GE_adj = self._adjusted_gene_expression_loss(
                predictions['expression_adj_pred'],
                targets['expression_gt'],
                predictions.get('adjusted_cell_types', None),
                cell_mask
            )
            total_loss += L_GE_adj * self.ge_adj_scale  # 除以细胞类型数
            loss_dict['L_GE_adj'] = L_GE_adj
        
        loss_dict['total_loss'] = total_loss
        return total_loss, loss_dict
    
    def _morphology_loss(self, morph_logits, morph_labels):
        """
        公式1: L_Morph = -1/N * Σ_i Σ_j l_ij * log(p_ij)
        细胞核分割和分类的交叉熵损失
        """
        # morph_logits: [N, M+1, H, W]  M个细胞类型+背景
        # morph_labels: [N, H, W] 每个像素的类别标签
        N, _, H, W = morph_logits.shape
        
        # 展平以便计算交叉熵
        logits_flat = morph_logits.permute(0, 2, 3, 1).reshape(-1, self.num_cell_types + 1)
        labels_flat = morph_labels.reshape(-1)
        
        return F.cross_entropy(logits_flat, labels_flat, reduction='mean')
    
    def _cell_type_classification_loss(self, cell_type_pred, cell_type_labels, cell_mask=None):
        """
        公式2: L_CT,class = -1/C * Σ_i Σ_j l_ij * log(p_ij)
        细胞类型分类的交叉熵损失
        """
        # cell_type_pred: [batch_size, num_cells, num_cell_types]
        # cell_type_labels: [batch_size, num_cells]
        
        if cell_mask is not None:
            # 只计算有效细胞的损失
            valid_cells = cell_mask.flatten() > 0
            if valid_cells.sum() == 0:
                return torch.tensor(0.0, device=cell_type_pred.device)
            
            pred_valid = cell_type_pred.flatten(0, 1)[valid_cells]
            labels_valid = cell_type_labels.flatten()[valid_cells]
            return F.cross_entropy(pred_valid, labels_valid, reduction='mean')
        else:
            return F.cross_entropy(
                cell_type_pred.flatten(0, 1), 
                cell_type_labels.flatten(), 
                reduction='mean'
            )
    
    def _embedding_consistency_loss(self, expr_embedding, gt_embedding, cell_mask=None):
        """
        公式3: L_CT,embed = 1/C * Σ_i (1 - cos(x_pr,i, x_gt,i))
        嵌入一致性损失（余弦相似度）
        """
        # expr_embedding: [batch_size, num_cells, embed_dim] - 预测表达的嵌入
        # gt_embedding: [batch_size, num_cells, embed_dim] - 真实表达的嵌入
        
        if cell_mask is not None:
            # 只计算有效细胞
            valid_cells = cell_mask.flatten() > 0
            if valid_cells.sum() == 0:
                return torch.tensor(0.0, device=expr_embedding.device)
            
            expr_valid = expr_embedding.flatten(0, 1)[valid_cells]
            gt_valid = gt_embedding.flatten(0, 1)[valid_cells]
            
            cosine_sim = F.cosine_similarity(expr_valid, gt_valid, dim=-1)
            return (1 - cosine_sim).mean()
        else:
            cosine_sim = F.cosine_similarity(
                expr_embedding.flatten(0, 1), 
                gt_embedding.flatten(0, 1), 
                dim=-1
            )
            return (1 - cosine_sim).mean()
    
    def _expression_consistency_loss(self, logits_from_pred, logits_from_gt, cell_mask=None):
        """
        公式4: L_CT,logits = 1/C * Σ_i (p_pr,i - p_gt,i)^2
        表达一致性损失（MSE在logits上）
        """
        # logits_from_pred: [batch_size, num_cells, num_cell_types] - 预测表达得到的logits
        # logits_from_gt: [batch_size, num_cells, num_cell_types] - 真实表达得到的logits
        
        if cell_mask is not None:
            valid_cells = cell_mask.flatten() > 0
            if valid_cells.sum() == 0:
                return torch.tensor(0.0, device=logits_from_pred.device)
            
            pred_valid = logits_from_pred.flatten(0, 1)[valid_cells]
            gt_valid = logits_from_gt.flatten(0, 1)[valid_cells]
            return F.mse_loss(pred_valid, gt_valid, reduction='mean')
        else:
            return F.mse_loss(
                logits_from_pred.flatten(0, 1), 
                logits_from_gt.flatten(0, 1), 
                reduction='mean'
            )
    
    def _neighborhood_composition_loss(self, pred_composition, gt_composition):
        """
        公式5和6: L_NC = D_KL(P || Q) = Σ_j p_j * log(p_j / q_j)
        邻居组成KL散度损失
        """
        # pred_composition: [batch_size, num_cell_types] - 预测的组成比例
        # gt_composition: [batch_size, num_cell_types] - 真实的组成比例
        
        # 添加小值避免log(0)
        epsilon = 1e-8
        pred_composition = torch.clamp(pred_composition, epsilon, 1.0)
        gt_composition = torch.clamp(gt_composition, epsilon, 1.0)
        
        # 计算KL散度: Σ p * log(p/q)
        kl_div = gt_composition * torch.log(gt_composition / pred_composition)
        return kl_div.sum(dim=-1).mean()
    
    def _gene_expression_loss(self, expression_pred, expression_gt, cell_mask=None):
        """
        公式15: L_GE = 1/C * Σ_i (y_pred,i - y_gt,i)^2
        基因表达MSE损失（单细胞模式）
        """
        # expression_pred: [batch_size, num_cells, num_genes]
        # expression_gt: [batch_size, num_cells, num_genes]
        
        if cell_mask is not None:
            # 扩展mask以匹配基因维度
            mask_expanded = cell_mask.unsqueeze(-1).expand_as(expression_pred)
            valid_elements = mask_expanded > 0
            
            if valid_elements.sum() == 0:
                return torch.tensor(0.0, device=expression_pred.device)
            
            pred_valid = expression_pred[valid_elements]
            gt_valid = expression_gt[valid_elements]
            return F.mse_loss(pred_valid, gt_valid, reduction='mean')
        else:
            return F.mse_loss(expression_pred, expression_gt, reduction='mean')
    
    def _adjusted_gene_expression_loss(self, expression_adj_pred, expression_gt, adjusted_cell_types, cell_mask=None):
        """
        公式17: L_GE,adj = 只在特定细胞类型上计算MSE损失
        调整后的基因表达损失
        """
        if adjusted_cell_types is None:
            return torch.tensor(0.0, device=expression_adj_pred.device)
        
        # adjusted_cell_types: [batch_size, num_cells] - 需要调整的细胞类型标签
        # 只计算特定细胞类型的损失
        
        if cell_mask is not None:
            # 结合细胞类型mask和有效细胞mask
            type_mask = adjusted_cell_types > 0
            valid_mask = cell_mask > 0
            combined_mask = type_mask & valid_mask
        else:
            combined_mask = adjusted_cell_types > 0
        
        if combined_mask.sum() == 0:
            return torch.tensor(0.0, device=expression_adj_pred.device)
        
        # 扩展mask以匹配基因维度
        mask_expanded = combined_mask.unsqueeze(-1).expand_as(expression_adj_pred)
        pred_valid = expression_adj_pred[mask_expanded]
        gt_valid = expression_gt[mask_expanded]
        
        return F.mse_loss(pred_valid, gt_valid, reduction='mean')

   

    

        



