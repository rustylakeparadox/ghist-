# 基础配置
BASE_CONFIG = {
    # 模型参数（应与训练时一致）
    'model': {
        'backbone': 'unet3plus',
        'num_genes': 280,
        'num_cell_types': 8,
        'use_avgexp': False,  # 不使用平均表达谱
        'use_celltype': True,
        'use_neighborhood': True,
        'feature_dim': 256,
    },
    
    # 数据配置
    'data': {
        'he_image_path': "data_demo/he_image.tif",
        'nuclei_seg_path': "data_demo/he_image_nuclei_seg.tif",
        'patch_size': 256,
        'use_pre_segmented': True,  # 使用预分割
        'use_nuclei_sizes': False,  # 不使用细胞核大小信息
        'stain_normalization': True,
    },
    
    # 预测配置
    'prediction': {
        'batch_size': 4,
        'num_workers': 4,
        'device': 'auto',
        'output_dir': './prediction_results',
        'save_visualizations': True,
    },
}

def get_config(**overrides):
    """获取配置，允许覆盖默认值"""
    import copy
    config = copy.deepcopy(BASE_CONFIG)
    
    # 递归更新配置
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                update_dict(d[k], v)
            else:
                d[k] = v
    
    update_dict(config, overrides)
    return config

# 为你的数据创建专用配置
DEMO_CONFIG = get_config(
    data={
        'he_image_path': "/homel/liuwy/GHIST-test/demo/sample/he_image.tif",
        'nuclei_seg_path': "/homel/liuwy/GHIST-test/demo/sample/he_image_nuclei_seg.tif",
        'use_nuclei_sizes': False,
    }
)

# 默认配置
DEFAULT_CONFIG = DEMO_CONFIG

def load_config_from_checkpoint(checkpoint_path):
    """从模型检查点加载配置"""
    import torch
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            model_config = checkpoint['config']
            # 更新配置
            config = get_config()
            config['model'].update({
                'num_genes': model_config.get('num_genes', 280),
                'num_cell_types': model_config.get('num_cell_types', 8),
                'use_avgexp': model_config.get('use_avgexp', False),
            })
            return config
    except Exception as e:
        print(f"Warning: Could not load config from checkpoint: {e}")
    
    return DEFAULT_CONFIG

if __name__ == "__main__":
    # 测试配置
    config = DEFAULT_CONFIG
    print("=== GHIST Prediction Configuration ===")
    print(f"H&E image: {config['data']['he_image_path']}")
    print(f"Nuclei segmentation: {config['data']['nuclei_seg_path']}")
    print(f"Use nuclei sizes: {config['data']['use_nuclei_sizes']}")
    print(f"Use average expression: {config['model']['use_avgexp']}")
    print(f"Output directory: {config['prediction']['output_dir']}")