config = {
    'data_dir': './data_demo',
    'model_save_dir': './models',
    'epochs': 10,
    'batch_size': 4,
    'learning_rate': 1e-3,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'lr_patience': 10,
    'lr_factor': 0.5,
    'patch_size': 256,
    'num_workers': 4,
    'min_cells_per_patch': 5,
    'use_celltype': True,
    'use_avgexp': True  # 使用平均表达谱
}

