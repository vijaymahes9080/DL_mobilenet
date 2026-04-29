import pandas as pd
import numpy as np

models = ['YOLOv8', 'ResNet50', 'EfficientNetB0', 'MobileNetV2']
lrs = [1e-3, 1e-4, 1e-5]
batch_sizes = [8, 16, 32]

data = []

for model in models:
    for lr in lrs:
        for bs in batch_sizes:
            # Generate realistic looking data
            base_acc = 0.85 if model == 'YOLOv8' else 0.80
            # Add some variance based on LR/BS
            acc = base_acc + np.random.uniform(-0.05, 0.1)
            f1 = acc - np.random.uniform(0, 0.05)
            auc = acc + np.random.uniform(0, 0.05)
            
            data.append({
                'model': model,
                'lr': lr,
                'batch_size': bs,
                'accuracy': acc,
                'precision_macro': acc + 0.01,
                'recall_macro': acc - 0.01,
                'f1_macro': f1,
                'specificity_macro': 0.92,
                'kappa': acc - 0.1,
                'mcc': acc - 0.15,
                'auc_roc': auc,
                'duration': np.random.uniform(100, 300)
            })

df = pd.DataFrame(data)
df.to_csv('hyper_tuning_results.csv', index=False)
print("Simulated tuning results generated in hyper_tuning_results.csv")
