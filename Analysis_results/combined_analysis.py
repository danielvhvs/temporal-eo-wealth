import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ResNet Data Preprocessing
resnet_files = ["resultsA.csv", "resultsB.csv", "resultsC.csv", "resultsD.csv", "resultsE.csv"]
resnet_data_frames = []
for i, file_path in enumerate(resnet_files):
    df = pd.read_csv(file_path)
    df['Fold'] = f'Fold {chr(65 + i)}'
    resnet_data_frames.append(df)
resnet_data = pd.concat(resnet_data_frames)
resnet_data = resnet_data[resnet_data['epoch'] <= 10]
resnet_summary = resnet_data.groupby(['epoch', 'split'])['r2'].agg(['mean', 'std']).reset_index()

# Transformer Data Preprocessing
transformer_train_files = [
    "train_resultsA.txt", "train_resultsB.txt", "train_resultsC.txt",
    "train_resultsD.txt", "train_resultsE.txt"
]
transformer_val_files = [
    "val_resultsA.txt", "val_resultsB.txt", "val_resultsC.txt",
    "val_resultsD.txt", "val_resultsE.txt"
]
columns = ['r2', 'R2', 'mse', 'rank']
transformer_data_frames = []
for i, (train_file, val_file) in enumerate(zip(transformer_train_files, transformer_val_files)):
    train_df = pd.read_csv(train_file, sep='\t', header=None, names=columns)
    val_df = pd.read_csv(val_file, sep='\t', header=None, names=columns)
    train_df['split'] = 'train_eval'
    val_df['split'] = 'validation'
    train_df['Fold'] = f'Fold {chr(65 + i)}'
    val_df['Fold'] = f'Fold {chr(65 + i)}'
    train_df['epoch'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    val_df['epoch'] = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]
    transformer_data_frames.append(train_df)
    transformer_data_frames.append(val_df)
transformer_data = pd.concat(transformer_data_frames)
transformer_data = transformer_data[transformer_data['epoch'] <= 11]
transformer_summary = transformer_data.groupby(['epoch', 'split'])['r2'].agg(['mean', 'std']).reset_index()

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Left subplot: ResNet
resnet_train = resnet_summary[resnet_summary['split'] == 'train_eval']
resnet_val = resnet_summary[resnet_summary['split'] == 'val']
axes[0].plot(resnet_train['epoch'], resnet_train['mean'], 'r--', label='ResNet Train (mean)')
axes[0].fill_between(resnet_train['epoch'], resnet_train['mean'] - resnet_train['std'], 
                     resnet_train['mean'] + resnet_train['std'], color='red', alpha=0.2)
axes[0].plot(resnet_val['epoch'], resnet_val['mean'], 'b-', label='ResNet Validation (mean)')
axes[0].fill_between(resnet_val['epoch'], resnet_val['mean'] - resnet_val['std'], 
                     resnet_val['mean'] + resnet_val['std'], color='blue', alpha=0.2)
axes[0].set_title('ResNet Results')
axes[0].legend(loc='upper left')
axes[0].grid(True)

# Right subplot: Transformer
transformer_train = transformer_summary[transformer_summary['split'] == 'train_eval']
transformer_val = transformer_summary[transformer_summary['split'] == 'validation']
axes[1].plot(transformer_train['epoch'], transformer_train['mean'], 'g--', label='Transformer Train (mean)')
axes[1].fill_between(transformer_train['epoch'], transformer_train['mean'] - transformer_train['std'], 
                     transformer_train['mean'] + transformer_train['std'], color='green', alpha=0.2)
axes[1].plot(transformer_val['epoch'], transformer_val['mean'], 'm-', label='Transformer Validation (mean)')
axes[1].fill_between(transformer_val['epoch'], transformer_val['mean'] - transformer_val['std'], 
                     transformer_val['mean'] + transformer_val['std'], color='magenta', alpha=0.2)
axes[1].set_title('Transformer Results')
axes[1].legend(loc='upper left')
axes[1].grid(True)

# Set labels
fig.supxlabel('Epoch', y=0.02, fontsize=12)  # Shared x-axis label
fig.supylabel('r² Score', x=0.04, fontsize=12)  # Shared y-axis label outside plot

# layout
plt.tight_layout(rect=[0.05, 0.05, 1, 1])  
plt.show()

# Second Plot: Validation comparison
plt.figure(figsize=(12, 6))
plt.plot(transformer_val['epoch'], transformer_val['mean'], 'm-', label='Transformer Validation (mean)')
plt.fill_between(transformer_val['epoch'], transformer_val['mean'] - transformer_val['std'], 
                 transformer_val['mean'] + transformer_val['std'], color='magenta', alpha=0.2)
plt.plot(resnet_val['epoch'], resnet_val['mean'], 'b-', label='ResNet Validation (mean)')
plt.fill_between(resnet_val['epoch'], resnet_val['mean'] - resnet_val['std'], 
                 resnet_val['mean'] + resnet_val['std'], color='blue', alpha=0.2)
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Validation Results: Transformer vs ResNet')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()
