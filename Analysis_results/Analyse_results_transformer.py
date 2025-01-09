import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths
train_files = [
    "train_resultsA.txt", "train_resultsB.txt", "train_resultsC.txt",
    "train_resultsD.txt", "train_resultsE.txt"
]
val_files = [
    "val_resultsA.txt", "val_resultsB.txt", "val_resultsC.txt",
    "val_resultsD.txt", "val_resultsE.txt"
]

# Columns corresponding to data in txt files (since headers are absent)
columns = ['r2', 'R2', 'mse', 'rank']

# Load and preprocess data
data_frames = []
for i, (train_file, val_file) in enumerate(zip(train_files, val_files)):
    # Load train and validation data
    train_df = pd.read_csv(train_file, sep='\t', header=None, names=columns)
    val_df = pd.read_csv(val_file, sep='\t', header=None, names=columns)
    
    # Add split and fold labels
    train_df['split'] = 'train_eval'
    val_df['split'] = 'validation'
    train_df['Fold'] = f'Fold {chr(65 + i)}'
    val_df['Fold'] = f'Fold {chr(65 + i)}'
    
    # Add epochs (based on the provided order)
    epochs_val = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 20]
    epochs_train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    train_df['epoch'] = epochs_train
    val_df['epoch'] = epochs_val
    
    # Combine train and validation data
    data_frames.append(train_df)
    data_frames.append(val_df)

# Concatenate all data
data = pd.concat(data_frames)

# Filter epochs ≤ 10
#data = data[data['epoch'] <= 10]

# Compute mean and std of R² by epoch and split
summary = data.groupby(['epoch', 'split'])['r2'].agg(['mean', 'std']).reset_index()

# Plotting
plt.figure(figsize=(10, 6))

# Train line (dotted)
train_data = summary[summary['split'] == 'train_eval']
plt.plot(train_data['epoch'], train_data['mean'], 'r--', label='Train (mean)')
plt.fill_between(train_data['epoch'], 
                 train_data['mean'] - train_data['std'], 
                 train_data['mean'] + train_data['std'], 
                 color='red', alpha=0.2, label='Train (std)')

# Validation line (solid)
val_data = summary[summary['split'] == 'validation']
plt.plot(val_data['epoch'], val_data['mean'], 'b-', label='Validation (mean)')
plt.fill_between(val_data['epoch'], 
                 val_data['mean'] - val_data['std'], 
                 val_data['mean'] + val_data['std'], 
                 color='blue', alpha=0.2, label='Validation (std)')

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Cross-Validation Results Transformer: Train vs Validation R² ')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()