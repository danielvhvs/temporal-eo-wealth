import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np


# Load and combine data
file_paths = ["resultsA.csv", "resultsB.csv", "resultsC.csv", "resultsD.csv", "resultsE.csv"]
data_frames = []
for i, file_path in enumerate(file_paths):
    df = pd.read_csv(file_path)
    df['Fold'] = f'Fold {chr(65 + i)}'  # Add fold label (Fold A, Fold B, etc.)
    data_frames.append(df)

# Concatenate all folds
data = pd.concat(data_frames)

# Filter epochs <= 10
data = data[data['epoch'] <= 10]

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
val_data = summary[summary['split'] == 'val']
plt.plot(val_data['epoch'], val_data['mean'], 'b-', label='Validation (mean)')
plt.fill_between(val_data['epoch'], 
                 val_data['mean'] - val_data['std'], 
                 val_data['mean'] + val_data['std'], 
                 color='blue', alpha=0.2, label='Validation (std)')

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Cross-Validation Results: Train vs Validation R²')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
