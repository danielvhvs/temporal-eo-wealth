import numpy as np
import os
import argparse

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import center_crop

import pickle
import scipy
import sklearn

class NumpyDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        """
        Args:
            image_dir (str): Directory containing the numpy image files.
            label_file (str): Path to the numpy file containing labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.labels = np.load(label_file)
        self.image_files = sorted(
            [f for f in os.listdir(image_dir)],
            key=lambda x: int(x.split('_')[-1].split('.')[0][6:])
        )
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = np.load(img_path)
        label = self.labels[idx]

        # Split the image into the first 6 bands and the 7th band
        first_3_bands = image[:3, :, :]  # First 6 bands
        first_6_bands = image[3:6, :, :]  # First 6 bands
        seventh_band = image[6, :, :]  # 7th band

        # Center crop both the first 6 bands and the 7th band
        first_3_bands = center_crop(torch.tensor(first_3_bands, dtype=torch.float32), (224, 224))
        first_6_bands = center_crop(torch.tensor(first_6_bands, dtype=torch.float32), (224, 224))
        seventh_band = center_crop(torch.tensor(seventh_band, dtype=torch.float32), (224, 224))

        first_3_bands_norm = (first_3_bands-torch.min(first_3_bands))/(torch.max(first_3_bands)-torch.min(first_3_bands))
        first_6_bands_norm = (first_6_bands-torch.min(first_6_bands))/(torch.max(first_6_bands)-torch.min(first_6_bands))
        seventh_band_norm = (seventh_band-torch.min(seventh_band))/(torch.max(seventh_band)-torch.min(seventh_band))

        # Duplicate the 7th band to create a 3-channel image
        seventh_band_norm = torch.stack([seventh_band_norm, seventh_band_norm, seventh_band_norm], dim=0)

        label = torch.tensor(label, dtype=torch.float32)  # Regression values should also be float32

        if self.transform:
            first_3_bands_norm = self.transform(first_3_bands_norm)
            first_6_bands_norm = self.transform(first_6_bands_norm)
            seventh_band_norm = self.transform(seventh_band_norm)

        return first_3_bands_norm,first_6_bands_norm, seventh_band_norm, label

# Define the combined model
class CombinedModel(nn.Module):
    def __init__(self, model_3_band, model_6_band, model_7_band, hidden_dim=128):
        super(CombinedModel, self).__init__()
        self.model_3_band = model_3_band
        self.model_6_band = model_6_band
        self.model_7_band = model_7_band
        self.fc = nn.Linear(3000, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, input_3_band, input_6_band, input_7_band):
        output_3_band = self.model_3_band(**input_3_band)
        output_6_band = self.model_6_band(**input_6_band)

        output_7_band = self.model_7_band(**input_7_band)
        combined = torch.cat((output_3_band.logits,output_6_band.logits, output_7_band.logits), dim=1)
        hidden = torch.relu(self.fc(combined))
        regression_output = self.regressor(hidden)
        return regression_output

def calc_score(labels: np.ndarray, preds: np.ndarray, metric: str) -> float:
    '''TODO

    See https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#Weighted_correlation_coefficient
    for the weighted correlation coefficient formula.

    Args
    - labels: np.array, shape [N]
    - preds: np.array, shape [N]
    - metric: str, one of ['r2', 'R2', 'mse', 'rank']
        - 'r2': (weighted) squared Pearson correlation coefficient
        - 'R2': (weighted) coefficient of determination
        - 'mse': (weighted) mean squared-error
        - 'rank': (unweighted) Spearman rank correlation coefficient
    - weights: np.array, shape [N]

    Returns: float
    '''
    if metric == 'r2':
        x = scipy.stats.pearsonr(labels, preds)[0] ** 2
        return x

    elif metric == 'R2':
        return sklearn.metrics.r2_score(y_true=labels, y_pred=preds)
    elif metric == 'mse':
        return np.average((labels - preds) ** 2)
    elif metric == 'rank':
        return scipy.stats.spearmanr(labels, preds)[0]
    else:
        raise ValueError(f'Unknown metric: "{metric}"')

def run_model(fold):
    folds_pickle_path = 'data/dhs_incountry_folds.pkl'
    with open(folds_pickle_path, 'rb') as f:
        incountry_folds = pickle.load(f)
        incountry_fold = incountry_folds[fold]
    
    indices_train = incountry_fold["train"]
    indices_val = incountry_fold["val"]
    indices_test = incountry_fold["test"]

    image_dir = "./images_saved/"  # Replace with the path to your images
    label_file = "./numpy_wealth.npy"  # Replace with the path to your labels file

    batch_size = 16  # Set your desired batch size here

    dataset = NumpyDataset(image_dir, label_file)

    dataset_train = Subset(dataset, indices_train)
    dataset_val = Subset(dataset, indices_val)
    dataset_test = Subset(dataset, indices_test)

    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

    # Load the models and processor

    processor_3_band = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model_3_band = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")


    processor_6_band = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model_6_band = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

    processor_7_band = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model_7_band = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")


    device = "cuda"
    # Define the combined model
    combined_model = CombinedModel(model_3_band,model_6_band, model_7_band)
    combined_model = combined_model.cuda()
    optimizer = optim.Adam(combined_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss().cuda()

    num_epochs = 20
    # Training loop

    train_metrics = []
    val_metrics = []
    for epoch in range(num_epochs):
        labellist,predlist = [],[]
        for first_3_bands, first_6_bands, seventh_band, labels in tqdm(train_dataloader):
            # Prepare the images using the processors
            # inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
            inputs_3_band = processor_3_band(first_3_bands, return_tensors="pt",do_rescale=False)
            inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
            inputs_7_band = processor_7_band(seventh_band, return_tensors="pt",do_rescale=False)

            inputs_3_band = inputs_3_band.to(torch.device(device))
            inputs_6_band = inputs_6_band.to(torch.device(device))
            inputs_7_band = inputs_7_band.to(torch.device(device))

            # Forward pass
            predictions = combined_model(inputs_3_band,inputs_6_band, inputs_7_band)
            loss = criterion(predictions.squeeze(), labels.to(torch.device(device)))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            labels,predictions = labels.cpu().detach().numpy().reshape((1,-1))[0], predictions.cpu().detach().numpy().reshape((1,-1))[0]
        
            labellist += list(labels)
            predlist += list(predictions)
        labellist, predlist = np.array(labellist),np.array(predlist)
        train_metrics.append([calc_score(labellist,predlist,'r2'),calc_score(labellist,predlist,'R2'),\
                                    calc_score(labellist,predlist,'mse'),calc_score(labellist,predlist,'rank')])
        if epoch % 2 == 0 or epoch==num_epochs-1:
            with torch.no_grad():
                labellist,predlist = [],[]
                for first_3_bands, first_6_bands, seventh_band, labels in tqdm(val_dataloader):
                    # Prepare the images using the processors
                    # inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
                    inputs_3_band = processor_3_band(first_3_bands, return_tensors="pt",do_rescale=False)
                    inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
                    inputs_7_band = processor_7_band(seventh_band, return_tensors="pt",do_rescale=False)

                    inputs_3_band = inputs_3_band.to(torch.device(device))
                    inputs_6_band = inputs_6_band.to(torch.device(device))
                    inputs_7_band = inputs_7_band.to(torch.device(device))

                    predictions = combined_model(inputs_3_band,inputs_6_band, inputs_7_band)
                    labels,predictions = labels.cpu().detach().numpy().reshape((1,-1))[0], predictions.cpu().detach().numpy().reshape((1,-1))[0]
                
                    labellist += list(labels)
                    predlist += list(predictions)
                labellist, predlist = np.array(labellist),np.array(predlist)
                val_metrics.append([calc_score(labellist,predlist,'r2'),calc_score(labellist,predlist,'R2'),\
                                            calc_score(labellist,predlist,'mse'),calc_score(labellist,predlist,'rank')])

        print("epoch done:", epoch)
        print(val_metrics[-1],"\n")

    with open(f"simple_train_results{fold}.txt","w") as f:
        for i in range(len(train_metrics)):
            f.write(f"{train_metrics[i][0]}\t{train_metrics[i][1]}\t{train_metrics[i][2]}\t{train_metrics[i][3]}\n")
    with open(f"simple_val_results{fold}.txt","w") as f:
        for i in range(len(val_metrics)):
            f.write(f"{val_metrics[i][0]}\t{val_metrics[i][1]}\t{val_metrics[i][2]}\t{val_metrics[i][3]}\n")

    with torch.no_grad():
        labellist,predlist = [],[]
        for first_3_bands, first_6_bands, seventh_band, labels in tqdm(test_dataloader):
            # Prepare the images using the processors
            # inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
            inputs_3_band = processor_3_band(first_3_bands, return_tensors="pt",do_rescale=False)
            inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
            inputs_7_band = processor_7_band(seventh_band, return_tensors="pt",do_rescale=False)

            inputs_3_band = inputs_3_band.to(torch.device(device))
            inputs_6_band = inputs_6_band.to(torch.device(device))
            inputs_7_band = inputs_7_band.to(torch.device(device))

            predictions = combined_model(inputs_3_band,inputs_6_band, inputs_7_band)
            labels,predictions = labels.cpu().detach().numpy().reshape((1,-1))[0], predictions.cpu().detach().numpy().reshape((1,-1))[0]
        
            labellist += list(labels)
            predlist += list(predictions)
        labellist, predlist = np.array(labellist),np.array(predlist)
        test_metrics = [calc_score(labellist,predlist,'r2'),calc_score(labellist,predlist,'R2'),\
                                    calc_score(labellist,predlist,'mse'),calc_score(labellist,predlist,'rank')]

    with open(f"simple_test_results{fold}.txt","w") as f:
        f.write(f"{test_metrics[0]}\t{test_metrics[1]}\t{test_metrics[2]}\t{test_metrics[3]}\n")

# Example usage
if __name__ == "__main__":
    folds = ["A","B","C","D","E"]
    # folds = ["A"]

    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument(
        '--fold', default='A',
        help='folds to run')
    args = parser.parse_args()
    run_model(args.fold)