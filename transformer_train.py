import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
from tqdm import tqdm

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import center_crop

import pickle
import scipy
import sklearn
import terratorch # even though we don't use the import directly, we need it so that the models are available in the timm registry
from terratorch.models import EncoderDecoderFactory
from terratorch.datasets import HLSBands

def create_gfm():
    model_factory = EncoderDecoderFactory()

    # Let's build a segmentation model
    # Parameters prefixed with backbone_ get passed to the backbone
    # Parameters prefixed with decoder_ get passed to the decoder
    # Parameters prefixed with head_ get passed to the head

    model = model_factory.build_model(task="segmentation",
            backbone="prithvi_vit_100",
            decoder="FCNDecoder",
            backbone_bands=[
                HLSBands.BLUE,
                HLSBands.GREEN,
                HLSBands.RED,
                HLSBands.NIR_NARROW,
                HLSBands.SWIR_1,
                HLSBands.SWIR_2,
            ],
            necks=[{"name": "SelectIndices", "indices": [-1]},
                {"name": "ReshapeTokensToImage"}],
            num_classes=4,
            backbone_pretrained=True,
            backbone_num_frames=1,
            decoder_channels=128,
            head_dropout=0.2
        )
    return model


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
        first_6_bands = image[:6, :, :]  # First 6 bands
        seventh_band = image[6, :, :]  # 7th band

        # Center crop both the first 6 bands and the 7th band
        first_6_bands = center_crop(torch.tensor(first_6_bands, dtype=torch.float32), (224, 224))
        seventh_band = center_crop(torch.tensor(seventh_band, dtype=torch.float32), (224, 224))

        first_6_bands_norm = (first_6_bands-torch.min(first_6_bands))/(torch.max(first_6_bands)-torch.min(first_6_bands))
        seventh_band_norm = (seventh_band-torch.min(seventh_band))/(torch.max(seventh_band)-torch.min(seventh_band))

        # Duplicate the 7th band to create a 3-channel image
        seventh_band_norm = torch.stack([seventh_band_norm, seventh_band_norm, seventh_band_norm], dim=0)

        label = torch.tensor(label, dtype=torch.float32)  # Regression values should also be float32

        if self.transform:
            first_6_bands_norm = self.transform(first_6_bands_norm)
            seventh_band_norm = self.transform(seventh_band_norm)

        return first_6_bands_norm, seventh_band_norm, label

# Define the combined model
class CombinedModel(nn.Module):
    def __init__(self, model_6_band, model_7_band, hidden_dim=128):
        super(CombinedModel, self).__init__()
        self.model_6_band = model_6_band
        self.model_7_band = model_7_band
        self.fc = nn.Linear(3809, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)
        self.conv = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=2,stride=1)
        self.conv2 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=3,stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()

    def forward(self, input_6_band, input_7_band):
        output_6_band = self.model_6_band(input_6_band).output
        output_6_band = self.pool(output_6_band)
        output_6_band = self.conv(output_6_band)
        output_6_band = self.pool(output_6_band)
        output_6_band = self.conv2(output_6_band)
        output_6_band = self.flatten(output_6_band)
        output_7_band = self.model_7_band(**input_7_band)
        combined = torch.cat((output_6_band, output_7_band.logits), dim=1)
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

def run_model(fold,naming):
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
    processor_6_band = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model_6_band = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

    processor_7_band = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model_7_band = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

    model_prit = create_gfm()

    device = "cuda"
    # Define the combined model
    combined_model = CombinedModel(model_prit, model_7_band)
    combined_model = combined_model.cuda()
    optimizer = optim.Adam(combined_model.parameters(), lr=1e-6)
    criterion = nn.MSELoss().cuda()

    num_epochs = 20
    # Training loop

    train_metrics = []
    val_metrics = []
    for epoch in range(num_epochs):
        labellist,predlist = [],[]
        for first_6_bands, seventh_band, labels in tqdm(train_dataloader):
            # Prepare the images using the processors
            # inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
            inputs_6_band = first_6_bands
            inputs_7_band = processor_7_band(seventh_band, return_tensors="pt",do_rescale=False)

            inputs_6_band = inputs_6_band.to(torch.device(device))
            inputs_7_band = inputs_7_band.to(torch.device(device))

            # Forward pass
            predictions = combined_model(inputs_6_band, inputs_7_band)
            loss = criterion(predictions.squeeze(), labels.to(torch.device(device)))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with open(f"{naming}_train_loss{fold}.txt","a") as f:
                f.write(f"{loss}\n")
            
            labels,predictions = labels.cpu().detach().numpy().reshape((1,-1))[0], predictions.cpu().detach().numpy().reshape((1,-1))[0]

            labellist += list(labels)
            predlist += list(predictions)
        labellist, predlist = np.array(labellist),np.array(predlist)
        train_metrics.append([calc_score(labellist,predlist,'r2'),calc_score(labellist,predlist,'R2'),\
                                    calc_score(labellist,predlist,'mse'),calc_score(labellist,predlist,'rank')])
        # if epoch % 1 == 0 or epoch==num_epochs-1:
        if True:
            with torch.no_grad():
                labellist,predlist = [],[]
                for first_6_bands, seventh_band, labels in tqdm(val_dataloader):
                    # Prepare the images using the processors
                    # inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
                    inputs_6_band = first_6_bands
                    inputs_7_band = processor_7_band(seventh_band, return_tensors="pt",do_rescale=False)

                    inputs_6_band = inputs_6_band.to(torch.device(device))
                    inputs_7_band = inputs_7_band.to(torch.device(device))

                    predictions = combined_model(inputs_6_band, inputs_7_band)

                    loss = criterion(predictions.squeeze(), labels.to(torch.device(device)))
                    with open(f"{naming}_vall_loss{fold}.txt","a") as f:
                        f.write(f"{loss}\n")

                    labels,predictions = labels.cpu().detach().numpy().reshape((1,-1))[0], predictions.cpu().detach().numpy().reshape((1,-1))[0]
                
                    labellist += list(labels)
                    predlist += list(predictions)
                labellist, predlist = np.array(labellist),np.array(predlist)
                val_metrics.append([calc_score(labellist,predlist,'r2'),calc_score(labellist,predlist,'R2'),\
                                            calc_score(labellist,predlist,'mse'),calc_score(labellist,predlist,'rank')])

        print("epoch done:", epoch)
        print(val_metrics[-1],"\n")

    with open(f"{naming}_train_results{fold}.txt","w") as f:
        for i in range(len(train_metrics)):
            f.write(f"{train_metrics[i][0]}\t{train_metrics[i][1]}\t{train_metrics[i][2]}\t{train_metrics[i][3]}\n")
    with open(f"{naming}_val_results{fold}.txt","w") as f:
        for i in range(len(val_metrics)):
            f.write(f"{val_metrics[i][0]}\t{val_metrics[i][1]}\t{val_metrics[i][2]}\t{val_metrics[i][3]}\n")

    with torch.no_grad():
        labellist,predlist = [],[]
        for first_6_bands, seventh_band, labels in tqdm(test_dataloader):
            # Prepare the images using the processors
            # inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
            inputs_6_band = first_6_bands
            inputs_7_band = processor_7_band(seventh_band, return_tensors="pt",do_rescale=False)

            inputs_6_band = inputs_6_band.to(torch.device(device))
            inputs_7_band = inputs_7_band.to(torch.device(device))

            predictions = combined_model(inputs_6_band, inputs_7_band)

            labels,predictions = labels.cpu().detach().numpy().reshape((1,-1))[0], predictions.cpu().detach().numpy().reshape((1,-1))[0]
        
            labellist += list(labels)
            predlist += list(predictions)
        labellist, predlist = np.array(labellist),np.array(predlist)
        test_metrics = [calc_score(labellist,predlist,'r2'),calc_score(labellist,predlist,'R2'),\
                                    calc_score(labellist,predlist,'mse'),calc_score(labellist,predlist,'rank')]

    with open(f"{naming}_test_results{fold}.txt","w") as f:
        f.write(f"{test_metrics[0]}\t{test_metrics[1]}\t{test_metrics[2]}\t{test_metrics[3]}\n")

# Example usage
if __name__ == "__main__":
    # folds = ["A","B","C","D","E"]
    # # folds = ["A"]
    # for fold in folds:
    #     run_model(fold)

    parser = argparse.ArgumentParser()
    # paths
    parser.add_argument(
        '--fold', default='A',
        help='folds to run')
    
    parser.add_argument(
        '--name', default='def',
        help='name for this run')

    args = parser.parse_args()

    run_model(args.fold,args.name)
