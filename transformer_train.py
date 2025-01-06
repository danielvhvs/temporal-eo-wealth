import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms.functional import center_crop

# import terratorch # even though we don't use the import directly, we need it so that the models are available in the timm registry
# from terratorch.models import EncoderDecoderFactory
# from terratorch.datasets import HLSBands

# def create_gfm():
#     model_factory = EncoderDecoderFactory()

#     # Let's build a segmentation model
#     # Parameters prefixed with backbone_ get passed to the backbone
#     # Parameters prefixed with decoder_ get passed to the decoder
#     # Parameters prefixed with head_ get passed to the head

#     model = model_factory.build_model(task="segmentation",
#             backbone="prithvi_vit_100",
#             decoder="FCNDecoder",
#             backbone_bands=[
#                 HLSBands.BLUE,
#                 HLSBands.GREEN,
#                 HLSBands.RED,
#                 HLSBands.NIR_NARROW,
#                 HLSBands.SWIR_1,
#                 HLSBands.SWIR_2,
#             ],
#             necks=[{"name": "SelectIndices", "indices": [-1]},
#                 {"name": "ReshapeTokensToImage"}],
#             num_classes=4,
#             backbone_pretrained=True,
#             backbone_num_frames=1,
#             decoder_channels=128,
#             head_dropout=0.2
#         )
#     return model


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
        first_6_bands = image[:3, :, :]  # First 6 bands
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
        self.fc = nn.Linear(2000, hidden_dim)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, input_6_band, input_7_band):
        output_6_band = self.model_6_band(**input_6_band)
        output_7_band = self.model_7_band(**input_7_band)
        combined = torch.cat((output_6_band.logits, output_7_band.logits), dim=1)
        hidden = torch.relu(self.fc(combined))
        regression_output = self.regressor(hidden)
        return regression_output

# Example usage
if __name__ == "__main__":
    image_dir = "./images_saved/"  # Replace with the path to your images
    label_file = "./numpy_wealth.npy"  # Replace with the path to your labels file

    batch_size = 16  # Set your desired batch size here

    dataset = NumpyDataset(image_dir, label_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load the models and processor
    processor_6_band = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model_6_band = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")

    processor_7_band = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    model_7_band = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
    device = "cuda"
    # Define the combined model
    combined_model = CombinedModel(model_6_band, model_7_band)
    combined_model = combined_model.cuda()
    optimizer = optim.Adam(combined_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss().cuda()

    num_epochs = 20
    # Training loop
    for epoch in range(num_epochs):
        for first_6_bands, seventh_band, labels in dataloader:
            # Prepare the images using the processors
            inputs_6_band = processor_6_band(first_6_bands, return_tensors="pt",do_rescale=False)
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

            print("Loss:", loss.item())
