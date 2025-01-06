from pprint import pprint
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../')
from batchers import batcher, dataset_constants
from preprocessing_helper import (
    get_first_feature_map,
    get_feature_types,
    print_scalar_values,
    analyze_tfrecord_batch,
    print_analysis_results)

import argparse
import json
import os
from pprint import pprint
import time
from typing import Any, Dict, List, Optional

from batchers import batcher, tfrecord_paths_utils
from models.resnet_model import Hyperspectral_Resnet
from utils.run import get_full_experiment_name
from utils.trainer import RegressionTrainer

import numpy as np
import tensorflow as tf

BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'NIR', 'NIGHTLIGHTS']
BAND_ORDER_NLSPLIT = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'NIR', 'DMSP', 'VIIRS']

dataset = 'DHS_incountry_A'
DATASET_NAME ='DHS'

MEANS = dataset_constants.MEANS_DICT[DATASET_NAME]
STD_DEVS = dataset_constants.STD_DEVS_DICT[DATASET_NAME]

def year_to_nltype(year):
    return 'DMSP' if year < 2012 else 'VIIRS'

def band_keys_for_year(band_keys, year):
    '''
    Args
    - band_keys: list of str, including 'NIGHTLIGHTS'
    - year: numeric

    Returns
    - new_band_keys: copy of band_keys with 'NIGHTLIGHTS' replaced by 'DMSP' or 'VIIRS'
    '''
    new_band_keys = list(band_keys) # make a local copy
    new_band_keys[band_keys.index('NIGHTLIGHTS')] = year_to_nltype(year)
    return new_band_keys

def extract_inout(feature_map):
    year = int(feature_map['year'].float_list.value[0])
    band_keys_nl = band_keys_for_year(BAND_ORDER, year)
    img_normalized = []
    for b_idx, b_name in enumerate(BAND_ORDER):
        band = np.asarray(feature_map[b_name].float_list.value, dtype=np.float32).reshape(255, 255)
        b = band_keys_nl[b_idx]
        band = (band - MEANS[b]) / STD_DEVS[b]
        img_normalized.append(band)
    img_normalized = np.stack(img_normalized, axis=2)
    return img_normalized

paths = tfrecord_paths_utils.dhs_incountry(dataset, splits=['all'])["all"]
wealth_list = []
from tqdm import tqdm
for n in tqdm(range(len(paths))):
    tfrecord_path = paths[n]
    feature_map = get_first_feature_map(tfrecord_path)
    if n > 17000:
        images = extract_inout(feature_map)
        np.save(f"./images_saved/numpy_images{n}.npy",images)
    with open(f"./wealth_text.txt","a") as f:
        ww = ((feature_map['wealthpooled']).float_list.value)[0]
        f.write(f"{ww}\n")
        f.close()
    wealth_list.append(ww)

np.save("./numpy_wealth.npy",np.array(wealth_list))

# # Load model directly
# from transformers import AutoImageProcessor, AutoModelForImageClassification

# processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
# model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")