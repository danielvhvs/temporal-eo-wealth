from typing import Iterable
from glob import glob
from pprint import pprint
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

from batchers import batcher, tfrecord_paths_utils
from preprocessing.helper import (
    analyze_tfrecord_batch,
    per_band_mean_std,
    print_analysis_results)

REQUIRED_BANDS = [
    'BLUE', 'GREEN', 'LAT', 'LON', 'NIGHTLIGHTS', 'NIR', 'RED',
    'SWIR1', 'SWIR2']

BANDS_ORDER = [
    'BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'NIR',
    'DMSP', 'VIIRS']

DHS_PROCESSED_FOLDER = './dhs_tfrecords'


def calculate_mean_std(tfrecord_paths):
    '''Calculates and prints the per-band means and std-devs'''
    iter_init, batch_op = batcher.Batcher(
        tfrecord_files=tfrecord_paths,
        label_name=None,
        ls_bands='ms',
        nl_band='merge',
        batch_size=128,
        shuffle=False,
        augment=False,
        clipneg=False,
        normalize=None).get_batch()

    stats = analyze_tfrecord_batch(
        iter_init, batch_op, total_num_images=len(tfrecord_paths),
        nbands=len(BANDS_ORDER), k=10)
    means, stds = per_band_mean_std(stats=stats, band_order=BANDS_ORDER)

    print('Means:')
    pprint(means)
    print()

    print('Std Devs:')
    pprint(stds)

    print('\n========== Additional Per-band Statistics ==========\n')
    print_analysis_results(stats, BANDS_ORDER)

dataset = 'DHS_incountry_A'
paths = tfrecord_paths_utils.dhs_incountry(dataset, splits=['train', 'val'])["train"]
print(paths) 
calculate_mean_std(paths)