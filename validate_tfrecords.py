from __future__ import annotations

from collections.abc import Iterable
from glob import glob
from pprint import pprint
import os
from typing import Optional

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
    'SWIR1', 'SWIR2', 'TEMP1']

BANDS_ORDER = [
    'BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR',
    'DMSP', 'VIIRS']


LSMS_PROCESSED_FOLDER = 'data/lsms_tfrecords2'

def validate_individual_tfrecords(tfrecord_paths: Iterable[str],
                                  csv_path: str,
                                  label_name: Optional[str] = None) -> None:
    '''
    Args
    - tfrecord_paths: list of str, paths to individual TFRecord files
        in the same order as in the CSV
    - csv_path: str, path to CSV file with columns ['lat', 'lon', 'wealthpooled', 'year']
    '''
    df = pd.read_csv(csv_path, float_precision='high', index_col=False)
    iter_init, batch_op = batcher.Batcher(
        tfrecord_files=tfrecord_paths,
        label_name=label_name,
        ls_bands=None,
        nl_band=None,
        batch_size=128,
        shuffle=False,
        augment=False,
        clipneg=False,
        normalize=None).get_batch()

    locs, years = [], []
    if label_name is not None:
        labels = []

    num_processed = 0
    with tf.Session() as sess:
        sess.run(iter_init)
        while True:
            try:
                if label_name is not None:
                    batch_np = sess.run((batch_op['locs'], batch_op['years'], batch_op['labels']))
                    labels.append(batch_np[2])
                else:
                    batch_np = sess.run((batch_op['locs'], batch_op['years']))
                locs.append(batch_np[0])
                years.append(batch_np[1])
                num_processed += len(batch_np[0])
                print(f'\rProcessed {num_processed} images', end='')
            except tf.errors.OutOfRangeError:
                break
    print()

    locs = np.concatenate(locs)
    years = np.concatenate(years)
    assert (locs == df[['lat', 'lon']].to_numpy(dtype=np.float32)).all()
    assert (years == df['year'].to_numpy(dtype=np.float32)).all()
    if label_name is not None:
        labels = np.concatenate(labels)
        assert (labels == df['wealthpooled'].to_numpy(dtype=np.float32)).all()

validate_individual_tfrecords(
    tfrecord_paths=tfrecord_paths_utils.lsms(),
    csv_path='data/lsms_clusters.csv')