from __future__ import annotations

from collections.abc import Iterable
from glob import glob
from pprint import pprint
import os
from typing import Optional

import re
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

LSMS_EXPORT_FOLDER = './lsms_tfrecords_raw3'

LSMS_PROCESSED_FOLDER = './lsms_tfrecords2'

def process_dataset(csv_path: str, input_dir: str, processed_dir: str) -> None:
    '''
    Args
    - csv_path: str, path to CSV of DHS or LSMS clusters
    - input_dir: str, path to TFRecords exported from Google Earth Engine
    - processed_dir: str, folder where to save processed TFRecords
    '''
    df = pd.read_csv(csv_path, float_precision='high', index_col=False)
    surveys = list(df.groupby(['country', 'year']).groups.keys())  # (country, year) tuples

    for country, year in surveys:
        country_year = f'{country}_{year}'
        print('Processing:', country_year)

        tfrecord_paths = glob(os.path.join(input_dir, country_year + '*'))
        out_dir = os.path.join(processed_dir, country_year)
        os.makedirs(out_dir, exist_ok=True)
        subset_df = df[(df['country'] == country) & (df['year'] == year)].reset_index(drop=True)
        validate_and_split_tfrecords(
            tfrecord_paths=tfrecord_paths, out_dir=out_dir, df=subset_df)


def validate_and_split_tfrecords(
        tfrecord_paths: Iterable[str],
        out_dir: str,
        df: pd.DataFrame
        ) -> None:
    '''Validates and splits a list of exported TFRecord files (for a
    given country-year survey) into individual TFrecords, one per cluster.

    "Validating" a TFRecord comprises of 2 parts
    1) verifying that it contains the required bands
    2) verifying that its other features match the values from the dataset CSV

    Args
    - tfrecord_paths: list of str, paths to exported TFRecords files
    - out_dir: str, path to dir to save processed individual TFRecords
    - df: pd.DataFrame, index is sequential and starts at 0
    '''
    # Create an iterator over the TFRecords file. The iterator yields
    # the binary representations of Example messages as strings.
    options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)

    # cast float64 => float32 and str => bytes
    for col in df.columns:
        if df[col].dtype == np.float64:
            df[col] = df[col].astype(np.float32)
        elif df[col].dtype == object:  # pandas uses 'object' type for str
            df[col] = df[col].astype(bytes)

    i = 0
    progbar = tqdm(total=len(df))

    for tfrecord_path in tfrecord_paths:
        i = int(re.findall('_\d+',tfrecord_path)[-1][1:])
        iterator = tf.io.tf_record_iterator(tfrecord_path, options=options)
        for idx,record_str in enumerate(iterator):
            if idx>0:
                print("herer")
            # parse into an actual Example message
            ex = tf.train.Example.FromString(record_str)
            feature_map = ex.features.feature

            # verify required bands exist
            for band in REQUIRED_BANDS:
                if not band in feature_map:
                    print(f'Band "{band}" not in record {i} of {tfrecord_path}')

            # compare feature map values against CSV values
            csv_feats = df.loc[i, :].to_dict()
            for col, val in csv_feats.items():
                ft_type = feature_map[col].WhichOneof('kind')
                ex_val = feature_map[col].__getattribute__(ft_type).value[0]
                if not (val == ex_val):
                    x = f'Expected {col}={val}, but found {ex_val} instead for record {i} of {tfrecord_path} {csv_feats}'
                    print(x)

            # serialize to string and write to file
            out_path = os.path.join(out_dir, f'{i:05d}.tfrecord.gz')  # all surveys have < 1e6 clusters
            with tf.io.TFRecordWriter(out_path, options=options) as writer:
                writer.write(ex.SerializeToString())

            progbar.update(1)
    progbar.close()

process_dataset(
    csv_path='data/lsms_clusters.csv',
    input_dir=LSMS_EXPORT_FOLDER,
    processed_dir=LSMS_PROCESSED_FOLDER)