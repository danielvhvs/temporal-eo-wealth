from glob import glob
import os
from pprint import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# sys.path.append('C:\Users\danie\Desktop\urban\africa_poverty')
from batchers import batcher, dataset_constants
from preprocessing.helper import get_first_feature_map, get_feature_types
from utils.run import run_epoch
from utils.plot import plot_image_by_band

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

DATASET_NAME = 'LSMS'
BAND_KEYS = ['RED', 'BLUE', 'GREEN', 'SWIR1', 'TEMP1', 'SWIR2', 'NIR', 'NIGHTLIGHTS']

MEANS = dataset_constants.MEANS_DICT[DATASET_NAME]
STD_DEVS = dataset_constants.STD_DEVS_DICT[DATASET_NAME]

out_root_dir = './lsms'

# file that Anne sent to me
LSMS_CSV_PATH = './data/lsms_labels_agg.csv'
lsms_df = pd.read_csv(LSMS_CSV_PATH, float_precision='high')
lsms_df['lat'] = lsms_df['lat'].astype(np.float32)
lsms_df['lon'] = lsms_df['lon'].astype(np.float32)
lsms_df['index'] = lsms_df['index'].astype(np.float32)
print(lsms_df.dtypes)

cid_to_c = {
    'et': 'ethiopia',
    'mw': 'malawi',
    'ng': 'nigeria',
    'tz': 'tanzania',
    'ug': 'uganda',
}
c_to_cid = {c: cid for c, cid in cid_to_c.items()}

# (lat, lon) => (cid, [list of years])
loc_info = {}

for name, group in lsms_df.groupby(['lat', 'lon']):
    loc = tuple(np.float32(name))
    cids = group['country'].unique()
    assert len(cids) == 1
    cid = cids[0]
    years = sorted(group['year'].unique())
    loc_info[loc] = (cid, years)

locs_df = lsms_df.groupby(['lat', 'lon', 'country']).size().index.to_frame(index=False)

def get_lat(lon, cid):
    rows = locs_df.loc[(locs_df['lon'] == lon) & (locs_df['country'] == cid)]
    print(len(rows))
    assert len(rows) == 1
    return rows.iloc[0]['lat']

tfrecord_path = './lsms_tfrecords_raw/ethiopia_2011_00.tfrecord.gz'
# tfrecord_path = 'lsms_orig/lx_median_2003-05_lsmslocs_ee_export.tfrecord.gz'
feature_map = get_first_feature_map(tfrecord_path)
feature_types = get_feature_types(feature_map)

print(f'TFRecord path: {tfrecord_path}')
print('Features and types:')
pprint(feature_types)

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

def plot_single_img(feature_map):
    feature_map = get_first_feature_map(tfrecord_path)
    lon = np.float32(feature_map['lon'].float_list.value[0])
    cid = feature_map['country'].bytes_list.value[0].decode()
    lat = get_lat(lon=lon, cid=cid)
    country = cid_to_c[cid]
    fig_title = f'Location: ({lat:.6f}, {lon:.6f}), {country}'
    print(fig_title)
    year = int(feature_map['year'].float_list.value[0])
    print(year)

    # choose 'DMSP' or 'VIIRS' for nightlights band name based on year
    band_keys_nl = band_keys_for_year(BAND_KEYS, year)

    img_normalized = []
    for b_idx, b_name in enumerate(BAND_KEYS):
        band = np.array(feature_map[b_name].float_list.value, dtype=np.float32).reshape(255, 255)
        b = band_keys_nl[b_idx]
        band = (band - MEANS[b]) / STD_DEVS[b]
        img_normalized.append(band)
    img_normalized = np.stack(img_normalized, axis=2)
    plot_image_by_band(img=img_normalized, band_order=band_keys_nl, nrows=3, title='',
                       rgb='add', colorbar=True)


REQUIRED_KEYS = [
    'BLUE', 'GREEN', 'LAT', 'LON', 'NIGHTLIGHTS', 'NIR', 'RED', 'SWIR1', 'SWIR2', 'TEMP1',
    'lon', 'year', 'index', 'ea_id', 'country'
]

NON_NEGATIVE_BANDS = ['RED', 'BLUE', 'GREEN', 'NIGHTLIGHTS', 'NIR', 'SWIR1', 'SWIR2', 'TEMP1']

def validate_example(feature_map, record_num, year_range):
    '''
    Args
    - feature_map: protobuf map from feature name strings to Feature
    - record_num: int
    - year_range: tuple (int, int), feature_map['year'] should be within year_range inclusive

    Returns:
    - False if example is invalid
    - (lat, lon, year) if example is valid
    '''
    def print_info():
        info = {}
        if 'lon' in feature_map:
            info['lon'] = np.float32(feature_map['lon'].float_list.value[0])
        if 'year' in feature_map:
            info['year'] = int(feature_map['year'].float_list.value[0])
        print(f'  {info}')  # indented print

    missing_req_keys = [key for key in REQUIRED_KEYS if key not in feature_map]
    if len(missing_req_keys) > 0:
        print(f'Record {record_num} missing required keys: {missing_req_keys}')
        print_info()
        return False

    year = int(feature_map['year'].float_list.value[0])
    if (year < year_range[0]) or (year > year_range[1]):
        print(f'Record {record_num} has invalid year {year}')
        print_info()
        return False

    lon = np.float32(feature_map['lon'].float_list.value[0])
    cid = feature_map['country'].bytes_list.value[0].decode()
    lat = get_lat(lon=lon, cid=cid)

    img_lon = np.float32(np.asarray(feature_map['LON'].float_list.value).reshape(255, 255)[127, 127])
    img_lat = np.float32(np.asarray(feature_map['LAT'].float_list.value).reshape(255, 255)[127, 127])

    if abs(lon - img_lon) > 1e-3:
        print(f'Record {record_num} contains mismatch: "lon"={lon}, "LON"={img_lon}')
        print_info()
        return False
    if abs(lat - img_lat) > 1e-3:
        print(f'Record {record_num} contains invalid lat. "lat"={lat}, "LAT"={img_lat}')
        print_info()
        return False
    feature_map['lat'].float_list.value.append(lat)

    negative_bands = [band for band in NON_NEGATIVE_BANDS if contains_neg(feature_map, band)]
    if len(negative_bands) > 0:
        print(f'Record {record_num} contains negative bands: {negative_bands}')
        for band in negative_bands:
            count = np.sum(np.asarray(feature_map[band].float_list.value) < 0)
            min_val = np.float32(np.min(feature_map[band].float_list.value))
            print(f'  Band "{band}" - count: {count}, min value: {min_val}')
        print_info()

    # rename 'index' => 'wealthpooled'
    wealthpooled = feature_map.pop('index').float_list.value[0]
    feature_map['wealthpooled'].float_list.value.append(wealthpooled)

    return (lat, lon, year)

def contains_neg(feature_map, band):
    '''
    Args
    - feature_map
    - band: str, key for a float_list feature in feature_map

    Returns: bool, True iff feature_map[band] contains negative values
    '''
    arr = np.asarray(feature_map[band].float_list.value)
    return np.any(arr < 0)

def process_tfrecord(tfrecord_path, out_root_dir, year_range):
    '''
    Args
    - tfrecord_path: str, path to TFrecord file
    - out_root_dir: str, path to dir to save processed individual TFRecords
    - year_range: tuple (int, int), start and end year (inclusive) of the imagery contained in the tfrecord
    '''
    # Create an iterator over the TFRecords file. The iterator yields
    # the binary representations of Example messages as strings.
    options = tf.io.TFRecordOptions(tf.io.TFRecordCompressionType.GZIP)
    iterator = tf.io.tf_record_iterator(tfrecord_path, options=options)

    num_good_records = 0
    for record_idx, record_str in enumerate(iterator):
        # parse the binary string
        ex = tf.train.Example.FromString(record_str)  # parse into an actual Example message
        features = ex.features  # get the Features message within the Example
        feature_map = features.feature  # get the mapping from feature name strings to Feature

        is_valid_record = validate_example(feature_map, record_idx, year_range)

        if is_valid_record != False:
            lat, lon, year = is_valid_record

            cid, years = loc_info[(lat, lon)]
            assert year in years
            country = cid_to_c[cid]

            country_year = f'{country}_{year}'
            out_dir = os.path.join(out_root_dir, country_year)
            out_path = os.path.join(out_dir, f'{num_good_records}.tfrecord.gz')

            # serialize to string and write to file
            os.makedirs(out_dir, exist_ok=True)
            with tf.io.TFRecordWriter(out_path, options=options) as writer:
                writer.write(ex.SerializeToString())
            num_good_records += 1

        if (record_idx + 1) % 50 == 0:
            print(f'Finished validating {record_idx + 1} records')

    print('Finished validating {} records: {} good, {} bad'.format(
        record_idx + 1, num_good_records, record_idx + 1 - num_good_records))