# coding=utf-8
# Copyright 2020 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict

from absl import logging
import numpy as np
import os
import pandas as pd
from sklearn.utils import shuffle
import torch

from cabby.geo import util as gutil
from cabby.model.text import util 

from cabby.geo import regions
from cabby.model.text.dual_encoder import dataset_item




def create_dataset(
                  data_dir: str, 
                  region: str, 
                  s2level: int, 
                  infer_only: bool = False
) -> dataset_item.TextGeoDataset:
  '''Loads data and creates datasets and train, validate and test sets.
  Arguments:
    data_dir: The directory of the data.
    region: The region of the data.
    s2level: The s2level of the cells.
  Returns:
    The train, validate and test sets and the dictionary of labels to cellids.
  '''
  ds = pd.read_json(os.path.join(data_dir, 'dataset.json'))
  ds['instructions'] = ds.groupby(
    ['id'])['instruction'].transform(lambda x: ' '.join(x))

  ds = ds.drop_duplicates(subset='id', keep="last")

  columns_keep = ds.columns.difference(
    ['map', 'id', 'instructions', 'end_point', 'start_point'])
  ds.drop(columns_keep, 1, inplace=True)

  ds = shuffle(ds)
  ds.reset_index(inplace=True, drop=True)

  dataset_size = ds.shape[0]
  logging.info(f"Size of dataset: {ds.shape[0]}")
  train_size = round(dataset_size*80/100)
  valid_size = round(dataset_size*10/100)

  train_ds = ds.iloc[:train_size]
  valid_ds = ds.iloc[train_size:train_size+valid_size]
  test_ds = ds.iloc[train_size+valid_size:]

  logging.info(train_ds.head(30))


  # Get labels.
  map_1 = regions.get_region("RUN-map1")
  map_2 = regions.get_region("RUN-map2")
  map_3 = regions.get_region("RUN-map3")
  # map_polygon = map_1.polygon.union(map_2.polygon).union(map_3.polygon)
  logging.info(map_1.polygon.wkt)
  logging.info(map_2.polygon.wkt)
  logging.info(map_3.polygon.wkt)

  unique_cellid_map_1 = gutil.cellids_from_polygon(map_1.polygon, s2level)
  unique_cellid_map_2 = gutil.cellids_from_polygon(map_2.polygon, s2level)
  unique_cellid_map_3 = gutil.cellids_from_polygon(map_3.polygon, s2level)

  unique_cellid = (
    unique_cellid_map_1 + unique_cellid_map_2 + unique_cellid_map_3)
  label_to_cellid = {idx: cellid for idx, cellid in enumerate(unique_cellid)}
  cellid_to_label = {cellid: idx for idx, cellid in enumerate(unique_cellid)}

  points = gutil.get_centers_from_s2cellids(unique_cellid)

  unique_cells_df = pd.DataFrame({'point': points, 'cellid': unique_cellid})
  
  unique_cells_df['far'] = unique_cells_df.point.apply(
      lambda x: gutil.far_cellid(x, unique_cells_df))

  vec_cells = util.binary_representation(unique_cells_df.cellid.to_numpy(), 
  dim = dataset_item.CELLID_DIM)
  tens_cells = torch.tensor(vec_cells)

  # Create RVS dataset.
  train_dataset = None
  val_dataset = None
  logging.info("Starting to create the splits")
  if infer_only == False:
    train_dataset = dataset_item.TextGeoSplit(
      train_ds, s2level, unique_cells_df, cellid_to_label)
    logging.info(
      f"Finished to create the train-set with {len(train_dataset)} samples")
    val_dataset = dataset_item.TextGeoSplit(
      valid_ds, s2level, unique_cells_df, cellid_to_label)
    logging.info(
      f"Finished to create the valid-set with {len(val_dataset)} samples")
  test_dataset = dataset_item.TextGeoSplit(
    test_ds, s2level, unique_cells_df, cellid_to_label)
  logging.info(
    f"Finished to create the test-set with {len(test_dataset)} samples")

  return dataset_item.TextGeoDataset.from_TextGeoSplit(
    train_dataset, val_dataset, test_dataset, np.array(unique_cellid), 
    tens_cells, label_to_cellid)

