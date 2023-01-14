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
'''Basic classes and functions for Wikigeo items.'''

from xmlrpc.client import Boolean
from absl import logging
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import numpy as np
import os
import pandas as pd
import re
from shapely.geometry.point import Point
from shapely.geometry import box, mapping, LineString
import sys
import swifter
from typing import Any, Dict, Text, Tuple
import torch

import attr

from cabby.geo import util as gutil
from cabby.model import util
from cabby.geo.util import far_cellid

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@attr.s
class TextGeoDataset:
  """Construct a RVSPath sample.
  `train` is the train split.
  `valid` is the valid split.
  `test` is the test split.
  `unique_cellids` is the unique S2Cells.
  `unique_cellids_binary`  is the binary tensor of the unique S2Cells.
  `label_to_cellid` is the dictionary mapping labels to cellids.
  """
  train: Any = attr.ib()
  valid: Any = attr.ib()
  test: Any = attr.ib()
  unique_cellids: np.ndarray = attr.ib()
  unique_cellids_binary: torch.tensor = attr.ib()
  label_to_cellid: Dict[int, int] = attr.ib()
  coord_to_cellid: Dict[str, int] = attr.ib()
  graph_embed_size: int = attr.ib()

  @classmethod
  def from_TextGeoSplit(cls, train, valid, test, unique_cellids,
                        unique_cellids_binary, label_to_cellid, coord_to_cellid, graph_embed_size):
    """Construct a TextGeoDataset."""
    return TextGeoDataset(
      train,
      valid,
      test,
      unique_cellids,
      unique_cellids_binary,
      label_to_cellid,
      coord_to_cellid,
      graph_embed_size,
    )

  @classmethod
  def load(cls, dataset_dir: Text, model_type: Text = None,
           s2_level: Text = None):
    if model_type:
      dataset_dir = os.path.join(dataset_dir, str(model_type))
    if s2_level:
      dataset_dir = os.path.join(dataset_dir, str(s2_level))

    train_path_dataset = os.path.join(dataset_dir, 'train.pth')
    valid_path_dataset = os.path.join(dataset_dir, 'valid.pth')
    test_path_dataset = os.path.join(dataset_dir, 'test.pth')
    unique_cellid_path = os.path.join(dataset_dir, "unique_cellid.npy")
    tensor_cellid_path = os.path.join(dataset_dir, "tensor_cellid.pth")
    label_to_cellid_path = os.path.join(dataset_dir, "label_to_cellid.npy")
    coord_to_cellid_path = os.path.join(dataset_dir, "coord_to_cellid.npy")
    graph_embed_size_path = os.path.join(dataset_dir, "graph_embed_size.npy")

    logging.info("Loading dataset from <== {}.".format(dataset_dir))
    train_dataset = torch.load(train_path_dataset)
    valid_dataset = torch.load(valid_path_dataset)
    test_dataset = torch.load(test_path_dataset)
    logging.info(f"Size of train set: {len(train_dataset)}" +
                 f", Size of validation set: {len(valid_dataset)}, Size of test set: {len(test_dataset)}")

    unique_cellid = np.load(unique_cellid_path, allow_pickle='TRUE')
    label_to_cellid = np.load(
      label_to_cellid_path, allow_pickle='TRUE').item()
    tens_cells = torch.load(tensor_cellid_path)
    coord_to_cellid = np.load(coord_to_cellid_path, allow_pickle='TRUE').item()
    graph_embed_size = np.load(graph_embed_size_path, allow_pickle='TRUE')
    logging.info(f"Loaded dataset with graph embedding size {graph_embed_size}")

    dataset_text = TextGeoDataset(
      train_dataset, valid_dataset, test_dataset,
      unique_cellid, tens_cells, label_to_cellid, coord_to_cellid, graph_embed_size)

    return dataset_text

  @classmethod
  def save(cls, dataset_text: Any, dataset_path: Text,
           graph_embed_size: int):
    os.mkdir(dataset_path)

    train_path_dataset = os.path.join(dataset_path, 'train.pth')
    valid_path_dataset = os.path.join(dataset_path, 'valid.pth')
    test_path_dataset = os.path.join(dataset_path, 'test.pth')
    unique_cellid_path = os.path.join(dataset_path, "unique_cellid.npy")
    tensor_cellid_path = os.path.join(dataset_path, "tensor_cellid.pth")
    label_to_cellid_path = os.path.join(dataset_path, "label_to_cellid.npy")
    coord_to_cellid_path = os.path.join(dataset_path, "coord_to_cellid.npy")
    graph_embed_size_path = os.path.join(dataset_path, "graph_embed_size.npy")

    torch.save(dataset_text.train, train_path_dataset)
    torch.save(dataset_text.valid, valid_path_dataset)
    torch.save(dataset_text.test, test_path_dataset)
    np.save(unique_cellid_path, dataset_text.unique_cellids)
    torch.save(dataset_text.unique_cellids_binary, tensor_cellid_path)
    np.save(label_to_cellid_path, dataset_text.label_to_cellid)
    np.save(coord_to_cellid_path, dataset_text.coord_to_cellid)
    np.save(graph_embed_size_path, graph_embed_size)
    logging.info("Saved data to ==> {}.".format(dataset_path))


class TextGeoSplit(torch.utils.data.Dataset):
  """A split of of the RUN dataset.
  `points`: The ground true end-points of the samples.
  `labels`: The ground true label of the cellid.
  `cellids`: The ground truth S2Cell id.
  `neighbor_cells`: One neighbor cell id of the ground truth S2Cell id.
  `far_cells`: One far away cell id (in the region defined) of the ground truth
  'dprob': Gamma distribution probability.
  S2Cell id.
  """

  def __init__(self, text_tokenizer, s2_tokenizer, data: pd.DataFrame, s2level: int,
               unique_cells_df: pd.DataFrame, cellid_to_label: Dict[int, int],
               model_type: str, dprob: util.DistanceProbability,
               cellid_to_coord,
               graph_embed_file: Any = None, is_dist: Boolean = False
               ):

    self.text_tokenizer = text_tokenizer
    self.s2_tokenizer = s2_tokenizer
    self.cellid_to_label = cellid_to_label
    self.cellid_to_coord = cellid_to_coord
    self.s2level = s2level
    self.is_dist = is_dist
    self.model_type = model_type
    self.graph_embed_file = graph_embed_file

    print("hello")
    data = data.assign(end_point=data.end_point)

    data['cellid'] = data.end_point.apply(lambda x: gutil.cellid_from_point(x, s2level))

    data['neighbor_cells'] = data.cellid.apply(lambda x: gutil.neighbor_cellid(x, unique_cells_df.cellid.tolist()))
    # Tokenize instructions.

    self.instruction_list = data.instructions.tolist()

    self.encodings = self.text_tokenizer(data.instructions.tolist(), truncation=True, padding=True,
                                         add_special_tokens=True, max_length=200)

    data['far_cells'] = data.end_point.apply(lambda end_point: far_cellid(end_point, unique_cells_df, 300))

    cellids_array = np.array(data.cellid.tolist())
    neighbor_cells_array = np.array(data.neighbor_cells.tolist())
    far_cells_array = np.array(data.far_cells.tolist())

    self.end_point = data.end_point.apply(lambda x: gutil.tuple_from_point(x)).tolist()

    self.labels = data.cellid.apply(lambda x: cellid_to_label[x] if x in cellid_to_label else 0).tolist()
    data['labels'] = self.labels

 

    self.cellids = self.s2_tokenizer(cellids_array)

    self.neighbor_cells = self.s2_tokenizer(neighbor_cells_array)

    self.far_cells = self.s2_tokenizer(far_cells_array)


    if graph_embed_file:
      self.graph_embed_end = data['cellid'].apply(
        lambda cell: util.get_valid_graph_embed(self.graph_embed_file, str(cell)))
      self.graph_embed_start = self.start_cells.apply(
        lambda cell: util.get_valid_graph_embed(self.graph_embed_file, str(cell)))

      data['landmarks_cells'] = data.landmarks.apply(
        lambda l: [gutil.cellid_from_point(x, self.s2level) for x in l])

      self.graph_embed_landmarks = data.landmarks_cells.apply(
        lambda l: [util.get_valid_graph_embed(
          self.graph_embed_file, str(cell)) for cell in l])

      self.start_embed_text_input_list = [
        str(i).replace(':', f': Start at {str(s)}.') for s, i in zip(
          self.graph_embed_start, data.instructions.tolist())]

    else:
      self.graph_embed_start = np.zeros(len(self.cellids))

    self.landmarks_dist_raw = []



    del self.graph_embed_file
    # del self.start_point_cells

    del self.text_tokenizer

  def get_cell_to_lablel(self, list_cells):
    if isinstance(list_cells[0], list):
      labels = []
      for c_list in list_cells:
        list_lables = []
        for c in c_list:
          list_lables.append(self.cellid_to_coord[c])

        labels.append('; '.join(list_lables))

    else:
      labels = [
        util.get_valid_cell_label(self.cellid_to_coord, int(c)) for c in list_cells]

    return labels

  def __getitem__(self, idx: int):
    '''Supports indexing such that TextGeoDataset[i] can be used to get
    i-th sample.
    Arguments:
      idx: The index for which a sample from the dataset will be returned.
    Returns:
      A single sample including text, the correct cellid, a neighbor cellid,
      a far cellid, a point of the cellid and the label of the cellid.
    '''

    neighbor_cells = torch.tensor(self.neighbor_cells[idx])
    far_cells = torch.tensor(self.far_cells[idx])
    cellid = torch.tensor(self.cellids[idx])
    # encoded_cell = torch.tensor(self.encoded_cells[idx])
    end_point = torch.tensor(self.end_point[idx])
    text_input = {key: torch.tensor(val[idx])
                    for key, val in self.encodings.items()}

    graph_embed_start = self.graph_embed_start[idx]
    sample = {'text': text_input, 'cellid': cellid,
              'neighbor_cells': neighbor_cells, 'far_cells': far_cells,
              'end_point': end_point,
              # 'encoded_cell' : encoded_cell,
              'graph_embed_start': graph_embed_start
              }

    return sample

  def __len__(self):
    return len(self.cellids)
