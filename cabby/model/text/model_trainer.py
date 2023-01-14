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

"""Model framework for text and S2Cellids matching.
Example command line call:
$ bazel-bin/cabby/model/text/model_trainer \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --dataset_dir ~/model/dataset/pittsburgh \
  --region Pittsburgh \
  --s2_level 12 \
  --output_dir ~/tmp/output/\
  --train_batch_size 32 \
  --test_batch_size 32 \
For infer:
$ bazel-bin/cabby/model/text/model_trainer \
  --data_dir ~/data/wikigeo/pittsburgh  \
  --dataset_dir ~/model/dataset/pittsburgh \
  --region Pittsburgh \
  --s2_level 12 \
  --test_batch_size 32 \
  --infer_only True \
  --model_path ~/tmp/model/ \
  --output_dir ~/tmp/output/\
  --task RVS
"""

from absl import app
from absl import flags

from absl import logging
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW


from cabby.evals import utils as eu
from cabby.model.text import train
from cabby.model import dataset_item
from cabby.model.text import models
from cabby.model import datasets
from cabby.model import util
from cabby.geo import regions

# Configure parameters.

DATA_DIR = '/home/nlp/itaimond1/caby/cabby/model/text/dataSamples/human_points'
DATASET_DIR = '/home/nlp/itaimond1/caby/cabby/model/text/dataset_dir'
REGION = 'Tel Aviv'
S2_LEVEL = '13'
MODEL = 'Dual-Encoder-Bert'
GRAPH_EMBEDDING = ''
INFER_ONLY = False
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 16
DEV_BATCH_SIZE = 16
MODEL_PATH = ''
LEARNING_RATE = 0.01
NUM_EPOCHS = 5
OUTPUT_DIR = '/home/nlp/itaimond1/caby/cabby/model/text/dataset_dir/output'


def main():
  print("Start training")

  if not os.path.exists(DATASET_DIR):
    sys.exit("Dataset path doesn't exist: {}.".format(DATASET_DIR))

  dataset_model_path = os.path.join(DATASET_DIR, MODEL)
  dataset_path = os.path.join(dataset_model_path, str(S2_LEVEL))
  dataset_init = datasets.HumanDataset

  if os.path.exists(dataset_path):
    dataset_text = dataset_item.TextGeoDataset.load(
      dataset_dir=DATASET_DIR,
      model_type=str(MODEL),
      s2_level=S2_LEVEL
    )

  else:
    dataset = dataset_init(
      data_dir=DATA_DIR,
      region=REGION,
      s2level=int(S2_LEVEL),
      model_type=MODEL,
      n_fixed_points=4,
      graph_embed_path=GRAPH_EMBEDDING)

    if not os.path.exists(dataset_model_path):
      os.mkdir(dataset_model_path)


    print("Preprocessing dataset")
    dataset_text = dataset.create_dataset(
      infer_only=INFER_ONLY,
      is_dist=False,
      far_cell_dist=2000,
    )

    dataset_item.TextGeoDataset.save(
      dataset_text=dataset_text,
      dataset_path=dataset_path,
      graph_embed_size=dataset.graph_embed_size)


    print("Saveed dataset")

  torch.cuda.empty_cache()
  n_cells = len(dataset_text.unique_cellids)

  train_loader = None
  valid_loader = None
  if not INFER_ONLY:
    train_loader = DataLoader(
      dataset_text.train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
      dataset_text.valid, batch_size=TEST_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
      dataset_text.test, batch_size=DEV_BATCH_SIZE, shuffle=False)

  device =  torch.device('cpu')

  if 'Dual-Encoder' in MODEL:
    run_model = models.DualEncoder(device=device)
  else:
    run_model = models.ClassificationModel(n_cells, device=device)

  # if MODEL_PATH is not None:
  #   if not os.path.exists(MODEL_PATH):
  #     sys.exit(f"The model's path does not exists: {MODEL_PATH}")
  #   util.load_checkpoint(
  #     load_path=MODEL_PATH, model=run_model, device=device)
  if torch.cuda.device_count() > 1:
    run_model = nn.DataParallel(run_model).module

  run_model.to(device)

  optimizer = torch.optim.Adam(run_model.parameters(), lr=LEARNING_RATE)

  trainer = train.Trainer(
    model=run_model,
    device=device,
    num_epochs=NUM_EPOCHS,
    optimizer=optimizer,
    train_loader=train_loader,
    valid_loader=valid_loader,
    test_loader=test_loader,
    unique_cells=dataset_text.unique_cellids,
    file_path=OUTPUT_DIR,
    cells_tensor=torch.tensor(dataset_text.unique_cellids_binary),
    label_to_cellid=dataset_text.label_to_cellid,
    is_single_sample_train=False,
  )
  if INFER_ONLY:
    logging.info("Starting to infer model.")
    valid_loss, predictions, true_vals, true_points, pred_points = (
      trainer.evaluate(validation_set=False))

    util.save_metrics_last_only(
      trainer.metrics_path,
      true_points,
      pred_points)

    evaluator = eu.Evaluator()
    error_distances = evaluator.get_error_distances(trainer.metrics_path)
    evaluator.compute_metrics(error_distances)

  else:
    logging.info("Starting to train model.")
    print("Starting to train model.")
    trainer.train_model()


if __name__ == '__main__':
  main()
