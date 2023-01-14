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

from absl import logging

import numpy as np
import sys
import torch
import torch.nn as nn
from transformers import BertModel, BertForSequenceClassification

from typing import Dict, Sequence

from cabby.model import util as mutil
from cabby.geo import util as gutil

BERT_TYPE = 'bert-base-multilingual-cased'

criterion = nn.CosineEmbeddingLoss()


class GeneralModel(nn.Module):
  def __init__(self, device):
    super(GeneralModel, self).__init__()
    self.device = device

  def get_dual_encoding(self, text_feat, cellid):
    return text_feat, cellid

  def forward(self, text: Dict, *args
              ):
    sys.exit("Implement compute_loss function in model")

  def predict(self, text, *args):
    sys.exit("Implement prediction function in model")


class DualEncoder(GeneralModel):
  def __init__(
    self,
    device,
    text_dim=768,
    hidden_dim=200,
    s2cell_dim=10,
    output_dim=100,
  ):
    GeneralModel.__init__(self, device)


    self.softmax = nn.Softmax(dim=-1)
    self.tanh = nn.Tanh()
    self.relu = nn.ReLU()
    self.model = BertModel.from_pretrained(
      BERT_TYPE)

    self.text_main = nn.Sequential(
      nn.Linear(text_dim, output_dim)
      # nn.ReLU(),
      # nn.Linear(hidden_dim, output_dim),
    )
    self.cellid_main = nn.Sequential(
      nn.Linear(s2cell_dim, output_dim),
    )
    self.cos = nn.CosineSimilarity(dim=2)

  def get_dual_encoding(self, text_tokens, cell_embedding):
    text_embedding = self.text_embed(text_tokens)

    cell_embedding = cell_embedding.to(torch.float32)
    cellid_embedding = self.cellid_main(cell_embedding)
    return text_embedding.shape[0], text_embedding, cellid_embedding

  def get_cell_embedding(self, cellid: str) -> torch.Tensor:
    EMBEDDING_PATH = 'cells_embedding_tel_aviv.npy'
    cell_id_to_vector = np.load(EMBEDDING_PATH, allow_pickle=True).item()
    return torch.tensor(cell_id_to_vector[cellid])

  def predict(self, text_tokens, all_cells_embedding, *args):

    batch_dim, text_embedding, cellid_embedding = self.get_dual_encoding(text_tokens, all_cells_embedding)

    cell_dim = cellid_embedding.shape[0]
    output_dim = cellid_embedding.shape[1]

    text_embedding_exp = text_embedding.unsqueeze(1).expand(batch_dim, cell_dim, output_dim)

    cellid_embedding_exp = cellid_embedding.expand(batch_dim, cell_dim, output_dim)

    label_to_cellid = args[0]
    assert cellid_embedding_exp.shape == text_embedding_exp.shape
    output = self.cos(cellid_embedding_exp, text_embedding_exp)

    output = output.detach().cpu().numpy()
    predictions = np.argmax(output, axis=1)
    print("____________________________________")
    print(predictions)
    points = mutil.predictions_to_points(predictions, label_to_cellid)
    return points

  def forward(self, text, cellid, *args):

    batch = args[0]

    neighbor_cells = batch['neighbor_cells']
    far_cells = batch['far_cells']


    # Correct cellid.

    target = torch.ones(cellid.shape[0]).to(self.device)
    _, text_embedding, cellid_embedding = self.get_dual_encoding(text, cellid)
    loss_cellid = criterion(text_embedding, cellid_embedding, target)

    # Neighbor cellid.

    target_neighbor = -1 * torch.ones(cellid.shape[0]).to(self.device)
    _, text_embedding_neighbor, cellid_embedding = self.get_dual_encoding(text, neighbor_cells)
    loss_neighbor = criterion(text_embedding_neighbor,
                              cellid_embedding, target_neighbor)

    # Far cellid.
    target_far = -1 * torch.ones(cellid.shape[0]).to(self.device)
    _, text_embedding_far, cellid_embedding = self.get_dual_encoding(text, far_cells)
    loss_far = criterion(text_embedding_far, cellid_embedding, target_far)
    # print("loss_far", loss_far)
    loss = loss_cellid + loss_neighbor + loss_far

    return loss.mean()

  def text_embed(self, text):
    # model = BertModel.from_pretrained("bert-base-multilingual-cased")
    outputs = self.model(**text)
    cls_token = outputs.last_hidden_state[:, -1, :]
    return self.text_main(cls_token)


class ClassificationModel(GeneralModel):
  def __init__(self, n_lables, device, hidden_dim=200):
    GeneralModel.__init__(self, device)

    # self.model = DistilBertForSequenceClassification.from_pretrained(
    #   'distilbert-base-multilingual-cased', num_labels=n_lables, return_dict=True)

    self.model = BertForSequenceClassification.from_pretrained('onlplab/alephbert-base', num_labels=n_lables, return_dict=True)
    self.criterion = nn.CrossEntropyLoss()

  def forward(self, text, cellid,labels):

    labels = torch.tensor(labels)
    outputs = self.model(
      input_ids=text['input_ids'],
      attention_mask=text['attention_mask'],
      labels=labels)

    return outputs.loss

  def predict(self, text, all_cells, *args):
    label_to_cellid = args[0]

    outputs = self.model(**text)
    logits = outputs.logits.detach().cpu().numpy()
    predictions = np.argmax(logits, axis=1)

    points = mutil.predictions_to_points(predictions, label_to_cellid)
    return points

