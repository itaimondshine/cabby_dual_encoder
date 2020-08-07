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

from absl import app
from absl import flags

from shapely.geometry.point import Point
import osmnx as ox
from map_structure import Map

FLAGS = flags.FLAGS
flags.DEFINE_string("place", None, "map area - Manhattan or Pittsburgh.")
flags.DEFINE_string("path", None, "path to save the map file.")

# Required flags.
flags.mark_flag_as_required("place")


def main(argv):
    del argv  # Unused.


if __name__ == '__main__':
    app.run(main)
