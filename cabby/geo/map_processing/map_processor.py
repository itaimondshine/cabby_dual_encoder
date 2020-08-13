# coding=utf-8 Copyright 2020 Google LLC Licensed under the Apache License,
# Version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

'''Command line application to output all POI in an area of the OSM.

Example:

$ bazel-bin/cabby/geo/map_processing/map_processor --region Manhattan --level 18
'''

from absl import app
from absl import flags

from shapely.geometry.point import Point
import osmnx as ox

from cabby.geo.map_processing import map_structure

FLAGS = flags.FLAGS
flags.DEFINE_enum(
  "region", None, ['Pittsburgh', 'Manhattan'], 
  "Map areas: Manhattan or Pittsburgh.")
flags.DEFINE_integer("level", None, "Minumum S2 level of the map.")

# Required flags.
flags.mark_flag_as_required("region")


def main(argv):
    del argv  # Unused.
    map = map_structure.Map(FLAGS.region, FLAGS.level)
    print(map.poi)


if __name__ == '__main__':
    app.run(main)