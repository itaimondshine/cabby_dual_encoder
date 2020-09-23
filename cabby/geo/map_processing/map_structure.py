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


import copy
import geopandas as gpd
from geopandas import GeoSeries
import networkx as nx
import os
import osmnx as ox
import pandas as pd
from shapely.geometry import box
from shapely.geometry.point import Point
from shapely.geometry.polygon import Polygon
from shapely import wkt
import sys
from typing import Dict, Tuple, Sequence, Text, Optional, Any

from cabby import logger
from cabby.geo import util
from cabby.geo.map_processing import graph
from cabby import logger

map_logger = logger.create_logger("map.log", 'map')

OSM_CRS = 32633  # UTM Zones (North).


class Map:

  def __init__(self, map_name: Text, level: int, load_directory: Text = None):
    assert (map_name == "Manhattan" or map_name == "Pittsburgh" or
        map_name == "Bologna")
    self.map_name = map_name
    self.s2_graph = None
    self.level = level

    if map_name == "Manhattan":
      self.polygon_area = wkt.loads(
        'POLYGON ((-73.9455846946375 40.7711351085905,-73.9841893202025 40.7873649535321,-73.9976322499213 40.7733311258037,-74.0035177432988 40.7642404854275,-74.0097394992375 40.7563218869601,-74.01237903206 40.741427380319,-74.0159612551762 40.7237048027967,-74.0199205544099 40.7110727528606,-74.0203570671504 40.7073623945662,-74.0188292725586 40.7010329598287,-74.0087894795267 40.7003781907179,-73.9976584046436 40.707144138196,-73.9767057930988 40.7104179837498,-73.9695033328803 40.730061057073,-73.9736502039152 40.7366087481808,-73.968412051029 40.7433746956588,-73.968412051029 40.7433746956588,-73.9455846946375 40.7711351085905))'
      )

    elif map_name == "Pittsburgh":
      self.polygon_area = box(
        miny=40.425, minx=-80.035, maxy=40.460, maxx=-79.930,
        ccw=True)
    else:  # Bologna.
      self.polygon_area = box(
        miny=44.4902, minx=11.3333, maxy=44.5000, maxx=11.3564,
        ccw=True)

    if load_directory is None:
      self.poi, self.streets = self.get_poi()
      self.nx_graph = ox.graph_from_polygon(
        self.polygon_area, network_type='walk')
      self.nodes, self.edges = ox.graph_to_gdfs(self.nx_graph)

      # Find closest nodes to POI.
      self.poi['node'] = self.poi['centroid'].apply(self.closest_nodes)

    else:
      self.load_map(load_directory)
    self.create_S2Graph(level)
    self.process_param()

  def process_param(self):
    '''Helper function for processing the class data objects.'''

    # Set the coordinate system.
    self.poi = self.poi.set_crs(epsg=OSM_CRS, allow_override=True)
    self.nodes = self.nodes.set_crs(epsg=OSM_CRS, allow_override=True)
    self.edges = self.edges.set_crs(epsg=OSM_CRS, allow_override=True)

    # Drop columns with list type.
    self.edges.drop(self.edges.columns.difference(
      ['osmid', 'length', 'geometry', 'u', 'v', 'key']), 1, inplace=True)
    self.edges['osmid'] = self.edges['osmid'].apply(lambda x: str(x))

  def closest_nodes(self, point: Point) -> int:
    '''Find closest nodes to POI. 
    Arguments:
      point: The point to which a closest node will be retrived.
    Returns:
      The Node id closest to point.
    '''
    ""
    point_xy = util.tuple_from_point(point)
    return ox.distance.get_nearest_node(self.nx_graph, point_xy)

  def get_poi(self) -> Tuple[GeoSeries, GeoSeries]:
    '''Helper funcion for extracting POI for the defined place.'''

    tags = {'name': True}

    osm_poi = ox.pois.pois_from_polygon(self.polygon_area, tags=tags)
    osm_poi_named_entities = osm_poi[osm_poi['name'].notnull()]
    osm_highway = osm_poi_named_entities['highway']
    osm_poi_no_streets = osm_poi_named_entities[osm_highway.isnull()]
    osm_poi_streets = osm_poi_named_entities[osm_highway.notnull()]

    # Get centroid for POI.
    osm_poi_no_streets = osm_poi_no_streets.set_crs(epsg=OSM_CRS,
                            allow_override=True)
    osm_poi_no_streets['centroid'] = osm_poi_no_streets['geometry'].apply(
      lambda x: x if isinstance(x, Point) else x.centroid)

    return osm_poi_no_streets, osm_poi_streets

  def get_cellids_for_poi(self, geometry: Any) -> Optional[Sequence[int]]:
    '''get cellids for POI. 
    Arguments:
      geometry: The geometry to which a cellids will be retrived.
    Returns:
      The the cellids.
    '''
    ""
    if isinstance(geometry, Point):
      return util.cellid_from_point(geometry, self.level)
    else:
      return util.cellid_from_polygon(geometry, self.level)

  def get_cellids_for_streets(
      self, geometry: Any) -> Optional[Sequence[int]]:
    '''get cellids for streets. 
    Arguments:
      geometry: The geometry to which a cellids will be retrived.
    Returns:
      The the cellids.
    '''
    ""
    if isinstance(geometry, Point):
      return util.cellid_from_point(geometry, self.level)
    else:
      return util.cellid_from_polygon(geometry, self.level)

  def create_S2Graph(self, level: int):
    '''Helper funcion for creating S2Graph.'''

    # Get cellids for POI.
    self.poi['cellids'] = self.poi['geometry'].apply(
      self.get_cellids_for_poi)

    # Get cellids for streets.
    self.streets['cellids'] = self.streets['geometry'].apply(self.
                                 get_cellids_for_streets)

    # Filter out entities that we didn't mange to get cellids covering.
    self.poi = self.poi[self.poi['cellids'].notnull()]
    self.streets = self.streets[self.streets['cellids'].notnull()]

    # Create graph.
    self.s2_graph = graph.MapGraph()

    # Add POI to graph.
    self.poi[['cellids', 'osmid']].apply(
      lambda x: self.s2_graph.add_poi(x.cellids, x.osmid), axis=1)

    # Add street to graph.
    self.streets[['cellids', 'osmid']].apply(
      lambda x: self.s2_graph.add_street(x.cellids, x.osmid), axis=1)

  def get_valid_path(self, dir_name: Text, name_ending: Text,
             file_ending: Text) -> Optional[Text]:
    '''Creates the file path and checks validity.
    Arguments:
      dir_name: The directory of the path.
      name_ending: the end of the name  of the file
      (_graph\_node\_poi\_streets).
      file_ending: the type of the file.
    Returns:
      The valid path.
    '''

    base_filename = self.map_name.lower() + name_ending

    # Check if directory is valid.
    assert os.path.exists(dir_name), (f"Current directory is: {os.getcwd()}."
                      f" The directory {dir_name} doesn't exist.")

    # Create path.
    path = os.path.join(dir_name, base_filename + file_ending)

    return path

  def write_map(self, dir_name: Text):
    '''Save POI to disk.'''

    # Write POI.
    pd_poi = copy.deepcopy(self.poi)
    pd_poi['cellids'] = pd_poi['cellids'].apply(
      lambda x: util.s2ids_from_s2cells(x))

    path = self.get_valid_path(dir_name, '_poi', '.pkl')
    if not os.path.exists(path):
      pd_poi.to_pickle(path)
    else:
      map_logger.info("Path {0} already exists.".format(path))

    # Write streets.
    pd_streets = copy.deepcopy(self.streets)
    pd_streets['cellids'] = pd_streets['cellids'].apply(
      lambda x: util.s2ids_from_s2cells(x))

    path = self.get_valid_path(dir_name, '_streets', '.pkl')
    if not os.path.exists(path):
      pd_streets.to_pickle(path)
    else:
      map_logger.info("Path {0} already exists.".format(path))

    # Write graph.
    base_filename = self.map_name.lower() + "_graph"
    path = os.path.join(dir_name, base_filename + ".gpickle")
    if not os.path.exists(path):
      nx.write_gpickle(self.nx_graph, path)
    else:
      map_logger.info("Path {0} already exists.".format(path))

    # Write nodes.
    path = self.get_valid_path(dir_name, '_nodes', '.geojson')
    if not os.path.exists(path):
      self.nodes.to_file(path, driver='GeoJSON')
    else:
      map_logger.info("Path {0} already exists.".format(path))

    # Write edges.
    path = self.get_valid_path(dir_name, '_edges', '.geojson')
    if not os.path.exists(path):
      self.edges.to_file(path, driver='GeoJSON')
    else:
      map_logger.info("Path {0} already exists.".format(path))

  def load_map(self, dir_name: Text):
    '''Load POI from disk.'''

    # Load POI.
    path = self.get_valid_path(dir_name, '_poi', '.pkl')
    assert os.path.exists(
      path), "Path {0} doesn't exist.".format(path)
    poi_pandas = pd.read_pickle(path)
    if 'cellids' in poi_pandas:
      poi_pandas['cellids'] = poi_pandas['cellids'].apply(
        lambda x: util.s2cells_from_cellids(x))
    self.poi = poi_pandas

    # Load streets.
    path = self.get_valid_path(dir_name, '_streets', '.pkl')
    assert os.path.exists(
      path), "Path {0} doesn't exist.".format(path)
    streets_pandas = pd.read_pickle(path)
    self.streets = streets_pandas

    # Load graph.
    path = self.get_valid_path(dir_name, '_graph', '.gpickle')
    assert os.path.exists(
      path), "Path {0} doesn't exist".format(path)
    self.nx_graph = nx.read_gpickle(path)

    # Load nodes.
    path = self.get_valid_path(dir_name, '_nodes', '.geojson')
    assert os.path.exists(
      path), "Path {0} doesn't exist.".format(path)
    self.nodes = gpd.read_file(path, driver='GeoJSON')

    # Load edges.
    path = self.get_valid_path(dir_name, '_edges', '.geojson')
    assert os.path.exists(
      path), "Path {0} doesn't exist.".format(path)
    self.edges = gpd.read_file(path, driver='GeoJSON')
