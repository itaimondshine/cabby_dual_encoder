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

'''Library to support sampling points, creating routes between them and pivots
along the path and near the goal.'''

from typing import Sequence, Optional, Dict, Any

from absl import logging
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
import inflect
import multiprocessing
import pathos.helpers.mp_helper
from multiprocessing import Semaphore
import numpy as np
import networkx as nx
import os
import osmnx as ox
import pandas as pd
import random
from shapely.ops import nearest_points
from shapely.geometry.point import Point
from shapely.geometry import LineString
import sys

from torch import NoneType

from cabby.geo import util
from cabby.geo.map_processing import map_structure
from cabby.geo import geo_item
from cabby.geo import osm

SMALL_POI = 4  # Less than 4 S2Cellids.

SEED = 1
MAX_SEED = 2 ** 32 - 1

SAVE_ENTITIES_EVERY = 100
MAX_BATCH_GEN = 100
MAX_BATCH_GEN = MAX_BATCH_GEN if MAX_BATCH_GEN < SAVE_ENTITIES_EVERY else SAVE_ENTITIES_EVERY
MAX_PATH_DIST = 2000
MIN_PATH_DIST = 200
NEAR_PIVOT_DIST = 80
ON_PIVOT_DIST = 10

# The max number of failed tries to generate a single path entities.
MAX_NUM_GEN_FAILED = 10

PIVOT_ALONG_ROUTE_MAX_DIST = 0.0007
MAX_NUM_BEYOND_TRY = 50

N_AROUND_PIVOTS = 10
N_MAIN_PIVOTS = 15

around_pivots = [f"around_goal_pivot_{n}" for n in range(1, N_AROUND_PIVOTS + 1)]

LANDMARK_TYPES = [
                   "end_point"] + [
                   "near_pivot", ] + around_pivots + ['main_near_pivot']

FEATURES_TYPES = [
  "spatial_rel_goal",

  "intersections",
  "goal_position",
  "spatial_rel_main_near"]

inflect_engine = inflect.engine()


class Walker:
  def __init__(self, map: map_structure.Map, rand_sample: bool = True):
    # whether to sample randomly.
    self.rand_sample = rand_sample
    self.map = map

  def get_generic_tag(self, poi: pd.Series) -> Optional[str]:
    '''Selects a non-specific tag (e.g., museum instead of "Austin Museum of
    Popular Culture") instead of a POI.
    Arguments:
      poi: The POI to select a non-specific tag for.
    Returns:
      A non-specific tag.
    '''
    for tag, addition in osm.NON_SPECIFIC_TAGS.items():
      if tag not in poi or not isinstance(poi[tag], str):
        continue
      if addition == True:
        return tag
      tag_value = poi[tag]
      tag_value_clean = tag_value.replace("_", " ")
      if tag_value in osm.CORRECTIONS:
        tag_value_clean = osm.CORRECTIONS[tag_value]
      if tag_value_clean in ['yes', 'no']:
        continue
      if addition == 'after':
        new_tag = tag_value_clean + " " + tag
      elif addition == "before":
        new_tag = tag + " " + tag_value_clean
      elif addition == False:
        new_tag = tag_value_clean
      elif tag_value not in addition:
        continue
      else:
        new_tag = tag_value_clean
      if new_tag in osm.CORRECTIONS:
        new_tag = osm.CORRECTIONS[new_tag]
      if new_tag in osm.BLOCK_LIST:
        continue
      return new_tag
    return None

  def select_generic_unique_pois(
    self, pois: pd.DataFrame,
    is_unique: bool = False,
    end_point: pd.DataFrame = None,
    avoid_pivots: Sequence[str] = []):
    '''Returns a non-specific POIs with main tag being the non-specific tag.
    Arguments:
      pois: all pois to select from.
      is_unique: if to filter unique tags.
      end_point: end point of the path.
      avoid_pivots: pivots to avoid picking.
    Returns:
      A number of non-specific POIs which are unique.
    '''
    # Assign main tag.
    main_tags = pois.apply(self.get_generic_tag, axis=1)
    new_pois = pois.assign(main_tag=main_tags)
    new_pois.dropna(subset=['main_tag'], inplace=True)
    new_pois = new_pois[~new_pois['osmid'].isin(avoid_pivots)]

    if end_point is not None:
      new_pois = new_pois[new_pois['osmid'] != end_point['osmid']]

    # Get Unique main tags.
    if is_unique:
      # Randomly select whether the near by pivot would be
      # a single pivot (e.g., `a toy shop` or
      # a group of unique landmark (e.g, `3 toy shops`)
      is_group = self.randomize_boolean(probabilty=30)

      if is_group:
        uniqueness = new_pois.duplicated(subset=['main_tag'], keep=False) == True
        new_pois_uniq = new_pois[uniqueness]

        if new_pois_uniq.shape[0] == 0:
          return self.get_generic_unique_pois_single(new_pois)

        count_by_tag = new_pois_uniq.main_tag.value_counts().to_dict()

        tag_list = list((count_by_tag.keys()))

        random.shuffle(tag_list)

        for chosen_tag in tag_list:

          chosen_count = count_by_tag[chosen_tag]
          if chosen_count <= 1:
            continue

          new_pois_uniq_group = new_pois_uniq[new_pois_uniq['main_tag'] == chosen_tag]
          if new_pois_uniq_group.shape[0] == 0:
            continue
          single_new_pois_uniq = new_pois_uniq_group.sample()
          anchor = single_new_pois_uniq.iloc[0]['centroid']
          entities_geo_group = new_pois_uniq_group[new_pois_uniq_group.apply(
            lambda x: (
              util.get_distance_between_geometries(
                x.geometry, anchor) <= ON_PIVOT_DIST and util.get_distance_between_geometries(
              x.geometry, anchor) > 0), axis=1)]

          chosen_count = entities_geo_group.shape[0]
          if chosen_count <= 1:
            continue

          by_word = self.randomize_boolean()
          if by_word:
            chosen_count = inflect_engine.number_to_words(chosen_count)
          single_new_pois_uniq['main_tag'] = str(chosen_count) + \
                                             " " + inflect_engine.plural(chosen_tag)

          single_new_pois_uniq.drop(
            single_new_pois_uniq.columns.difference(
              [
                'main_tag', 'centroid', 'geometry', 'osmid'] + \
              osm.PROMINENT_TAGS_ORDERED + list(osm.NON_SPECIFIC_TAGS.keys())),
            1, inplace=True)
          single_new_pois_uniq['name'] = single_new_pois_uniq['main_tag']

          single_new_pois_uniq['grouped'] = True
          return single_new_pois_uniq

      return self.get_generic_unique_pois_single(new_pois)
    return new_pois

  def get_generic_unique_pois_single(self, pois: pd.DataFrame):
    uniqueness = pois.duplicated(subset=['main_tag'], keep=False) == False
    new_pois_uniq = pois[uniqueness]
    return new_pois_uniq

  def select_generic_poi(self, pois: pd.DataFrame):
    '''Returns a non-specific POI with main tag being the non-specific tag.
    Arguments:
      pois: all pois to select from.
    Returns:
      A single sample of a POI with main tag being the non-specific tag.
    '''

    pois_generic = self.select_generic_unique_pois(pois)

    if pois_generic.shape[0] == 0:
      return None
    # Sample POI.
    poi = self.sample_point(pois_generic)
    poi['geometry'] = poi.centroid
    return poi

  def get_end_poi(self) -> Optional[GeoSeries]:
    '''Returns a random POI.
    Returns:
      A single POI.
    '''

    # Filter large POI.
    small_poi = self.map.poi[self.map.poi['s2cellids'].str.len() <= SMALL_POI]

    if small_poi.shape[0] == 0:
      return None

    # Filter non-specific tags.
    return self.select_generic_poi(small_poi)

  def randomize_boolean(self, probabilty: int = 50) -> bool:
    '''Returns a random\non random boolean value.
    Arguments:
      probabilty: probabilty it will be True (0-100).
    Returns:
      Returns a random\non random boolean value.
    '''
    if self.rand_sample:
      rand_int = random.randint(0, 100)
      return rand_int <= probabilty
    return True

  def sample_point(self,
                   df: gpd.GeoDataFrame
                   ) -> GeoSeries:
    '''Returns a random\non random 1 sample of a POI.
    Arguments:
      df: data to sample.
    Returns:
      A single sample of a POI.
    '''
    if self.rand_sample:
      return df.sample(1, random_state=random.randint(0, MAX_SEED)).iloc[0]
    return df.sample(1, random_state=SEED).iloc[0]

  def get_start_poi(self,
                    end_point: Dict
                    ) -> Optional[GeoSeries]:
    '''Returns the a random POI within distance of a given POI.
    Arguments:
      end_point: The POI to which the picked POI should be within distance
      range.
    Returns:
      A single POI.
    '''

    # Get closest nodes to points.
    dest_osmid = end_point['osmid']

    try:
      # Find nodes within 2000 meter path distance.
      outer_circle_graph = ox.truncate.truncate_graph_dist(
        self.map.nx_graph, dest_osmid,
        max_dist=MAX_PATH_DIST, weight='true_length')

      outer_circle_graph_osmid = list(outer_circle_graph.nodes.keys())
    except nx.exception.NetworkXPointlessConcept:  # GeoDataFrame returned empty
      return None

    try:
      # Get graph that is too close (less than 200 meter path distance)
      inner_circle_graph = ox.truncate.truncate_graph_dist(
        self.map.nx_graph, dest_osmid,
        max_dist=MIN_PATH_DIST, weight='true_length')
      inner_circle_graph_osmid = list(inner_circle_graph.nodes.keys())

    except nx.exception.NetworkXPointlessConcept:  # GeoDataFrame returned empty
      inner_circle_graph_osmid = []

    osmid_in_range = [
      osmid for osmid in outer_circle_graph_osmid if osmid not in
                                                     inner_circle_graph_osmid]

    poi_in_ring = self.map.poi[self.map.poi['osmid'].isin(osmid_in_range)]

    # Filter large POI.
    small_poi = poi_in_ring[poi_in_ring['s2cellids'].str.len() <= SMALL_POI]

    # Filter by distance
    small_poi = small_poi[
      small_poi.apply(
        lambda x: util.get_distance_between_geometries(
          x.geometry,
          end_point['centroid']) > MIN_PATH_DIST, axis=1)]

    # Filter non-specific tags.
    return self.select_generic_poi(small_poi)

  def get_landmark_if_tag_exists(self,
                                 gdf: GeoDataFrame,
                                 tag: str,
                                 pick_generic_name: bool = False
                                 ) -> GeoSeries:
    '''Check if tag exists, set main tag name and choose pivot.
    Arguments:
      gdf: The set of landmarks.
      tag: tag to check if exists.
      main_tag: the tag that should be set as the main tag.
    Returns:
      A single landmark.
    '''

    candidate_landmarks = gdf.columns
    secondary_pivots = []
    if tag in candidate_landmarks:
      pivots = gdf[gdf[tag].notnull()]

      if pivots.shape[0]:
        if pick_generic_name:
          tags_keys = osm.NON_SPECIFIC_TAGS.keys()
        else:
          tags_keys = osm.SPECIFIC_TAGS
        for tag_k in tags_keys:
          if tag_k not in pivots:
            continue
          pivots_tag = pivots[pivots[tag_k].notnull()]

          if pick_generic_name and isinstance(
            osm.NON_SPECIFIC_TAGS[tag_k], list):
            pivots_tag = pivots_tag[pivots_tag[tag_k].isin(osm.NON_SPECIFIC_TAGS[tag_k])]
          if pivots_tag.shape[0]:
            if 'main_tag' not in pivots:
              pivots_tag = pivots_tag.assign(main_tag=pivots_tag[tag_k])

            pivots_prominent = pivots_tag[
              ~pivots_tag['amenity'].isin(osm.NEGLIGIBLE_AMENITY)]
            if pivots_prominent.shape[0] == 0:
              pivot = self.sample_point(pivots_tag)
              secondary_pivots.append(pivot)
              continue
            pivot = self.sample_point(pivots_prominent)
            return pivot

    if len(secondary_pivots) > 0:
      return secondary_pivots.pop()
    return None

  def pick_prominent_pivot(self,
                           df_pivots: GeoDataFrame,
                           end_point: Dict[str, Any],
                           pick_generic_name: bool = False
                           ) -> Optional[GeoSeries]:
    '''Select a landmark from a set of landmarks by priority.
    Arguments:
      df_pivots: The set of landmarks.
      end_point: The goal location.
      path_geom: The geometry of the path.
    Returns:
      A single landmark.
    '''

    # Remove goal location.
    try:
      df_pivots = df_pivots[df_pivots['osmid'] != end_point['osmid']]
    except:
      pass

    if df_pivots.shape[0] == 0:
      return None

    pivot = None

    for main_tag in osm.PROMINENT_TAGS_ORDERED:

      pivot = self.get_landmark_if_tag_exists(df_pivots,
                                              main_tag,
                                              pick_generic_name
                                              )
      if pivot is not None:
        return pivot

    return pivot

  def get_pivot_near_goal(self,
                          end_point: GeoSeries,

                          max_distance_from_goal: int,
                          min_distance_from_goal: int,
                          avoid_pivots: Sequence[str] = []
                          ) -> Optional[GeoSeries]:
    '''Return a picked landmark near the end_point.
    Arguments:
      end_point: The goal location.
      path_geom: The geometry of the path selected.
      max_distance_from_goal: The max distance from goal.
      min_distance_from_goal: The min distance from goal.
      avoid_pivots: pivots to avoid picking.
    Returns:
      A single landmark near the goal location.
    '''

    near_poi_con = self.map.poi.apply(
      lambda x: util.get_distance_between_geometries(
        x.geometry,
        end_point['centroid']) < max_distance_from_goal and util.get_distance_between_geometries(
        x.geometry,
        end_point['centroid']) > min_distance_from_goal, axis=1)

    poi = self.map.poi[near_poi_con]

    columns_empty = self.map.nodes.columns.tolist() + ['main_tag']
    if poi.shape[0] == 0:
      return GeoDataFrame(index=[0], columns=columns_empty).iloc[0]

    # Remove streets and roads.
    if 'highway' in poi.columns:
      poi = poi[poi['highway'].isnull()]

    # Remove the endpoint.
    nearby_poi = poi[poi['osmid'] != end_point['osmid']]

    # Filter non-specific tags.
    unique_poi = self.select_generic_unique_pois(
      nearby_poi, is_unique=True, end_point=end_point, avoid_pivots=avoid_pivots)

    if unique_poi.shape[0] == 0:
      return GeoDataFrame(index=[0], columns=columns_empty).iloc[0]

    prominent_poi = self.pick_prominent_pivot(
      unique_poi, end_point, pick_generic_name=True)
    if prominent_poi is None:
      return GeoDataFrame(index=[0], columns=columns_empty).iloc[0]

    return prominent_poi

  def get_street_name(self, end_point: Dict) -> str:
    '''Return the street name of the end_point.
    Arguments:
      end_point: The goal location.
    Returns:
      The street name.
    '''


    pass







  def get_pivots(self,
                 end_point: Dict
                 ) -> Optional[Sequence[GeoSeries]]:
    '''Return a picked landmark on a given route.
    Arguments:
      route: The route along which a landmark will be chosen.
      end_point: The goal location.
      start_point: The start location.
    Returns:
      A single landmark.
    '''

    # Get pivot along the goal location.

    # Get pivot near the goal location.
    near_pivot = self.get_pivot_near_goal(
      end_point=end_point, max_distance_from_goal=NEAR_PIVOT_DIST, min_distance_from_goal=2)

    if near_pivot['geometry'] is None:
      return None

    list_around_goal_pivots_osmid = [near_pivot['osmid']]
    list_around_goal_pivots = []
    # Get a second and third pivots along the goal location.
    for _ in range(0, N_AROUND_PIVOTS):
      around_goal_pivot_x = self.get_pivot_near_goal(
        end_point,
        2 * NEAR_PIVOT_DIST,
        NEAR_PIVOT_DIST,
        list_around_goal_pivots_osmid)
      list_around_goal_pivots_osmid.append(around_goal_pivot_x['osmid'])
      list_around_goal_pivots.append(around_goal_pivot_x)

    # Get pivot located past the goal location and beyond the route.

    list_pivots = [near_pivot] + list_around_goal_pivots

    # for pivot in list_pivots:
    #   if ce
    #

    return list_pivots

  def get_egocentric_spatial_relation_pivot(self,
                                            ref_point: Point,
                                            route: GeoDataFrame
                                            ) -> str:
    line = LineString(route['geometry'].tolist())
    dist_projected = line.project(ref_point)
    cut_geometry = util.cut(line, dist_projected)
    first_segment = cut_geometry[0]
    coords = list(first_segment.coords)
    return self.calc_spatial_relation_for_line(
      ref_point, Point(coords[-1]), Point(coords[-2]))

  def calc_spatial_relation_for_line(self,
                                     ref_point: Point,
                                     line_point_last_part: Point,
                                     line_point_second_from_last: Point,
                                     ) -> str:

    # Calculate the angle of the last segment of the line_point_last_part.
    azim_route = util.get_bearing(
      line_point_second_from_last, line_point_last_part)

    # Calculate the angle between the last segment of the route and the goal.
    azim_ref_point = util.get_bearing(
      line_point_last_part, ref_point)

    diff_azim = (azim_ref_point - azim_route) % 360

    if diff_azim < 180:
      return "right"

    return "left"

  def get_sample(self) -> Optional[geo_item.GeoEntity]:

    '''Sample start and end point, a pivot landmark and route.
    Returns:
      A start and end point, a pivot landmark and route.
    '''

    geo_landmarks = {}
    # Select end point.
    geo_landmarks['end_point'] = self.get_end_poi()
    if geo_landmarks['end_point'] is None:
      return None

    # Select pivots.
    result = self.get_pivots(
      geo_landmarks['end_point'])
    if result is None:
      return None

    for landmark in result:
      try:
        landmark.geometry = landmark.centroid
      except AttributeError:
        continue





    geo_landmarks['near_pivot'] = result[0]



    for i in range(1, len(result)):
      geo_landmarks[f'around_goal_pivot_{i}'] = result[i]

    geo_features = {}

    # Get Egocentric spatial relation from goal.

    # Get Egocentric spatial relation from main pivot.

    # Get number of intersections between main pivot and goal location.

    rvs_path_entity = geo_item.GeoEntity.add_entity(
      route=None,
      geo_features=geo_features,
      geo_landmarks=geo_landmarks)

    return rvs_path_entity

  def get_single_sample(
    self,
    index: int,
    sema: Any,
    n_samples: int,
    return_dict: Dict[int, geo_item.GeoEntity]):
    '''Sample exactly one RVS path sample.
    Arguments:
      index: index of sample.
      sema: Semaphore Object.
      n_samples: the total number of samples to generate.
      return_dict: The dictionary of samples generated.
    '''
    sema.acquire()
    entity = None
    attempt = 0
    while entity is None:
      if attempt >= MAX_NUM_GEN_FAILED:
        sys.exit(f"Reached max number of failed attempts for sample {index}.")
      entity = self.get_sample()
      print(entity)
      attempt += 1

    logging.info(f"Created sample {index}/{n_samples}.")
    return_dict[index] = entity
    sema.release()

  def generate_and_save_rvs_routes(self,
                                   path_rvs_path: str,
                                   n_samples: int,
                                   n_cpu: int = multiprocessing.cpu_count() - 1
                                   ):
    '''Sample start and end point, a pivot landmark and route and save to file.
    Arguments:
      path_rvs_path: The path to which the data will be appended.
      map: The map of a specific region.
      n_samples: the max number of samples to generate.
    '''

    manager = multiprocessing.Manager()

    sema = Semaphore(n_cpu)
    new_entities = []
    lst = list(range(n_samples))
    # batches = [
    #   lst[i:i + MAX_BATCH_GEN] for i in range(0, len(lst), MAX_BATCH_GEN)]

    return_dict = manager.dict()

    for i in range(n_samples):
        self.get_single_sample(i, sema, n_samples, return_dict)
    # p = multiprocessing.Process(
    #         target=self.get_single_sample,
    #         args=(i+1, sema ,n_samples,
    #         return_dict))
    #       jobs.append(p)
    #       p.start()
    #   for proc in jobs:
    #       proc.join()
    new_entities += [entity for idx_entity, entity in return_dict.items()]

    if len(new_entities) >= SAVE_ENTITIES_EVERY:
      geo_item.save(new_entities, path_rvs_path)
      new_entities = []
    if len(new_entities) > 0:
      print(type(new_entities))
      geo_item.save(new_entities, path_rvs_path)


def load_entities(path: str) -> Sequence[geo_item.GeoEntity]:
  if not os.path.exists(path):
    return []

  geo_types_all = {}
  for landmark_type in LANDMARK_TYPES:
    geo_types_all[landmark_type] = gpd.read_file(path, layer=landmark_type)

  geo_types_all['route'] = gpd.read_file(path, layer='path_features')['geometry']
  geo_types_all['path_features'] = gpd.read_file(path, layer='path_features')
  geo_entities = []

  for row_idx in range(geo_types_all[LANDMARK_TYPES[0]].shape[0]):

    landmarks = {}
    for landmark_type in LANDMARK_TYPES:
      landmarks[landmark_type] = geo_types_all[landmark_type].iloc[row_idx]
    features = geo_types_all['path_features'].iloc[row_idx].to_dict()
    del features['geometry']
    route = geo_types_all['route'].iloc[row_idx]

    geo_item_cur = geo_item.GeoEntity.add_entity(
      geo_landmarks=landmarks,
      geo_features=features,
      route=LineString(route.exterior.coords[:-1])
    )
    geo_entities.append(geo_item_cur)

  logging.info(f"Loaded entities {len(geo_entities)} from <= {path}")
  return geo_entities
