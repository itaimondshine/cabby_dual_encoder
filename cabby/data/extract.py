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

'''Library to support data extraction from Wikipedia and Wikidata.'''

import json
import os
from typing import Dict, Tuple, Sequence, Text, Optional
from cabby.data.wikidata import query as wdq
from cabby.data.wikipedia import query as wpq
from cabby.data.wikidata import item as wdi
from cabby.data.wikipedia import item as wpi
from cabby.data import wikigeo


def get_wikigeo_data(wikidata_items: Sequence[wdi.WikidataEntity]) ->Sequence:
    '''Get data from Wikipedia based on Wikidata items" 
    Arguments:
        wikidata_items: The Wikidata items to which corresponding Wikigeo  items will be extracted.
    Returns:
        The Wikigeo items found (composed of Wikipedia (text,title) and Wikidata (location) data).
    '''
        # Get Wikipedia titles.
    titles = [entity.wikipedia_title for entity in wikidata_items]

    # Get Wikipedia pages.
    wikipedia_pages = wpq.get_wikipedia_items(titles)
    wikipedia_items = [wpi.WikipediaEntity.from_api_result(
        result) for result in wikipedia_pages]

    # # Get Wikipedia titles.
    wikipedia_titles = [entity.title for entity in wikipedia_items]

    # # Change to Geodata dataset foramt.
    geo_data = []
    for wikipedia, wikidata in zip(wikipedia_items, wikidata_items):
        geo_data.append(wikigeo.WikigeoEntity.from_wiki_items(
            wikipedia, wikipedia, wikidata).sample)

    # # Get backlinks for Wikipedia pages.
    backlinks_pages = wpq.get_backlinks_items_from_wikipedia_titles(
        wikipedia_titles)
    backlinks_items = []
    for list_backlinks in backlinks_pages:
        backlinks_items.append(
            [wpi.WikipediaEntity.from_backlinks_api_result(result) for result in list_backlinks])

    # Change backlinks pages to Geodata dataset format.
    for list_backlinks, original_wikipedia, original_wikidata in zip(backlinks_items, wikipedia_items, wikidata_items):
        for backlink in list_backlinks:
            wikigeo_sample  = wikigeo.WikigeoEntity.from_wiki_items(
                backlink, original_wikipedia, original_wikidata).sample
            if wikigeo_sample in geo_data:
                continue
            geo_data.append(wikigeo_sample)

    return geo_data

def get_data_by_qid(qid: Text) -> Sequence:
    '''Get data from Wikipedia and Wikidata by region" 
    Arguments:
        qid(Text): The qid of the Wikidata to extract items from.
    Returns:
        The Wikipedia (text,title) and Wikidata (location) data found.
    '''

    # Get Wikidata items by region.
    wikidata_results = wdq.get_geofenced_wikidata_items_by_qid(qid)
    wikidata_items = [wdi.WikidataEntity.from_sparql_result(result)
                      for result in wikidata_results]
    
    return get_wikigeo_data(wikidata_items)





def get_data_by_region(region: Text) -> Sequence:
    '''Get data from Wikipedia and Wikidata by region" 
    Arguments:
        region(Text): The region to extract items from.
    Returns:
        The Wikipedia (text,title) and Wikidata (location) data found.
    '''

    # Get Wikidata items by region.
    wikidata_results = wdq.get_geofenced_wikidata_items(region)
    wikidata_items = [wdi.WikidataEntity.from_sparql_result(result)
                      for result in wikidata_results]

    return get_wikigeo_data(wikidata_items)

