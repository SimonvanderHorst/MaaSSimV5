################################################################################
# Module: utils/network.py
# Network (graph) loading, downloading, saving, and statistics
################################################################################

import os
import pandas as pd
from collections import defaultdict
from dotmap import DotMap

import osmnx as ox
import networkx as nx

from .config import set_t0


def networkstats(inData):
    """
    for a given network calculates it center of gravity (avg of node coordinates),
    gets nearest node and network radius (75th percentile of lengths from the center)
    returns a dictionary with center and radius
    """
    center_x = pd.DataFrame((inData.G.nodes(data='x')))[1].mean()
    center_y = pd.DataFrame((inData.G.nodes(data='y')))[1].mean()

    # Use osmnx 0.15.1 API (for old environments) vs 2.x API (for new environments)
    try:
        from osmnx.distance import get_nearest_node
        nearest = get_nearest_node(inData.G, (center_y, center_x))  # Note: (lat, lon) order
    except ImportError:
        # osmnx 2.x API
        nearest = ox.distance.nearest_nodes(inData.G, center_x, center_y)
    ret = DotMap({'center': nearest, 'radius': inData.skim[nearest].quantile(0.75)})
    return ret


def _compute_safe_nodes(G, max_speed_kmh):
    """
    Return frozenset of nodes NOT adjacent to any edge with speed limit > max_speed_kmh.

    maxspeed parsing: numeric string -> km/h; "N mph" -> km/h; list -> max value;
    non-parseable/missing -> highway-type fallback.
    Highway fallbacks: motorway/trunk -> 100, primary -> 80, secondary -> 60,
                       tertiary/unclassified -> 50, residential/service/else -> 30.
    """
    _HW_FALLBACK = {
        'motorway': 100, 'motorway_link': 100, 'trunk': 100, 'trunk_link': 100,
        'primary': 80, 'primary_link': 80,
        'secondary': 60, 'secondary_link': 60,
        'tertiary': 50, 'tertiary_link': 50, 'unclassified': 50,
    }

    def _parse_speed(s):
        if s is None:
            return None
        s = str(s).strip()
        if 'mph' in s.lower():
            try:
                return int(float(s.lower().replace('mph', '').strip()) * 1.60934)
            except ValueError:
                return None
        return int(s) if s.isdigit() else None

    def _hw_fallback(hw):
        if isinstance(hw, list):
            hw = hw[0] if hw else ''
        return _HW_FALLBACK.get(str(hw).strip(), 30)

    node_max = defaultdict(int)
    for u, v, _k, d in G.edges(data=True, keys=True):
        ms = d.get('maxspeed')
        if isinstance(ms, list):
            parsed = [_parse_speed(s) for s in ms]
            speed = max((x for x in parsed if x is not None), default=None)
        else:
            speed = _parse_speed(ms)
        if speed is None:
            speed = _hw_fallback(d.get('highway', ''))
        node_max[u] = max(node_max[u], speed)
        node_max[v] = max(node_max[v], speed)

    return frozenset(n for n in G.nodes() if node_max.get(n, 0) <= max_speed_kmh)


def load_G(_inData, _params=None, stats=True, set_t=True):
    # loads graph and skim from a params paths
    if set_t:
        _params = set_t0(_params)
    _inData.G = ox.load_graphml(_params.paths.G)
    _inData.nodes = pd.DataFrame.from_dict(dict(_inData.G.nodes(data=True)), orient='index')

    # Directed driving skim
    skim = pd.read_csv(_params.paths.skim, index_col='Unnamed: 0')
    skim.columns = [int(c) for c in skim.columns]
    skim.index = [int(i) for i in skim.index]
    _inData.skim = skim

    # Undirected walking skim (optional -- falls back to runtime computation in make_skims)
    walk_path = _params.paths.get('skim_walk', None)
    if walk_path and os.path.exists(walk_path):
        skim_walk = pd.read_csv(walk_path, index_col='Unnamed: 0')
        skim_walk.columns = [int(c) for c in skim_walk.columns]
        skim_walk.index = [int(i) for i in skim_walk.index]
        _inData.walk_dist = skim_walk

    # Freeflow ride-time skim (optional -- pre-computed per-edge travel times in seconds)
    # When present, make_skims() uses this as skims.ride_freeflow instead of dist/flat_speed.
    ride_time_path = _params.paths.get('skim_ride_time', None)
    if ride_time_path and os.path.exists(ride_time_path):
        skim_ride_time = pd.read_csv(ride_time_path, index_col='Unnamed: 0')
        skim_ride_time.columns = [int(c) for c in skim_ride_time.columns]
        skim_ride_time.index = [int(i) for i in skim_ride_time.index]
        _inData.ride_time = skim_ride_time

    # Speed-limit node filter (optional -- None = no filtering)
    max_node_speed = _params.get('max_node_speed', None)
    if max_node_speed is not None:
        _inData.safe_nodes = _compute_safe_nodes(_inData.G, max_node_speed)
    else:
        _inData.safe_nodes = None

    if stats:
        _inData.stats = networkstats(_inData)  # calculate center of network, radius and central node
    return _inData


def download_G(inData, _params, make_skims=True):
    # uses osmnx to download the graph
    print('Downloading network for {} witn osmnx'.format(_params.city))
    inData.G = ox.graph_from_place(_params.city, network_type='drive')
    inData.nodes = pd.DataFrame.from_dict(dict(inData.G.nodes(data=True)), orient='index')
    if make_skims:
        inData.skim_generator = nx.all_pairs_dijkstra_path_length(inData.G,
                                                                  weight='length')
        inData.skim_dict = dict(inData.skim_generator)  # filled dict is more usable
        inData.skim = pd.DataFrame(inData.skim_dict).fillna(_params.simulation.dist_threshold).T.astype(
            int)  # and dataframe is more intuitive

    return inData


def save_G(inData, _params, path=None):
    # saves graph and skims to files
    ox.save_graphml(inData.G, filepath=_params.paths.G)
    inData.skim.to_csv(_params.paths.skim, chunksize=20000000)
