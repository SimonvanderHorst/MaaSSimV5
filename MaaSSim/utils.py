################################################################################
# Module: utils.py
# Reusable functions and methods used throughout the simulator
# Rafal Kucharski @ TU Delft
################################################################################

import pandas as pd
from dotmap import DotMap
import math
import random
import numpy as np
import os

import osmnx as ox
import osmnx as ox
import networkx as nx
import json

from .traveller import travellerEvent
from .driver import driverEvent


# Rider heterogeneity: per-rider beta_zero from Pellegrini & Fielbaum (2025) ASC DTD distribution for AUD, scaled to EUR by 0.60 factor (based on observed relative differences in mean values across datasets).
_ASC_DTD_BINS_AUD = [
    (-3.0, -2.0, 0.200), (-2.0, -1.0, 0.203), (-1.0, 0.0, 0.200),
    (0.0, 1.0, 0.107), (1.0, 2.0, 0.058), (2.0, 3.0, 0.040),
    (3.0, 4.0, 0.008), (4.0, 5.0, 0.008), (5.0, 8.0, 0.003),
    (8.0, 10.0, 0.004), (10.0, 11.0, 0.015), (11.0, 12.0, 0.031),
    (12.0, 13.0, 0.033), (13.0, 14.5, 0.033),
]
_asc_lows = np.array([b[0] for b in _ASC_DTD_BINS_AUD])
_asc_highs = np.array([b[1] for b in _ASC_DTD_BINS_AUD])
_asc_probs = np.array([b[2] for b in _ASC_DTD_BINS_AUD])
_asc_probs = _asc_probs / _asc_probs.sum()


def _sample_beta_zero(n, seed=None):
    """Sample per-rider beta_zero (EUR) from empirical ASC DTD distribution."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(_asc_probs), size=n, p=_asc_probs)
    u = rng.uniform(0, 1, size=n)
    return (_asc_lows[idx] + u * (_asc_highs[idx] - _asc_lows[idx])) * 0.60  # AUD->EUR


def rand_node(df):
    # returns a random node of a graph
    return df.loc[random.choice(df.index.tolist())].name


def generic_generator(generator, n):
    # to create multiple passengers/vehicles/etc
    return pd.concat([generator(i) for i in range(1, n + 1)], axis=1, keys=range(1, n + 1)).T


def empty_series(df, name=None):
    # returns empty Series from a given DataFrame, to be used for consistency of adding new rows to DataFrames
    if name is None:
        name = len(df.index) + 1
    return pd.Series(index=df.columns, name=name)


def initialize_df(df):
    # deletes rows in DataFrame and leaves the columns and index
    # returns empty DataFrame
    if type(df) == pd.core.frame.DataFrame:
        cols = df.columns
    else:
        cols = list(df.keys())
    df = pd.DataFrame(columns=cols)
    df.index.name = 'id'
    return df


def get_config(path, root_path=None, set_t0=False):
    """
    reads a .json file with MaaSSim configuration
    use as: params = get_config(config.json)
    :param path:
    :param root_path: adjust the paths with the main path while reading (used mainly for Travis tests on linux server)
    :param set_t0: adjust the t0 string to pandas datetime
    :return: params DotMap
    """
    with open(path) as json_file:
        data = json.load(json_file)
        params = DotMap(data)
    if root_path is not None:
        params.paths.G = os.path.join(root_path, params.paths.G)  # graphml of a current .city
        params.paths.skim = os.path.join(root_path, params.paths.skim)  # csv with a skim between the nodes of the .city
    if set_t0:
        params.t0 = pd.to_datetime(params.t0)
    return params


def save_config(_params, path=None):
    if path is None:
        path = os.path.join(_params.paths.params, _params.NAME + ".json")
    with open(path, "w") as write_file:
        json.dump(_params, write_file)


def set_t0(_params, now=True):
    if now:
        _params.t0 = pd.Timestamp.now().floor('1s')
    else:
        _params.t0 = pd.to_datetime(_params.t0)
    return _params


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

    maxspeed parsing: numeric string → km/h; "N mph" → km/h; list → max value;
    non-parseable/missing → highway-type fallback.
    Highway fallbacks: motorway/trunk → 100, primary → 80, secondary → 60,
                       tertiary/unclassified → 50, residential/service/else → 30.
    """
    from collections import defaultdict

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

    # Undirected walking skim (optional — falls back to runtime computation in make_skims)
    walk_path = _params.paths.get('skim_walk', None)
    if walk_path and os.path.exists(walk_path):
        skim_walk = pd.read_csv(walk_path, index_col='Unnamed: 0')
        skim_walk.columns = [int(c) for c in skim_walk.columns]
        skim_walk.index = [int(i) for i in skim_walk.index]
        _inData.walk_dist = skim_walk

    # Freeflow ride-time skim (optional — pre-computed per-edge travel times in seconds)
    # When present, make_skims() uses this as skims.ride_freeflow instead of dist/flat_speed.
    ride_time_path = _params.paths.get('skim_ride_time', None)
    if ride_time_path and os.path.exists(ride_time_path):
        skim_ride_time = pd.read_csv(ride_time_path, index_col='Unnamed: 0')
        skim_ride_time.columns = [int(c) for c in skim_ride_time.columns]
        skim_ride_time.index = [int(i) for i in skim_ride_time.index]
        _inData.ride_time = skim_ride_time

    # Speed-limit node filter (optional — None = no filtering)
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
        inData.skim = pd.DataFrame(inData.skim_dict).fillna(_params.dist_threshold).T.astype(
            int)  # and dataframe is more intuitive

    return inData


def save_G(inData, _params, path=None):
    # saves graph and skims to files
    ox.save_graphml(inData.G, filepath=_params.paths.G)
    inData.skim.to_csv(_params.paths.skim, chunksize=20000000)


def generate_vehicles(_inData, nV):
    """
    generates single vehicle (database row with structure defined in DataStructures)
    index is consecutive number if dataframe
    position is random graph node
    status is IDLE
    """
    vehs = list()
    for i in range(nV):
        vehs.append(empty_series(_inData.vehicles, name=i))

    vehs = pd.concat(vehs, axis=1, keys=range(1, nV + 1)).T
    vehs.event = driverEvent.STARTS_DAY
    vehs.platform = 0
    vehs.shift_start = 0
    vehs.shift_end = 60 * 60 * 24
    vehs.pos = vehs.pos.apply(lambda x: int(rand_node(_inData.nodes)))

    return vehs


def generate_demand(_inData, _params=None, avg_speed=False):
    # generates nP requests with a given temporal and spatial distribution of origins and destinations
    # returns _inData with dataframes requests and passengers populated.

    try:
        _params.t0 = pd.to_datetime(_params.t0)
    except:
        pass

    df = pd.DataFrame(index=np.arange(0, _params.nP), columns=_inData.passengers.columns)
    df.status = travellerEvent.STARTS_DAY
    df.pos = _inData.nodes.sample(_params.nP).index  # df.pos = df.apply(lambda x: rand_node(_inData.nodes), axis=1)
    _inData.passengers = df
    requests = pd.DataFrame(index=df.index, columns=_inData.requests.columns)
    distances = _inData.skim[_inData.stats['center']].to_frame().dropna()  # compute distances from center
    distances.columns = ['distance']
    distances = distances[distances['distance'] < _params.dist_threshold]
    # Exclude nodes on high-speed roads from demand sampling
    if getattr(_inData, 'safe_nodes', None) is not None:
        distances = distances[distances.index.isin(_inData.safe_nodes)]
    # apply negative exponential distributions
    distances['p_origin'] = distances['distance'].apply(lambda x:
                                                        math.exp(
                                                            _params.demand_structure.origins_dispertion * x))

    distances['p_destination'] = distances['distance'].apply(
        lambda x: math.exp(_params.demand_structure.destinations_dispertion * x))
    if _params.demand_structure.temporal_distribution == 'uniform':
        treq = np.random.uniform(-_params.simTime * 60 * 60 / 2, _params.simTime * 60 * 60 / 2,
                                 _params.nP)  # apply uniform distribution on request times
    elif _params.demand_structure.temporal_distribution == 'normal':
        treq = np.random.normal(_params.simTime * 60 * 60 / 2,
                                _params.demand_structure.temporal_dispertion * _params.simTime * 60 * 60 / 2,
                                _params.nP)  # apply normal distribution on request times
    elif _params.demand_structure.temporal_distribution == 'profile':
        profile = getattr(_params.demand_structure, 'temporal_profile', None)
        if not profile:
            raise ValueError("temporal_distribution='profile' requires demand_structure.temporal_profile "
                             "to be a non-empty list of weights")
        weights = np.array(profile, dtype=float)
        weights /= weights.sum()
        n_slots = len(weights)
        total_seconds = _params.simTime * 60 * 60
        slot_duration = total_seconds / n_slots
        counts = np.random.multinomial(_params.nP, weights)
        treq_parts = []
        for slot_idx, count in enumerate(counts):
            if count == 0:
                continue
            slot_start = slot_idx * slot_duration
            offset = slot_start + np.random.uniform(0, slot_duration, count) - total_seconds / 2
            treq_parts.append(offset)
        treq = np.concatenate(treq_parts)
    else:
        treq = None
    requests.treq = [_params.t0 + pd.Timedelta(int(_), 's') for _ in treq]
    requests.origin = list(
        distances.sample(_params.nP, weights='p_origin', replace=True).index)  # sample origin nodes from a distribution
    requests.destination = list(distances.sample(_params.nP, weights='p_destination',
                                                 replace=True).index)  # sample destination nodes from a distribution

    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    while len(requests[requests.dist >= _params.dist_threshold]) > 0:
        requests.origin = requests.apply(lambda request: (distances.sample(1, weights='p_origin').index[0]
                                                          if request.dist >= _params.dist_threshold else
                                                          request.origin),
                                         axis=1)
        requests.destination = requests.apply(lambda request: (distances.sample(1, weights='p_destination').index[0]
                                                               if request.dist >= _params.dist_threshold else
                                                               request.destination),
                                              axis=1)
        requests.dist = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)

    requests['ttrav'] = requests.apply(lambda request: pd.Timedelta(request.dist, 's').floor('s'), axis=1)
    # requests.ttrav = pd.to_timedelta(requests.ttrav)
    if avg_speed:
        requests.ttrav = (pd.to_timedelta(requests.ttrav) / _params.speeds.ride).dt.floor('1s')
    requests.tarr = [request.treq + request.ttrav for _, request in requests.iterrows()]
    requests = requests.sort_values('treq')
    requests.index = df.index
    requests.pax_id = df.index
    requests.shareable = False

    _inData.requests = requests
    _inData.passengers.pos = _inData.requests.origin

    _inData.passengers.platforms = _inData.passengers.platforms.apply(lambda x: [0])

    # Rider heterogeneity: assign per-rider beta_zero for PUDO behavioral model.
    # Only active when both rider_heterogeneity=true AND behavioral.enabled=true.
    if (_params.get('pudo', {}).get('rider_heterogeneity', False) and
            _params.get('pudo', {}).get('behavioral', {}).get('enabled', False)):
        _inData.passengers['beta_zero'] = _sample_beta_zero(len(_inData.passengers))

    return _inData


def read_requests_csv(inData, path):
    from MaaSSim.data_structures import structures
    inData.requests = pd.read_csv(path, index_col=1)
    inData.requests.treq = pd.to_datetime(inData.requests.treq)
    inData.requests['pax_id'] = inData.requests.index.copy()
    inData.requests.ttrav = pd.to_timedelta(inData.requests.ttrav)
    inData.passengers = pd.DataFrame(index=inData.requests.index, columns=structures.passengers.columns)
    inData.passengers['pax_id'] = inData.passengers.index.copy()
    inData.passengers.pos = inData.requests.origin.copy()
    inData.passengers.platforms = inData.passengers.platforms.apply(lambda x: [0])
    return inData

def read_vehicle_positions(inData, path):
    inData.vehicles = pd.read_csv(path, index_col=0)
    return inData


def make_config_paths(params, main=None, rel = False):
    # call it whenever you change a city name, or main path
    if main is None:
        if rel:
            main = '../..'
        else:
            main = os.path.join(os.getcwd(), "../..")
    if rel:
        params.paths.main = main
    else:
        params.paths.main = os.path.abspath(main)  # main repo folder


    params.paths.data = os.path.join(params.paths.main, 'data')  # data folder (not synced with repo)
    params.paths.params = os.path.join(params.paths.data, 'configs')
    params.paths.postcodes = os.path.join(params.paths.data, 'postcodes',
                                          "PC4_Nederland_2015.shp")  # PCA4 codes shapefile
    params.paths.albatross = os.path.join(params.paths.data, 'albatross')  # albatross data
    params.paths.sblt = os.path.join(params.paths.data, 'sblt')  # sblt results
    params.paths.G = os.path.join(params.paths.data, 'graphs',
                                  params.city.split(",")[0] + ".graphml")  # graphml of a current .city
    params.paths.skim = os.path.join(params.paths.main, 'data', 'graphs',
                                     params.city.split(",")[0] + "_directed_driving.csv")
    params.paths.skim_walk = os.path.join(params.paths.main, 'data', 'graphs',
                                          params.city.split(",")[0] + "_undirected_walk.csv")
    params.paths.skim_ride_time = os.path.join(params.paths.main, 'data', 'graphs',
                                               params.city.split(",")[0] + "_directed_driving_time.csv")
    params.paths.NYC = os.path.join(params.paths.main, 'data',
                                    'fhv_tripdata_2018-01.csv')  # csv with a skim between the nodes of the .city
    return params


def prep_supply_and_demand(_inData, params):
    _inData = generate_demand(_inData, params, avg_speed=True)
    _inData.vehicles = generate_vehicles(_inData, params.nV)
    _inData.vehicles.platform = _inData.vehicles.apply(lambda x: 0, axis=1)
    _inData.passengers.platforms = _inData.passengers.apply(lambda x: [0], axis=1)
    _inData.requests['platform'] = _inData.requests.apply(lambda row: _inData.passengers.loc[row.name].platforms[0],
                                                          axis=1)

    _inData.platforms = initialize_df(_inData.platforms)
    batch_time_value = getattr(params, 'batch_time', 60)
    _inData.platforms.loc[0, 'fare'] = 1
    _inData.platforms.loc[0, 'name'] = 'Platform'
    _inData.platforms.loc[0, 'batch_time'] = batch_time_value
    return _inData


#################
# PARALLEL RUNS #
#################


def test_space():
    # to see if code works
    test_space = DotMap()
    test_space.nP = [30, 40]  # number of requests per sim time
    test_space.nV = [10, 20]  # number of requests per sim time
    return test_space


def slice_space(s, replications=1, _print=False):
    # util to feed the np.optimize.brute with a search space
    def sliceme(l):
        return slice(0, len(l), 1)

    ret = list()
    sizes = list()
    size = 1
    for key in s.keys():
        ret += [sliceme(s[key])]
        sizes += [len(s[key])]
        size *= sizes[-1]
    if replications > 1:
        sizes += [replications]
        size *= sizes[-1]
        ret += [slice(0, replications, 1)]
    if _print:
        print('Search space to explore of dimensions {} and total size of {}'.format(sizes, size))
    return tuple(ret)


def collect_results(path):
    from pathlib import Path
    import zipfile
    collections = DotMap()
    first = True
    for archive in Path(path).rglob('*.zip'):
        zf = zipfile.ZipFile(archive)
        if first:
            for file in zf.namelist():
                collections[file[:-4]] = list()
            first = False
        for file in zf.namelist():
            df = pd.read_csv(zf.open(file))
            for key in archive.stem.split('-')[1:]:
                field, value = key.split('_')
                df[field] = value

            collections[file[:-4]].append(df)
    for key in collections.keys():
        collections[key] = pd.concat(collections[key])
    return collections
