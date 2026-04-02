################################################################################
# Module: utils/demand.py
# Demand and supply generation for simulation
################################################################################

import math
import numpy as np
import pandas as pd

from ..traveller import travellerEvent
from ..driver import driverEvent

from .helpers import rand_node, empty_series, initialize_df


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


def _sample_request_times(params):
    """Sample request timestamps based on temporal distribution config.

    Supports three modes:
    - 'uniform': Uniform distribution across simulation window
    - 'normal': Normal distribution centered on simulation midpoint
    - 'profile': Multinomial sampling from a weighted time-slot profile

    Args:
        params: Configuration with demand_structure.temporal_distribution,
                simTime (hours), nP (number of passengers)

    Returns:
        numpy array of time offsets in seconds (relative to t0)
    """
    total_seconds = params.simulation.simTime * 60 * 60
    mode = params.demand_structure.temporal_distribution

    if mode == 'uniform':
        return np.random.uniform(-total_seconds / 2, total_seconds / 2, params.simulation.nP)
    elif mode == 'normal':
        return np.random.normal(
            total_seconds / 2,
            params.demand_structure.temporal_dispertion * total_seconds / 2,
            params.simulation.nP)
    elif mode == 'profile':
        profile = getattr(params.demand_structure, 'temporal_profile', None)
        if not profile:
            raise ValueError("temporal_distribution='profile' requires demand_structure.temporal_profile "
                             "to be a non-empty list of weights")
        weights = np.array(profile, dtype=float)
        weights /= weights.sum()
        n_slots = len(weights)
        slot_duration = total_seconds / n_slots
        counts = np.random.multinomial(params.simulation.nP, weights)
        treq_parts = []
        for slot_idx, count in enumerate(counts):
            if count == 0:
                continue
            slot_start = slot_idx * slot_duration
            offset = slot_start + np.random.uniform(0, slot_duration, count) - total_seconds / 2
            treq_parts.append(offset)
        return np.concatenate(treq_parts)
    else:
        return None


def generate_demand(_inData, _params=None, avg_speed=False):
    # generates nP requests with a given temporal and spatial distribution of origins and destinations
    # returns _inData with dataframes requests and passengers populated.

    try:
        _params.simulation.t0 = pd.to_datetime(_params.simulation.t0)
    except (ValueError, TypeError):
        pass

    df = pd.DataFrame(index=np.arange(0, _params.simulation.nP), columns=_inData.passengers.columns)
    df.status = travellerEvent.STARTS_DAY
    df.pos = _inData.nodes.sample(_params.simulation.nP, replace=True).index  # df.pos = df.apply(lambda x: rand_node(_inData.nodes), axis=1)
    _inData.passengers = df
    requests = pd.DataFrame(index=df.index, columns=_inData.requests.columns)
    distances = _inData.skim[_inData.stats['center']].to_frame().dropna()  # compute distances from center
    distances.columns = ['distance']
    distances = distances[distances['distance'] < _params.simulation.dist_threshold]
    # Exclude nodes on high-speed roads from demand sampling
    if getattr(_inData, 'safe_nodes', None) is not None:
        distances = distances[distances.index.isin(_inData.safe_nodes)]
    # apply negative exponential distributions
    distances['p_origin'] = distances['distance'].apply(lambda x:
                                                        math.exp(
                                                            _params.demand_structure.origins_dispertion * x))

    distances['p_destination'] = distances['distance'].apply(
        lambda x: math.exp(_params.demand_structure.destinations_dispertion * x))
    treq = _sample_request_times(_params)
    requests.treq = [_params.simulation.t0 + pd.Timedelta(int(_), 's') for _ in treq]
    requests.origin = list(
        distances.sample(_params.simulation.nP, weights='p_origin', replace=True).index)  # sample origin nodes from a distribution
    requests.destination = list(distances.sample(_params.simulation.nP, weights='p_destination',
                                                 replace=True).index)  # sample destination nodes from a distribution

    requests['dist'] = requests.apply(lambda request: _inData.skim.loc[request.origin, request.destination], axis=1)
    while len(requests[requests.dist >= _params.simulation.dist_threshold]) > 0:
        requests.origin = requests.apply(lambda request: (distances.sample(1, weights='p_origin').index[0]
                                                          if request.dist >= _params.simulation.dist_threshold else
                                                          request.origin),
                                         axis=1)
        requests.destination = requests.apply(lambda request: (distances.sample(1, weights='p_destination').index[0]
                                                               if request.dist >= _params.simulation.dist_threshold else
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


def prep_supply_and_demand(_inData, params):
    _inData = generate_demand(_inData, params, avg_speed=True)
    _inData.vehicles = generate_vehicles(_inData, params.simulation.nV)
    _inData.vehicles.platform = _inData.vehicles.apply(lambda x: 0, axis=1)
    _inData.passengers.platforms = _inData.passengers.apply(lambda x: [0], axis=1)
    _inData.requests['platform'] = _inData.requests.apply(lambda row: _inData.passengers.loc[row.name].platforms[0],
                                                          axis=1)

    _inData.platforms = initialize_df(_inData.platforms)
    batch_time_value = params.platform.batch_time
    _inData.platforms.loc[0, 'fare'] = 1
    _inData.platforms.loc[0, 'name'] = 'Platform'
    _inData.platforms.loc[0, 'batch_time'] = batch_time_value
    return _inData
