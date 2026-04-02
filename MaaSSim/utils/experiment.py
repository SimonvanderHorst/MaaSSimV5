################################################################################
# Module: utils/experiment.py
# Parallel run and experiment utilities
################################################################################

import random
import numpy as np
import pandas as pd
from dotmap import DotMap

from MaaSSim.data_structures import structures
from .config import get_config
from .network import load_G
from .demand import generate_demand, generate_vehicles
from .helpers import initialize_df


def generate_shared_demand(config_path, seed=None, shared_network=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    params = get_config(config_path)
    inData = structures.copy()
    if shared_network is not None:
        # reuse pre-loaded network/skims to save memory
        for attr in ('G', 'nodes', 'skim', 'walk_dist', 'ride_time', 'stats', 'safe_nodes'):
            if hasattr(shared_network, attr):
                setattr(inData, attr, getattr(shared_network, attr))
    else:
        inData = load_G(inData, params, stats=True)
    inData = generate_demand(inData, params, avg_speed=True)
    inData.vehicles = generate_vehicles(inData, params.simulation.nV)
    if len(inData.platforms) == 0:
        inData.platforms = initialize_df(inData.platforms)
        inData.platforms.loc[0, 'fare'] = params.platform.fare_per_km
        inData.platforms.loc[0, 'fare_per_min'] = params.platform.fare_per_min
        inData.platforms.loc[0, 'name'] = 'Platform'
        inData.platforms.loc[0, 'batch_time'] = params.platform.batch_time
    from MaaSSim.simulators import _prep_rides
    inData = _prep_rides(inData)
    return inData, params


def copy_indata(inData_base):
    c = DotMap()
    c.G = inData_base.G
    c.nodes = inData_base.nodes
    c.skim = inData_base.skim
    c.stats = inData_base.stats
    c.passengers = inData_base.passengers.copy()
    c.requests = inData_base.requests.copy()
    c.vehicles = inData_base.vehicles.copy()
    c.platforms = inData_base.platforms.copy()
    if 'sim_schedule' in c.requests.columns:
        c.requests['sim_schedule'] = c.requests['sim_schedule'].apply(
            lambda x: x.copy() if isinstance(x, pd.DataFrame) else x
        )
    for attr in ('walk_dist', 'ride_time'):
        if isinstance(getattr(inData_base, attr, None), pd.DataFrame):
            setattr(c, attr, getattr(inData_base, attr))
    if hasattr(inData_base, 'safe_nodes'):
        c.safe_nodes = inData_base.safe_nodes
    return c


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
