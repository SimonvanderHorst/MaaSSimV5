################################################################################
# Module: utils/config.py
# Configuration loading, saving, and path management
################################################################################

import os
import json
import pandas as pd
from dotmap import DotMap


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
        params.simulation.t0 = pd.to_datetime(params.simulation.t0)
    return params


def save_config(_params, path=None):
    if path is None:
        path = os.path.join(_params.paths.params, _params.NAME + ".json")
    with open(path, "w") as write_file:
        json.dump(_params, write_file)


def set_t0(_params, now=True):
    if now:
        _params.simulation.t0 = pd.Timestamp.now().floor('1s')
    else:
        _params.simulation.t0 = pd.to_datetime(_params.simulation.t0)
    return _params


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
