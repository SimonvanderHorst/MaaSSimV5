# Usage:  python tools/precompute_skims.py --city Delft
# Downloads the graph from OSMnx if {city}.graphml doesn't exist yet.
# Outputs: data/graphs/{city}_directed_driving.csv, _undirected_walk.csv, _directed_driving_time.csv

import argparse
import os
import time

import networkx as nx
import osmnx as ox
import pandas as pd

DIST_THRESHOLD = 100_000


def _all_pairs_skim(G, weight='length'):
    """All-pairs Dijkstra → DataFrame where df[dest][orig] = cost."""
    d = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    return pd.DataFrame(d).fillna(DIST_THRESHOLD).T.astype(int)


def _add_freeflow_times(G):
    """Add travel_time_freeflow edge attr from OSM speed limits."""
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    for _, _, _, data in G.edges(keys=True, data=True):
        data['travel_time_freeflow'] = data.get('travel_time', 0)
    return G


def _save_skim(df, path, label):
    df.to_csv(path)
    print(f"  {label}: {os.path.getsize(path) / 1e6:.1f} MB")


def _spot_check(G, drive_dist, drive_time):
    nodes = list(G.nodes())[:2]
    a, b = nodes[0], nodes[1]
    for orig, dest in [(a, b), (b, a)]:
        nx_d = nx.shortest_path_length(G, orig, dest, weight='length')
        skim_d = drive_dist[dest][orig]
        skim_t = drive_time[dest][orig]
        assert skim_d == int(nx_d), f"Distance mismatch {orig}->{dest}"
        print(f"  {orig}->{dest}: {skim_d}m, {skim_t}s ({skim_d / skim_t * 3.6:.1f} km/h)")
    print("  Spot-check: PASS")


def precompute_skims(city=None, name=None, bbox=None, root=None, force=False):
    name = name or city
    root = root or os.getcwd()
    graphs = os.path.join(root, 'data', 'graphs')
    graph_path = os.path.join(graphs, f'{name}.graphml')

    os.makedirs(graphs, exist_ok=True)

    if force and os.path.exists(graph_path):
        os.remove(graph_path)
        print(f"Removed existing {graph_path} (--force)")

    if os.path.exists(graph_path):
        G = ox.load_graphml(graph_path)
        print(f"Loaded {graph_path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print(f"Graph not found at {graph_path}, downloading from OSMnx...")
        if bbox:
            # Download from bounding box: (north, south, east, west)
            print(f"Downloading from bbox {bbox}...")
            G = ox.graph_from_bbox(*bbox, network_type='drive')
        elif isinstance(city, list):
            # Download each place with truncate_by_edge=True to keep boundary-
            # crossing roads, then compose.  Shared OSM nodes at the boundary
            # keep the combined graph connected.
            place_list = [p.strip() for p in city]
            print(f"Downloading {place_list} and composing...")
            parts = []
            for place in place_list:
                g = ox.graph_from_place(place, network_type='drive',
                                        truncate_by_edge=True)
                parts.append(g)
                print(f"  {place}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
            G = nx.compose_all(parts)
            shared = set.intersection(*(set(g.nodes()) for g in parts))
            print(f"  Combined: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"
                  f" ({len(shared)} shared boundary nodes)")
        else:
            G = ox.graph_from_place(city, network_type='drive')
        ox.save_graphml(G, graph_path)
        print(f"Saved graph to {graph_path}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    n = G.number_of_nodes()

    # Directed driving distances (metres)
    print(f"\nDriving distances ({n}x{n})...")
    t0 = time.time()
    drive_dist = _all_pairs_skim(G, weight='length')
    print(f"  Computed in {time.time() - t0:.1f}s")
    _save_skim(drive_dist, os.path.join(graphs, f'{name}_directed_driving.csv'), 'Saved')

    # Undirected walking distances (metres)
    print(f"\nWalking distances ({n}x{n})...")
    t0 = time.time()
    walk_dist = _all_pairs_skim(G.to_undirected(), weight='length')
    print(f"  Computed in {time.time() - t0:.1f}s")
    _save_skim(walk_dist, os.path.join(graphs, f'{name}_undirected_walk.csv'), 'Saved')

    # Directed freeflow travel times (seconds)
    print(f"\nFreeflow times ({n}x{n})...")
    G = _add_freeflow_times(G)
    t0 = time.time()
    drive_time = _all_pairs_skim(G, weight='travel_time_freeflow')
    print(f"  Computed in {time.time() - t0:.1f}s")
    _save_skim(drive_time, os.path.join(graphs, f'{name}_directed_driving_time.csv'), 'Saved')

    # Verify
    print("\nVerification:")
    _spot_check(G, drive_dist, drive_time)
    print("\nDone.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Pre-compute skim matrices for a city graph.')
    p.add_argument('--city', default=None,
                   help='City name(s), pipe-separated for multi-place (e.g. "Delft, Netherlands|Rijswijk, Netherlands")')
    p.add_argument('--bbox', default=None,
                   help='Bounding box (north,south,east,west) instead of city name.')
    p.add_argument('--name', default=None,
                   help='Output stem (default: city). Use when city is a list.')
    p.add_argument('--root', default=None, help='MaaSSim project root (default: cwd)')
    p.add_argument('--force', action='store_true', help='Re-download graph even if .graphml exists')
    args = p.parse_args()

    if not args.city and not args.bbox:
        p.error("Either --city or --bbox is required")

    city_arg = None
    bbox_arg = None
    if args.city:
        cities = args.city.split('|')
        city_arg = cities if len(cities) > 1 else cities[0]
    if args.bbox:
        bbox_arg = tuple(float(x) for x in args.bbox.split(','))

    precompute_skims(city=city_arg, bbox=bbox_arg, name=args.name, root=args.root, force=args.force)
