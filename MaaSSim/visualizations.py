################################################################################
# Module: utils.py
# Reusable functions to visualize and plot MaaSSim results
# Rafal Kucharski @ TU Delft
################################################################################

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import osmnx as ox
import networkx as nx

from matplotlib.collections import LineCollection


# ---------------------------------------------------------------------------
# PUDO route comparison helpers
# ---------------------------------------------------------------------------

def _get_node_xy(G, node_id):
    """Extract (x, y) coordinates from a graph node."""
    try:
        return G.nodes[node_id]['x'], G.nodes[node_id]['y']
    except (KeyError, TypeError):
        return None, None


_OUTCOME_TITLE_LABELS = {
    'pudo_accepted': 'PUDO Accepted',
    'd2d_fallback_driver': 'D2D Fallback (driver rejected PUDO)',
    'd2d_fallback_rider': 'D2D Fallback (rider rejected PUDO)',
    'driver_s1_rejected': 'Driver S1 Rejected (dispatch/fare)',
    'driver_s2_rejected': 'Driver S2 Rejected (behavioral)',
    'rider_s1_rejected': 'Rider S1 Rejected (ratio check)',
    'rider_s2_rejected': 'Rider S2 Rejected (behavioral)',
}

_OUTCOME_FILE_TAGS = {
    'pudo_accepted': 'accepted',
    'd2d_fallback_driver': 'fb_driver',
    'd2d_fallback_rider': 'fb_rider',
    'driver_s1_rejected': 'rej_drv_s1',
    'driver_s2_rejected': 'rej_drv_s2',
    'rider_s1_rejected': 'rej_rid_s1',
    'rider_s2_rejected': 'rej_rid_s2',
}

_OUTCOME_COLORS = {
    'pudo_accepted': 'green',
    'd2d_fallback_driver': 'darkorange',
    'd2d_fallback_rider': 'darkorange',
    'driver_s1_rejected': 'red',
    'driver_s2_rejected': 'red',
    'rider_s1_rejected': 'red',
    'rider_s2_rejected': 'red',
}


def add_route(G, ax, route, color='grey', lw=2, alpha=0.5, linestyle='solid'):
    # plots route on the graph alrready plotted on ax
    edge_nodes = list(zip(route[:-1], route[1:]))
    lines = []
    for u, v in edge_nodes:
        # if there are parallel edges, select the shortest in length
        data = min(G.get_edge_data(u, v).values(), key=lambda x: x['length'])
        # if it has a geometry attribute (ie, a list of line segments)
        if 'geometry' in data:
            # add them to the list of lines to plot
            xs, ys = data['geometry'].xy
            lines.append(list(zip(xs, ys)))
        else:
            # if it doesn't have a geometry attribute, the edge is a straight
            # line from node to node
            x1 = G.nodes[u]['x']
            y1 = G.nodes[u]['y']
            x2 = G.nodes[v]['x']
            y2 = G.nodes[v]['y']
            line = [(x1, y1), (x2, y2)]
            lines.append(line)
    lc = LineCollection(lines, colors=color, linewidths=lw, alpha=alpha, linestyles=linestyle, zorder=3)
    ax.add_collection(lc)


def plot_map_rides(G, ts, light=True, m_size=30, lw=3):

    fig, ax = ox.plot_graph(G, figsize=(15, 15), node_size=0, edge_linewidth=0.3,
                            show=False, close=False,
                            edge_color='grey')

    colors = {1: 'orange', 2: 'teal', 3: 'maroon', 4: 'black', 5: 'green'}
    for t in ts:
        deg = t.req_id.nunique() - 1
        for i in t.req_id.dropna().unique():
            r = t[t.req_id == i]
            o = r[r.od == 'o'].iloc[0].node
            d = r[r.od == 'd'].iloc[0].node
            ax.scatter(G.nodes[o]['x'], G.nodes[o]['y'], s=m_size, c='black', marker='x')
            ax.scatter(G.nodes[d]['x'], G.nodes[d]['y'], s=m_size, c='black', marker='v')

            if not light:
                ax.annotate('o' + str(i), (G.nodes[o]['x'] * 1.0002, G.nodes[o]['y'] * 1.00001))
                ax.annotate('d' + str(i), (G.nodes[d]['x'] * 1.0002, G.nodes[d]['y'] * 1.00001))
                route = nx.shortest_path(G, o, d, weight='length')

                add_route(G, ax, route, color='black', lw=int(lw / 2), alpha=0.5)

        routes = list()  # ride segments
        o = t.node.dropna().values[0]

        for d in t.node.dropna().values[1:]:
            routes.append(nx.shortest_path(G, o, d, weight='length'))
            o = d
        for route in routes:
            add_route(G, ax, route, color=colors[deg], lw=lw, alpha=0.7)


def plot_demand(_inData, t0=None, vehicles=False, s=10, params=None):
    import matplotlib.pyplot as plt
    if t0 is None:
        t0 = _inData.requests.treq.mean()

    # plot osmnx graph, its center, scattered nodes of requests origins and destinations
    # plots requests temporal distribution
    fig, ax = plt.subplots(1, 3)
    ((t0 - _inData.requests.treq) / np.timedelta64(1, 'h')).plot.kde(title='Temporal distribution', ax=ax[0])
    (_inData.requests.ttrav / np.timedelta64(1, 'm')).plot(kind='box', title='Trips travel times [min]', ax=ax[1])
    _inData.requests.dist.plot(kind='box', title='Trips distance [m]', ax=ax[2])
    # (inData.requests.ttrav / np.timedelta64(1, 'm')).describe().to_frame().T
    plt.show()
    fig, ax = ox.plot_graph(_inData.G, figsize=(15, 15), node_size=0, edge_linewidth=0.5,
                            show=False, close=False,
                            edge_color='grey', bgcolor='white')
    for _, r in _inData.requests.iterrows():
        ax.scatter(_inData.G.nodes[r.origin]['x'], _inData.G.nodes[r.origin]['y'], c='green', s=s, marker='D')
        ax.scatter(_inData.G.nodes[r.destination]['x'], _inData.G.nodes[r.destination]['y'], c='orange', s=s)
    if vehicles:
        for _, r in _inData.vehicles.iterrows():
            ax.scatter(_inData.G.nodes[r.pos]['x'], _inData.G.nodes[r.pos]['y'], c='blue', s=s, marker='x')
    ax.scatter(_inData.G.nodes[_inData.stats['center']]['x'], _inData.G.nodes[_inData.stats['center']]['y'], c='red',
               s=10 * s, marker='+')
    plt.title(
        'Demand in {} with origins marked in green, destinations in orange and vehicles in blue'.format(params.city))
    plt.show()


def plot_veh_sim(sim, veh_id):
    t =  sim.runs[0].rides[sim.runs[0].rides.veh == veh_id]
    return plot_veh(sim.inData.G, t)

def plot_veh(G, t, m_size=30, lw=2, annotate = False):
    """
    plots a trace of vehicle rides on a graph
    :param G: osmnx graph (inData.G, or sim.inData.G)
    :param t: trips
    :param m_size: marker_size
    :param lw: linew weight
    :return: None
    """

    fig, ax = ox.plot_graph(G, figsize=(10, 10), node_size=0, edge_linewidth=0.3,
                            show=False, close=False,
                            edge_color='grey', bgcolor='white')

    t['node'] = t.pos

    degs = t.apply(lambda x: min(2,len(x.paxes)), axis=1)

    color_empty = 'lightsalmon'
    color_full = 'sienna'
    alphas = [1, 0.4, 1]
    colors = ['black', 'tab:blue', 'tab:green']

    routes = list()  # ride segments
    o = t.node.dropna().values[0]
    ax.scatter(G.nodes[o]['x'], G.nodes[o]['y'], s=m_size, c='tab:blue', marker='o')
    row = t.iloc[0]
    if annotate:
        ax.annotate("t:{}, paxes: {} {}".format(int(row.t), row.paxes, row.event),
                    (G.nodes[o]['x'] * 1.0002, G.nodes[o]['y'] * 1.00001))

    for row in t.iloc[1:].iterrows():
        d = row[1].pos
        if o != d:
            ax.scatter(G.nodes[d]['x'], G.nodes[d]['y'], s=m_size, c='tab:blue', marker='o')
            if annotate:
                ax.annotate("t:{}, paxes: {} {}".format(int(row[1].t), row[1].paxes, row[1].event),
                            (G.nodes[d]['x'] * 1.0002, G.nodes[d]['y'] * 1.00001))
        routes.append(nx.shortest_path(G, o, d, weight='length'))
        o = d
    for i, route in enumerate(routes):
        add_route(G, ax, route, color=colors[degs[i+1]], lw=lw*(1 + 3*degs[i + 1]),
                  alpha=alphas[degs[i+1]])
        # add_route(G, ax, route, color=color_empty if degs[i + 1] == 0 else color_full, lw=lw + degs[i + 1] ** 2 / 2,
        #           alpha=0.9)
    return ax

def plot_trip(sim, pax_id, run_id=None):
    from MaaSSim.traveller import travellerEvent
    G = sim.inData.G
    # space time
    if run_id is None:
        run_id = list(sim.runs.keys())[-1]
    df = sim.runs[run_id].trips
    df = df[df.pax == pax_id]
    df['status_num'] = df.apply(lambda x: travellerEvent[x.event].value, axis=1)

    fig, ax = plt.subplots()
    df.plot(x='t', y='status_num', ax=ax, drawstyle="steps-post")
    ax.yticks = plt.yticks(df.index, df.event)
    plt.show()

    # map
    routes = list()
    prev_node = df.pos.iloc[0]
    for node in df.pos[1:]:
        if prev_node != node:
            routes.append(nx.shortest_path(G, prev_node, node, weight='length'))
            routes.append(nx.shortest_path(G, prev_node, node, weight='length'))
        prev_node = node
    ox.plot_graph_routes(G, routes, node_size=0,
                         edge_color='grey', bgcolor='white')
    return ax


# =============================================================================
# Route Comparison — D2D vs PUDO end-to-end journey
# =============================================================================

def plot_route_comparison(decision_log, sim, req_ids, output_dir,
                          outcome_labels=None):
    """Side-by-side D2D vs PUDO route on the road network for each request.

    Left panel: D2D route (Vehicle -> Origin -> Destination).
    Right panel: PUDO route (Vehicle -> Pickup -> Dropoff) + walking segments.
    Both panels share the same map extent for direct visual comparison.

    Args:
        decision_log: Dict from PudoDecisionLog.to_dict() (or merged)
        sim: Simulator object (for graph, requests, vehicles)
        req_ids: List of request IDs to visualize
        output_dir: Directory to save per-request PNGs
        outcome_labels: Optional dict {req_id: outcome_str} for title annotation
    """
    G = sim.inData.G
    G_walk = G.to_undirected()
    assignments = decision_log.get('assignments', [])
    best_edges = decision_log.get('best_per_edge', [])

    # Build lookups
    assigned_info = {}
    for a in assignments:
        assigned_info[int(a['req_id'])] = {
            'veh_id': int(a['veh_id']),
            'veh_pos': int(a['veh_pos']),
            'pickup': int(a['pickup_node']),
            'dropoff': int(a['dropoff_node']),
            'origin': int(a['origin']),
            'destination': int(a['destination']),
            'savings': float(a.get('savings', 0)),
            'cost_d2d': float(a.get('cost_d2d', 0)),
            'cost_pudo': float(a.get('cost_pudo', 0)),
        }

    edge_lookup = {}
    best_edge_by_req = {}
    for e in best_edges:
        edge_lookup[(e['veh_id'], e['req_id'])] = e
        r_id = int(e['req_id'])
        if r_id not in best_edge_by_req or e.get('savings', 0) > best_edge_by_req[r_id].get('savings', 0):
            best_edge_by_req[r_id] = e

    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for req_id in req_ids:
        assign = assigned_info.get(int(req_id))
        # Fallback to best_per_edge for rejected requests
        if assign is None:
            edge = best_edge_by_req.get(int(req_id))
            if edge is None:
                continue
            req_data = sim.inData.requests.loc[int(req_id)]
            veh_id_fb = edge['veh_id']
            try:
                veh_pos_fb = int(sim.inData.vehicles.loc[veh_id_fb, 'pos'])
            except (KeyError, IndexError):
                continue
            assign = {
                'veh_id': veh_id_fb,
                'veh_pos': veh_pos_fb,
                'pickup': int(edge['best_pickup']),
                'dropoff': int(edge['best_dropoff']),
                'origin': int(req_data.origin),
                'destination': int(req_data.destination),
                'savings': float(edge.get('savings', 0)),
                'cost_d2d': float(edge.get('cost_d2d', 0)),
                'cost_pudo': float(edge.get('cost_pudo', 0)),
            }

        veh_pos = assign['veh_pos']
        veh_id = assign['veh_id']
        origin = assign['origin']
        destination = assign['destination']
        pickup = assign['pickup']
        dropoff = assign['dropoff']

        edge_data = edge_lookup.get((veh_id, req_id), {})
        components = edge_data.get('components', {})

        # Get coordinates for all key nodes
        nodes = {'veh': veh_pos, 'origin': origin, 'destination': destination,
                 'pickup': pickup, 'dropoff': dropoff}
        coords = {}
        skip = False
        for name, nid in nodes.items():
            x, y = _get_node_xy(G, nid)
            if x is None:
                skip = True
                break
            coords[name] = (x, y)
        if skip:
            continue

        # Compute shared map extent
        all_xs = [c[0] for c in coords.values()]
        all_ys = [c[1] for c in coords.values()]

        route_pairs = [
            (veh_pos, origin, G),
            (origin, destination, G),
            (veh_pos, pickup, G),
            (pickup, dropoff, G),
            (origin, pickup, G_walk),
            (dropoff, destination, G_walk),
        ]
        for src, dst, graph in route_pairs:
            if src == dst:
                continue
            try:
                route = nx.shortest_path(graph, src, dst, weight='length')
                for node in route:
                    all_xs.append(graph.nodes[node]['x'])
                    all_ys.append(graph.nodes[node]['y'])
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                pass

        pad_x = (max(all_xs) - min(all_xs)) * 0.15 + 0.001
        pad_y = (max(all_ys) - min(all_ys)) * 0.15 + 0.001
        xlim = (min(all_xs) - pad_x, max(all_xs) + pad_x)
        ylim = (min(all_ys) - pad_y, max(all_ys) + pad_y)

        fig, (ax_d2d, ax_pudo) = plt.subplots(1, 2, figsize=(26, 12))

        def _plot_segment(ax, src, dst, color, lw, alpha, linestyle,
                          label_text=None, graph=None):
            _G = graph if graph is not None else G
            try:
                route = nx.shortest_path(_G, src, dst, weight='length')
                add_route(_G, ax, route, color=color, lw=lw, alpha=alpha,
                          linestyle=linestyle)
                if label_text:
                    mid_node = route[len(route) // 2]
                    mx, my = _G.nodes[mid_node]['x'], _G.nodes[mid_node]['y']
                    ax.annotate(label_text, (mx, my), fontsize=8,
                                fontweight='bold', ha='center', va='bottom',
                                xytext=(0, 6), textcoords='offset points',
                                bbox=dict(boxstyle='round,pad=0.2',
                                          facecolor='white', alpha=0.85,
                                          edgecolor='grey'))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                sx, sy = _G.nodes[src]['x'], _G.nodes[src]['y']
                dx, dy = _G.nodes[dst]['x'], _G.nodes[dst]['y']
                ax.plot([sx, dx], [sy, dy], color=color, lw=lw, alpha=alpha,
                        linestyle=linestyle)

        def _get_path_length(src, dst):
            try:
                return nx.shortest_path_length(G, src, dst, weight='length')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return 0

        # --- LEFT PANEL: D2D ---
        ox.plot_graph(G, ax=ax_d2d, node_size=0, edge_linewidth=0.3,
                      show=False, close=False, edge_color='#cccccc',
                      bgcolor='white')

        d2d_cruise = _get_path_length(veh_pos, origin)
        d2d_ride = _get_path_length(origin, destination)

        if veh_pos != origin:
            _plot_segment(ax_d2d, veh_pos, origin, color='grey', lw=2,
                          alpha=0.5, linestyle='dashed',
                          label_text=f'Cruise: {d2d_cruise:.0f}m')
        _plot_segment(ax_d2d, origin, destination, color='#1f77b4', lw=3,
                      alpha=0.8, linestyle='solid',
                      label_text=f'Ride: {d2d_ride:.0f}m')

        ax_d2d.scatter(*coords['veh'], s=220, c='#555555', marker='s',
                       edgecolors='black', linewidths=1, zorder=6)
        ax_d2d.scatter(*coords['origin'], s=200, c='green', marker='D',
                       edgecolors='black', linewidths=0.8, zorder=7)
        ax_d2d.scatter(*coords['destination'], s=200, c='orange', marker='o',
                       edgecolors='black', linewidths=0.8, zorder=7)

        ax_d2d.annotate(f'V{veh_id}', coords['veh'], fontsize=9,
                        fontweight='bold', ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points')
        ax_d2d.annotate('Origin', coords['origin'], fontsize=8,
                        ha='center', va='bottom', xytext=(0, 8),
                        textcoords='offset points', color='darkgreen')
        ax_d2d.annotate('Dest', coords['destination'], fontsize=8,
                        ha='center', va='bottom', xytext=(0, 8),
                        textcoords='offset points', color='darkorange')

        d2d_total = d2d_cruise + d2d_ride
        ax_d2d.set_title(
            f'D2D (Door-to-Door)\n'
            f'Vehicle drives {d2d_total:.0f}m total '
            f'(cruise {d2d_cruise:.0f}m + ride {d2d_ride:.0f}m)\n'
            f'Cost: \u20ac{assign["cost_d2d"]:.4f}',
            fontsize=11, fontweight='bold')
        ax_d2d.set_xlim(xlim)
        ax_d2d.set_ylim(ylim)
        ax_d2d.set_aspect('equal')

        # --- RIGHT PANEL: PUDO ---
        ox.plot_graph(G, ax=ax_pudo, node_size=0, edge_linewidth=0.3,
                      show=False, close=False, edge_color='#cccccc',
                      bgcolor='white')

        pudo_cruise = _get_path_length(veh_pos, pickup)
        pudo_ride = _get_path_length(pickup, dropoff)
        walk_pickup = components.get('walk_to_pickup_m',
                                     _get_path_length(origin, pickup))
        walk_dropoff = components.get('walk_from_dropoff_m',
                                      _get_path_length(dropoff, destination))

        if veh_pos != pickup:
            _plot_segment(ax_pudo, veh_pos, pickup, color='grey', lw=2,
                          alpha=0.5, linestyle='dashed',
                          label_text=f'Cruise: {pudo_cruise:.0f}m')
        _plot_segment(ax_pudo, pickup, dropoff, color='#1f77b4', lw=3,
                      alpha=0.8, linestyle='solid',
                      label_text=f'Ride: {pudo_ride:.0f}m')

        if pickup != origin:
            _plot_segment(ax_pudo, origin, pickup, color='green', lw=2.5,
                          alpha=0.7, linestyle='dotted',
                          label_text=f'Walk: {walk_pickup:.0f}m',
                          graph=G_walk)
        if dropoff != destination:
            _plot_segment(ax_pudo, dropoff, destination, color='darkorange',
                          lw=2.5, alpha=0.7, linestyle='dotted',
                          label_text=f'Walk: {walk_dropoff:.0f}m',
                          graph=G_walk)

        ax_pudo.scatter(*coords['veh'], s=220, c='#555555', marker='s',
                        edgecolors='black', linewidths=1, zorder=6)
        ax_pudo.scatter(*coords['origin'], s=200, c='green', marker='D',
                        edgecolors='black', linewidths=0.8, zorder=7)
        ax_pudo.scatter(*coords['destination'], s=200, c='orange', marker='o',
                        edgecolors='black', linewidths=0.8, zorder=7)
        if pickup != origin:
            ax_pudo.scatter(*coords['pickup'], s=280, c='blue', marker='*',
                            edgecolors='black', linewidths=0.8, zorder=8)
            ax_pudo.annotate('PUDO\nPickup', coords['pickup'], fontsize=8,
                             fontweight='bold', ha='center', va='bottom',
                             xytext=(0, 10), textcoords='offset points',
                             color='blue')
        if dropoff != destination:
            ax_pudo.scatter(*coords['dropoff'], s=280, c='red', marker='*',
                            edgecolors='black', linewidths=0.8, zorder=8)
            ax_pudo.annotate('PUDO\nDropoff', coords['dropoff'], fontsize=8,
                             fontweight='bold', ha='center', va='bottom',
                             xytext=(0, 10), textcoords='offset points',
                             color='red')

        ax_pudo.annotate(f'V{veh_id}', coords['veh'], fontsize=9,
                         fontweight='bold', ha='center', va='bottom',
                         xytext=(0, 8), textcoords='offset points')
        ax_pudo.annotate('Origin', coords['origin'], fontsize=8,
                         ha='center', va='bottom', xytext=(0, 8),
                         textcoords='offset points', color='darkgreen')
        ax_pudo.annotate('Dest', coords['destination'], fontsize=8,
                         ha='center', va='bottom', xytext=(0, 8),
                         textcoords='offset points', color='darkorange')

        pudo_veh_total = pudo_cruise + pudo_ride
        ax_pudo.set_title(
            f'PUDO (Pick-Up / Drop-Off)\n'
            f'Vehicle drives {pudo_veh_total:.0f}m total '
            f'(cruise {pudo_cruise:.0f}m + ride {pudo_ride:.0f}m)\n'
            f'Rider walks {walk_pickup:.0f}m + {walk_dropoff:.0f}m | '
            f'Cost: \u20ac{assign["cost_pudo"]:.4f}',
            fontsize=11, fontweight='bold')
        ax_pudo.set_xlim(xlim)
        ax_pudo.set_ylim(ylim)
        ax_pudo.set_aspect('equal')

        # --- Shared legend ---
        legend_elements = [
            Line2D([0], [0], color='#1f77b4', lw=3,
                   label='Ride (with passenger)'),
            Line2D([0], [0], color='grey', lw=2, linestyle='--',
                   label='Cruise (empty)'),
            Line2D([0], [0], color='green', lw=2.5, linestyle=':',
                   label='Walk to pickup'),
            Line2D([0], [0], color='darkorange', lw=2.5, linestyle=':',
                   label='Walk from dropoff'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='#555555',
                   markersize=12, markeredgecolor='black', label='Vehicle'),
            Line2D([0], [0], marker='D', color='w', markerfacecolor='green',
                   markersize=12, markeredgecolor='black', label='Origin'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                   markersize=12, markeredgecolor='black', label='Destination'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='blue',
                   markersize=16, label='PUDO Pickup'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red',
                   markersize=16, label='PUDO Dropoff'),
        ]
        fig.legend(handles=legend_elements, loc='lower center', ncol=5,
                   fontsize=10, bbox_to_anchor=(0.5, -0.01))

        # --- Suptitle with outcome ---
        outcome_line = ""
        file_tag = ""
        title_color = 'black'
        if outcome_labels and req_id in outcome_labels:
            oc = outcome_labels[req_id]
            outcome_line = f"\n{_OUTCOME_TITLE_LABELS.get(oc, oc)}"
            file_tag = f"_{_OUTCOME_FILE_TAGS.get(oc, oc)}"
            title_color = _OUTCOME_COLORS.get(oc, 'black')

        veh_saved = d2d_total - pudo_veh_total
        batch_type = decision_log.get('meta', {}).get(
            'batch_type', 'unknown').upper()
        fig.suptitle(
            f"Route Comparison ({batch_type}): R{req_id} x V{veh_id}\n"
            f"D2D cost: \u20ac{assign['cost_d2d']:.4f} -> PUDO cost: "
            f"\u20ac{assign['cost_pudo']:.4f} | "
            f"Savings: \u20ac{assign['savings']:.4f} | "
            f"Vehicle saves {veh_saved:.0f}m"
            f"{outcome_line}",
            fontsize=13, fontweight='bold', color=title_color, y=1.02)

        plt.tight_layout(rect=[0, 0.04, 1, 0.95])
        out_path = os.path.join(output_dir,
                                f'route_comparison_R{req_id}{file_tag}.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        generated.append(out_path)
        print(f"  [OK] Route comparison R{req_id}: {out_path}")

    return generated