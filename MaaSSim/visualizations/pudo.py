################################################################################
# Module: visualizations/pudo.py
# PUDO-specific route comparison visualization
################################################################################

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import networkx as nx
import osmnx as ox

from .core import _get_node_xy, add_route


# ---------------------------------------------------------------------------
# PUDO outcome label / color / tag lookups
# ---------------------------------------------------------------------------

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
