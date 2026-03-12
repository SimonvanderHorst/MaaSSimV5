################################################################################
# Module: pudo_logger.py
# Description: Structured decision logging for PUDO matching pipeline
# Records every decision step for auditability and visualization
################################################################################

import json
import time


class PudoDecisionLog:
    """Accumulates per-batch decision data for one matching invocation.

    Usage:
        log = PudoDecisionLog(batch_time=120.0, batch_type='milp', vehQ=[1,2,3], reqQ=[10,11])
        log.log_feasibility(req_id=10, side='origin', ...)
        log.log_best_edge(veh_id=1, req_id=10, ...)
        ...
        data = log.to_dict()
    """

    def __init__(self, batch_time, batch_type, vehQ, reqQ, log_level='summary'):
        """
        Args:
            batch_time: Simulation time of this batch
            batch_type: 'milp'
            vehQ: List of vehicle IDs in queue
            reqQ: List of request IDs in queue
            log_level: 'summary' or 'full' (full includes per-combo cost details)
        """
        self.log_level = log_level
        self.meta = {
            't': float(batch_time),
            'batch_type': batch_type,
            'log_level': log_level,
        }
        self.queue_snapshot = {
            'veh_ids': [int(v) for v in vehQ],
            'req_ids': [int(r) for r in reqQ],
            'n_vehicles': len(vehQ),
            'n_requests': len(reqQ),
        }
        self.feasibility = {}        # req_id -> {'origin': record, 'destination': record}
        self.cost_details = []       # list of CostRecord dicts (only in 'full' mode)
        self.best_per_edge = []      # list of BestEdgeRecord dicts
        self.milp_record = None      # MilpRecord dict
        self.offers = []             # list of OfferRecord dicts
        self.assignments = []        # final match list

    def log_feasibility(self, req_id, side, anchor_node, feasible_nodes,
                        walk_distances, walk_times, max_walk_distance):
        """Log feasible PUDO nodes for one side (origin or destination) of a request.

        Args:
            req_id: Request identifier
            side: 'origin' or 'destination'
            anchor_node: The request's origin or destination node
            feasible_nodes: List of node IDs within walking distance
            walk_distances: Dict {node_id: distance_meters}
            walk_times: Dict {node_id: time_seconds}
            max_walk_distance: Config parameter (meters)
        """
        req_key = int(req_id)
        if req_key not in self.feasibility:
            self.feasibility[req_key] = {}

        self.feasibility[req_key][side] = {
            'anchor_node': int(anchor_node),
            'feasible_nodes': [int(n) for n in feasible_nodes],
            'walk_distances_m': {int(k): float(v) for k, v in walk_distances.items()},
            'walk_times_s': {int(k): float(v) for k, v in walk_times.items()},
            'max_walk_distance': float(max_walk_distance),
            'n_feasible': len(feasible_nodes),
        }

    def log_cost_detail(self, veh_id, req_id, pickup_node, dropoff_node,
                        dist_veh_to_origin, dist_origin_to_dest, cost_d2d,
                        dist_veh_to_pickup, dist_pickup_to_dropoff,
                        cost_pudo_vehicle, walk_to_pickup_m, walk_from_dropoff_m,
                        cost_pudo_walking,
                        cost_pudo_total, savings, **kwargs):
        """Log cost components for one (v, q, p, d) combination. Only in 'full' mode."""
        if self.log_level != 'full':
            return

        entry = {
            'veh_id': int(veh_id),
            'req_id': int(req_id),
            'pickup_node': int(pickup_node),
            'dropoff_node': int(dropoff_node),
            'dist_veh_to_origin': float(dist_veh_to_origin),
            'dist_origin_to_dest': float(dist_origin_to_dest),
            'cost_d2d': float(cost_d2d),
            'dist_veh_to_pickup': float(dist_veh_to_pickup),
            'dist_pickup_to_dropoff': float(dist_pickup_to_dropoff),
            'cost_pudo_vehicle': float(cost_pudo_vehicle),
            'walk_to_pickup_m': float(walk_to_pickup_m),
            'walk_from_dropoff_m': float(walk_from_dropoff_m),
            'cost_pudo_walking': float(cost_pudo_walking),
            'cost_pudo_total': float(cost_pudo_total),
            'savings': float(savings),
        }
        # Rider-aware fields (optional)
        if 'rider_side_cost' in kwargs:
            entry['rider_side_cost'] = float(kwargs['rider_side_cost'])
            entry['ranking_score'] = float(kwargs['ranking_score'])
        self.cost_details.append(entry)

    def log_best_edge(self, veh_id, req_id, best_pickup, best_dropoff,
                      cost_d2d, cost_pudo, savings, n_combos_evaluated,
                      cost_components=None):
        """Log the best PUDO pair selected for one (vehicle, request) edge.

        Args:
            cost_components: Optional dict with cost breakdown:
                {cost_vehicle_driving, cost_walking,
                 walk_to_pickup_m, walk_from_dropoff_m,
                 dist_veh_to_pickup_m, dist_pickup_to_dropoff_m}
        """
        entry = {
            'veh_id': int(veh_id),
            'req_id': int(req_id),
            'best_pickup': int(best_pickup),
            'best_dropoff': int(best_dropoff),
            'cost_d2d': float(cost_d2d),
            'cost_pudo': float(cost_pudo),
            'savings': float(savings),
            'n_combos_evaluated': int(n_combos_evaluated),
            'is_d2d_fallback': bool(savings == 0),
        }
        if cost_components is not None:
            entry['components'] = {k: float(v) for k, v in cost_components.items()}
        self.best_per_edge.append(entry)

    def log_milp(self, n_variables, n_vehicle_constraints, n_request_constraints,
                 big_M, objective_coefficients, solver_status, solver_time_s,
                 selected_pairs, unselected_pairs):
        """Log MILP solver details.

        Args:
            n_variables: Number of binary decision variables
            n_vehicle_constraints: Number of vehicle capacity constraints
            n_request_constraints: Number of request fulfillment constraints
            big_M: Big-M value used in objective
            objective_coefficients: List of {veh_id, req_id, cost_pudo, coeff}
            solver_status: String status from PuLP
            solver_time_s: Wall-clock solve time
            selected_pairs: List of {veh_id, req_id, x_value} for x=1
            unselected_pairs: List of {veh_id, req_id, cost_pudo, coeff} for x=0
        """
        self.milp_record = {
            'n_variables': int(n_variables),
            'n_vehicle_constraints': int(n_vehicle_constraints),
            'n_request_constraints': int(n_request_constraints),
            'big_M': float(big_M),
            'objective_coefficients': objective_coefficients,
            'solver_status': solver_status,
            'solver_time_s': float(solver_time_s),
            'selected_pairs': selected_pairs,
            'unselected_pairs': unselected_pairs,
        }

    def log_offer(self, veh_id, req_id, pickup_node, dropoff_node,
                  wait_time, travel_time, baseline_fare, rider_discount,
                  final_fare, walk_to_pickup_m, walk_from_dropoff_m, outcome,
                  d2d_fallback=False, **behavioral_kwargs):
        """Log offer creation and its outcome.

        Args:
            d2d_fallback: Whether this offer was downgraded from PUDO to D2D
            behavioral_kwargs: Optional driver behavioral fields:
                driver_utility, driver_alpha, driver_accepted
        """
        entry = {
            'veh_id': int(veh_id),
            'req_id': int(req_id),
            'pickup_node': int(pickup_node),
            'dropoff_node': int(dropoff_node),
            'wait_time_s': float(wait_time),
            'travel_time_s': float(travel_time),
            'baseline_fare': float(baseline_fare),
            'rider_discount': float(rider_discount),
            'final_fare': float(final_fare),
            'walk_to_pickup_m': float(walk_to_pickup_m),
            'walk_from_dropoff_m': float(walk_from_dropoff_m),
            'outcome': outcome,
            'd2d_fallback': bool(d2d_fallback),
        }
        if behavioral_kwargs:
            entry['behavioral'] = {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in behavioral_kwargs.items()}
        self.offers.append(entry)

    def log_behavioral_decision(self, agent_type, agent_id, req_id,
                                utility, alpha, accepted, components):
        """Log a behavioral acceptance/rejection decision.

        Args:
            agent_type: 'driver' or 'rider'
            agent_id: Vehicle ID or passenger ID
            req_id: Request ID
            utility: Computed utility value
            alpha: Sigmoid acceptance probability
            accepted: Boolean result
            components: Dict of utility components for auditability
        """
        if not hasattr(self, 'behavioral_decisions'):
            self.behavioral_decisions = []

        self.behavioral_decisions.append({
            'agent_type': agent_type,
            'agent_id': int(agent_id),
            'req_id': int(req_id),
            'utility': float(utility),
            'acceptance_prob': float(alpha),
            'accepted': bool(accepted),
            'components': {k: float(v) if isinstance(v, (int, float)) else v
                           for k, v in components.items()},
        })

    def to_dict(self):
        """Serialize the full log to a JSON-compatible dict."""
        d = {
            'meta': self.meta,
            'queue_snapshot': self.queue_snapshot,
            'feasibility': {str(k): v for k, v in self.feasibility.items()},
            'best_per_edge': self.best_per_edge,
            'offers': self.offers,
            'assignments': self.assignments,
        }
        if self.cost_details:
            d['cost_details'] = self.cost_details
        if self.milp_record is not None:
            d['milp_record'] = self.milp_record
        if hasattr(self, 'behavioral_decisions') and self.behavioral_decisions:
            d['behavioral_decisions'] = self.behavioral_decisions
        return d

    def to_json(self, path):
        """Write the full log to a JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
