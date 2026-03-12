################################################################################
# Module: decisions/_helpers.py
# Description: Shared utility functions for agent decision modules
################################################################################
from math import exp
from dotmap import DotMap


def _compute_baseline_fare(platform_obj, dist_m, ride_time_s):
    """Compute rider-facing fare from distance + time components.

    Args:
        platform_obj: Platform row with 'fare' (EUR/km) and optional 'fare_per_min' (EUR/min)
        dist_m: Trip distance in meters
        ride_time_s: Trip ride time in seconds

    Returns:
        fare in EUR
    """
    fare_per_km = platform_obj.fare
    fare_per_min = getattr(platform_obj, 'fare_per_min', 0.0)
    return fare_per_km * dist_m / 1000 + fare_per_min * ride_time_s / 60


def _downgrade_offer_to_d2d(sim, platform, offer):
    """Rewrite a PUDO offer in-place to Door-to-Door terms.

    Sets pickup=origin, dropoff=destination, no walking, full fare.
    Used as fallback when driver or rider rejects PUDO but may accept D2D.
    """
    request = offer['request']
    veh_id = offer['veh_id']
    vehicle = sim.vehicles.loc[veh_id]

    offer['pickup_node'] = request.origin
    offer['dropoff_node'] = request.destination
    offer['walk_to_pickup'] = 0
    offer['walk_from_dropoff'] = 0
    offer['savings'] = 0
    offer['rider_incentive'] = 0
    offer['_d2d_fallback'] = True

    # Thesis behavioral model fields (all zero for D2D)
    offer['delta_dist_m'] = 0
    offer['delta_time_s'] = 0
    offer['delta_pi'] = 0
    offer['d_walk_meters'] = 0

    offer['wait_time'] = sim.skims.ride[vehicle.pos][request.origin]
    offer['travel_time'] = sim.skims.ride[request.origin][request.destination]

    d2d_ride_time = sim.skims.ride[request.origin][request.destination]
    offer['fare'] = _compute_baseline_fare(platform.platform, request.dist, d2d_ride_time)
    return offer


def _driver_s1_check(wait_time, fare, ride_speed):
    """Stage 1 driver feasibility check: dispatch distance and minimum fare.

    Rejects if dispatch distance exceeds 2000m (scaled by ride speed)
    or if fare is below EUR 0.10.

    Args:
        wait_time: Dispatch time in seconds (vehicle to pickup)
        fare: Offer fare in EUR
        ride_speed: Current ride speed in m/s

    Returns:
        True if driver should REJECT (S1 fail), False if S1 passes.
    """
    WAIT_DIST_LIMIT_M = 2000.0
    MIN_FARE_EUR = 0.10
    wait_limit_s = WAIT_DIST_LIMIT_M / ride_speed
    return (wait_time >= wait_limit_s) or (fare < MIN_FARE_EUR)


def _safe_sigmoid(x):
    """sigmoid function."""
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        ex = exp(x)
        return ex / (1.0 + ex)


#################
#    DUMMIES    #
#################


def dummy_False(*args, **kwargs):
    # dummy function to always return False,
    # used as default function inside of functionality
    # (if the behaviour is not modelled)
    return False


def dummy_True(*args, **kwargs):
    # dummy function to always return True
    return True


def f_dummy_repos(*args, **kwargs):
    # handles the vehiciles when they become IDLE (after comppleting the request or entering the system)
    repos = DotMap()
    repos.flag = False
    # repos.pos = None
    # repos.time = 0
    return repos
