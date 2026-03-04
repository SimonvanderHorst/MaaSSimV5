################################################################################
# Module: performance.py
# Description: Processes raw simulation results into dataframes with network-wide and sinlge pax/veh KPIs
# Rafal Kucharski @ TU Delft, The Netherlands
################################################################################



from .traveller import travellerEvent
from .driver import driverEvent
import pandas as pd

# matplotlib is optional - only used for commented-out plotting code
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None



def kpi_pax(*args,**kwargs):
    # calculate passenger indicators (global and individual)

    sim = kwargs.get('sim', None)
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    paxindex = sim.inData.passengers.index
    df = simrun['trips'].copy()  # results of previous simulation
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.pax == df.pax_s]  # filter for the same vehicles only
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['pax', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event

    ret.columns.name = None
    ret = ret.reindex(paxindex)  # update for vehicles with no record

    ret.index.name = 'pax'
    ret = ret.fillna(0)

    for status in travellerEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuses

    # meaningful names
    ret['TRAVEL'] = ret['ARRIVES_AT_DROPOFF']  # time with traveller (paid time)
    ret['WAIT'] = ret['RECEIVES_OFFER'] + ret[
        'MEETS_DRIVER_AT_PICKUP']  # time waiting for traveller (by default zero)
    ret['OPERATIONS'] = ret['ACCEPTS_OFFER'] + ret['DEPARTS_FROM_PICKUP'] + ret['SETS_OFF_FOR_DEST']

    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nP'] = ret.shape[0]
    return {'pax_exp': ret, 'pax_kpi': kpi}


def kpi_veh(*args, **kwargs):
    """
    calculate vehicle KPIs (global and individual)
    it bases of duration of each event.
    The time per each event denotes the time spent by vehicle BEFORE that event took place.
    From this we can interpret duration of each segments.
    :param args:
    :param kwargs:
    :return: dictionary with kpi per vehicle and system-wide
    """
    sim =  kwargs.get('sim', None)
    run_id = kwargs.get('run_id', None)
    simrun = sim.runs[run_id]
    vehindex = sim.inData.vehicles.index
    df = simrun['rides'].copy()  # results of previous simulation
    DECIDES_NOT_TO_DRIVE = df[df.event == driverEvent.DECIDES_NOT_TO_DRIVE.name].veh  # track drivers out
    dfs = df.shift(-1)  # to map time periods between events
    dfs.columns = [_ + "_s" for _ in df.columns]  # columns with _s are shifted
    df = pd.concat([df, dfs], axis=1)  # now we have time periods
    df = df[df.veh == df.veh_s]  # filter for the same vehicles only
    df = df[~(df.t == df.t_s)]  # filter for positive time periods only
    df['dt'] = df.t_s - df.t  # make time intervals
    ret = df.groupby(['veh', 'event_s'])['dt'].sum().unstack()  # aggreagted by vehicle and event
    ret.columns.name = None
    ret = ret.reindex(vehindex)  # update for vehicles with no record
    ret['nRIDES'] = df[df.event == driverEvent.ARRIVES_AT_DROPOFF.name].groupby(
        ['veh']).size().reindex(ret.index)
    ret['nREJECTED'] = df[df.event == driverEvent.IS_REJECTED_BY_TRAVELLER.name].groupby(
        ['veh']).size().reindex(ret.index)
    for status in driverEvent:
        if status.name not in ret.columns:
            ret[status.name] = 0  # cover all statuss

    DECIDES_NOT_TO_DRIVE.index = DECIDES_NOT_TO_DRIVE.values
    ret['OUT'] = DECIDES_NOT_TO_DRIVE
    ret['OUT'] = ~ret['OUT'].isnull()
    ret = ret[['nRIDES', 'nREJECTED', 'OUT'] + [_.name for _ in driverEvent]].fillna(0)  # nans become 0

    # meaningful names
    ret['TRAVEL'] = ret['ARRIVES_AT_DROPOFF']  # time with traveller (paid time)
    ret['WAIT'] = ret['MEETS_TRAVELLER_AT_PICKUP']  # time waiting for traveller (by default zero)
    ret['CRUISE'] = ret['ARRIVES_AT_PICKUP'] + ret['REPOSITIONED']  # time to arrive for traveller
    ret['OPERATIONS'] = ret['ACCEPTS_REQUEST'] + ret['DEPARTS_FROM_PICKUP'] + ret['IS_ACCEPTED_BY_TRAVELLER']
    ret['IDLE'] = ret['ENDS_SHIFT'] - ret['OPENS_APP'] - ret['OPERATIONS'] - ret['CRUISE'] - ret['WAIT'] - ret['TRAVEL']

    def _pax_dist_m(req):
        """Actual driving distance in metres for a request: PUDO pickup→dropoff, D2D origin→dest."""
        try:
            pickup = req.get('pudo_pickup_node', None)
            if pd.notna(pickup):
                return float(sim.skims.dist.at[int(req.pudo_dropoff_node), int(pickup)])
        except (KeyError, ValueError, TypeError):
            pass
        return float(req.dist)

    # Precompute per-request distances once (avoids nested apply fragility)
    req_dist_km = sim.inData.requests.apply(_pax_dist_m, axis=1) / 1000.0  # Series[pax_id -> km]

    def _veh_pax_km(veh_row):
        raw_ids = simrun.trips[simrun.trips.veh_id == veh_row.name].pax.dropna().unique()
        pax_ids = [int(p) for p in raw_ids if int(p) in req_dist_km.index]
        return req_dist_km.loc[pax_ids].sum() if pax_ids else 0.0

    ret['PAX_KM'] = ret.apply(_veh_pax_km, axis=1)
#     ret.apply(lambda x: print(sim.inData.platforms.loc[sim.inData.vehicles.loc[x.name].platform]))
#     print(sim.inData.platforms.loc[sim.inData.vehicles.loc['name'].platform])
    
    '''
    rides = sim.inData.sblts.rides
    profits_idx = []
    for i in range(1, len(ret.index)+1):
        profits_idx.append((max(pd.DataFrame(sim.vehs[i].myrides)['paxes'].to_list())))
    print(profits_idx)
    profits = []
    for i in profits_idx:
        row = sim.inData.sblts.rides['indexes_orig'].apply(lambda x: x == i)

        profits.append(sim.inData.sblts.rides[row.values]['driver_revenue'].to_list()[0] if sim.inData.sblts.rides[row.values]['driver_revenue'].to_list() else 0)

    
    
    '''
#     ret['REVENUE'] = ret.apply(lambda x: sim.inData.platforms.loc[sim.inData.vehicles.loc[
#         x.name].platform].fare, axis=1)
    ret['REVENUE'] = ret.apply(
    lambda x: sim.inData.platforms.loc[sim.inData.vehicles.loc[x.name].platform].fare
              * (x['nRIDES'] if pd.notnull(x['nRIDES']) else 0),
    axis=1
)
    
    ret['nREJECTS'] = df[df.event==driverEvent.REJECTS_REQUEST.name].groupby(['veh']).size().reindex(ret.index)
    ret.index.name = 'veh'
    total_rev = ret['REVENUE'].sum()
# This is a code for plotting
#plotting seaborn
    # plot graph of driver revenue
   # vehicles  = list(sim.vehs.keys())
   # fig, ax = plt.subplots(figsize = (10,5))
    #bars = ax.barh(vehicles, profits)
   # ax.bar_label(bars)
   # for bars in ax.containers:
   #     ax.bar_label(bars)
    

    #plt.xlabel("Revenue")
   # plt.ylabel("Vehicles")
   # plt.title("revenue against driver")
   # plt.show()
    # KPIs
    kpi = ret.agg(['sum', 'mean', 'std'])
    kpi['nV'] = ret.shape[0]
    return {'veh_exp': ret, 'veh_kpi': kpi, 'all_kpi': pd.Series({'total_revenue': total_rev})}


def calculate_pudo_metrics(sim, run_id=0):
    """
    Calculate PUDO-specific performance metrics.
    Compares D2D baseline vs. realized PUDO performance.

    Args:
        sim: Simulator object
        run_id: Run identifier (default 0)

    Returns:
        Dictionary with PUDO-specific KPIs:
        - total_savings_offered: Total operational savings from all PUDO matches
        - pudo_acceptance_rate: Fraction of PUDO offers that were accepted
        - avg_walk_to_pickup: Average walking distance to pickup (meters)
        - avg_walk_from_dropoff: Average walking distance from dropoff (meters)
        - total_walk_distance: Total walking distance by all passengers (meters)
        - vkt_reduction: Reduction in Vehicle Kilometers Traveled vs D2D baseline
        - num_pudo_trips: Number of trips that used PUDO
        - num_total_trips: Total number of trips
    """
    results = {}

    # Filter requests that had PUDO offers
    pudo_requests = sim.inData.requests[
        sim.inData.requests.pudo_pickup_node.notna()
    ]

    if len(pudo_requests) == 0:
        # No PUDO trips - return empty metrics
        results['total_savings_offered'] = 0
        results['pudo_acceptance_rate'] = 0
        results['avg_walk_to_pickup'] = 0
        results['avg_walk_from_dropoff'] = 0
        results['total_walk_distance'] = 0
        results['vkt_reduction'] = 0
        results['num_pudo_trips'] = 0
        results['num_total_trips'] = len(sim.inData.requests)
        return results

    # Total savings pool
    results['total_savings_offered'] = pudo_requests.pudo_savings.sum()

    # Acceptance rate (trips with positive savings were accepted)
    accepted = pudo_requests[pudo_requests.pudo_savings > 0]
    results['pudo_acceptance_rate'] = len(accepted) / len(pudo_requests) if len(pudo_requests) > 0 else 0

    # Walking statistics
    results['avg_walk_to_pickup'] = accepted.walk_to_pickup_dist.mean() if len(accepted) > 0 else 0
    results['avg_walk_from_dropoff'] = accepted.walk_from_dropoff_dist.mean() if len(accepted) > 0 else 0
    results['total_walk_distance'] = (
        accepted.walk_to_pickup_dist.sum() +
        accepted.walk_from_dropoff_dist.sum()
    ) if len(accepted) > 0 else 0

    # VKT reduction calculation
    # Compare actual VKT (from vehicle KPI) with what D2D would have been
    # This is a simplified calculation - can be refined based on actual trip data
    actual_vkt = sim.res[run_id]['veh_exp']['PAX_KM'].sum() if 'veh_exp' in sim.res[run_id] else 0

    # Estimate D2D VKT by adding back the savings
    # VKT reduction ≈ total operational savings / operating_cost_per_km
    operating_cost_per_km = sim.params.pudo.get('operating_cost_per_km', 0.3)
    if operating_cost_per_km > 0:
        vkt_saved = results['total_savings_offered'] / operating_cost_per_km
        results['vkt_reduction'] = vkt_saved
    else:
        results['vkt_reduction'] = 0

    results['num_pudo_trips'] = len(accepted)
    results['num_total_trips'] = len(sim.inData.requests)

    return results


def calculate_vkt(sim, run_id=0):
    """
    Helper function to calculate total Vehicle Kilometers Traveled.
    Includes both passenger-carrying distance (PAX_KM) and empty repositioning (CRUISE).

    Args:
        sim: Simulator object
        run_id: Run identifier (default 0)

    Returns:
        Total VKT in kilometers (PAX_KM + CRUISE distance)
    """
    if run_id in sim.res and 'veh_exp' in sim.res[run_id]:
        veh_exp = sim.res[run_id]['veh_exp']

        # Passenger-carrying distance (already in km)
        pax_km = veh_exp['PAX_KM'].sum()

        # Empty repositioning distance (convert from time to distance)
        # CRUISE is in seconds, convert to km using ride speed
        ride_speed = getattr(sim.params.speeds, 'ride', 10)  # m/s, default 10 m/s
        cruise_km = veh_exp['CRUISE'].sum() * ride_speed / 1000  # seconds * m/s / 1000 = km

        import numpy as _np
        total_vkt = float(_np.asarray(pax_km + cruise_km).sum())
        return total_vkt
    return 0.0







