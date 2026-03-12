"""Agent decision function templates.

This package splits decision functions by domain (driver, rider, matching).
All public names are re-exported here for backward compatibility.
"""

# Shared helpers
from MaaSSim.decisions._helpers import (
    _compute_baseline_fare,
    _downgrade_offer_to_d2d,
    _driver_s1_check,
    _safe_sigmoid,
    dummy_False,
    dummy_True,
    f_dummy_repos,
)

# Driver decisions
from MaaSSim.decisions.driver import (
    f_driver_out,
    f_repos,
    f_pudo_driver_decline,
)

# Rider decisions
from MaaSSim.decisions.rider import (
    f_platform_opt_out,
    f_out,
    f_mode,
    _d2d_ratio_check,
    f_pudo_rider_mode,
    f_platform_choice,
)

# Matching / platform
from MaaSSim.decisions.matching import (
    create_pudo_offer,
    f_match_pudo,
    f_match,
    f_stop_crit,
)

__all__ = [
    # helpers
    '_compute_baseline_fare', '_downgrade_offer_to_d2d', '_driver_s1_check', '_safe_sigmoid',
    'dummy_False', 'dummy_True', 'f_dummy_repos',
    # driver
    'f_driver_out', 'f_repos', 'f_pudo_driver_decline',
    # rider
    'f_platform_opt_out', 'f_out', 'f_mode', '_d2d_ratio_check',
    'f_pudo_rider_mode', 'f_platform_choice',
    # matching
    'create_pudo_offer', 'f_match_pudo',
    'f_match', 'f_stop_crit',
]
