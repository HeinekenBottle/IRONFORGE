"""IRONFORGE Version Information"""

__version__ = "1.0.0"
__version_info__ = (0, 9, 0)
__release_notes__ = "Macro-Micro Analytics Suite: HTF→trade horizon quantification, cross-session influence, session prototypes, adjacent possible exploration"

# Feature version tracking
NODE_FEATURES_VERSION = "1.1"
EDGE_FEATURES_VERSION = "1.0"

# HTF Context Features (f45-f50)
HTF_FEATURES = [
    "f45_sv_m15_z",  # M15 synthetic volume z-score
    "f46_sv_h1_z",  # H1 synthetic volume z-score
    "f47_barpos_m15",  # M15 bar position [0,1]
    "f48_barpos_h1",  # H1 bar position [0,1]
    "f49_dist_daily_mid",  # Distance to daily midpoint (normalized)
    "f50_htf_regime",  # HTF regime classification {0,1,2}
]

TOTAL_NODE_FEATURES = 51  # 45 base + 6 HTF
TOTAL_EDGE_FEATURES = 20  # Unchanged from v1.0
