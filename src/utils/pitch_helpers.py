"""
pitch_helpers.py — Shared pitch utilities used across demo and model modules.
"""

# Defaults for optional pitch fields the user might omit in simulation input
PITCH_DEFAULTS = {
    "release_spin_rate": 2400.0,
    "pfx_x":            -0.5,
    "pfx_z":             1.2,
    "release_pos_x":    -1.5,
    "release_pos_z":     6.0,
    "release_extension":  6.3,
}


def location_tag(plate_x: float, plate_z: float) -> str:
    """Human-readable zone label from catcher's perspective."""
    if plate_z > 3.2:
        vert = "elevated"
    elif plate_z > 2.6:
        vert = "mid"
    elif plate_z > 1.8:
        vert = "low"
    else:
        vert = "below-zone"

    if plate_x > 0.85:
        horiz = "way-away"
    elif plate_x > 0.4:
        horiz = "away"
    elif plate_x > -0.4:
        horiz = "middle"
    elif plate_x > -0.85:
        horiz = "in"
    else:
        horiz = "way-in"

    if horiz == "middle":
        return vert
    return f"{vert}-{horiz}"
