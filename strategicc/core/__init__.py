from .transitions import build_transition_index, TransitionRecord
from .adjacency   import compute_neighbor_fractions
from .spatial     import load_spatial_multipliers
from .multipliers import sample_transition_multipliers, describe_multiplier_rules
from .age         import (
    build_initial_age_from_raster,
    build_initial_age_from_rules,
    update_age,
    age_gate_mask,
    save_age_tif,
)
from .patches import (
    sample_patch_size_ha,
    grow_patch,
    grow_patches_for_group,
)

__all__ = [
    "build_transition_index",
    "TransitionRecord",
    "compute_neighbor_fractions",
    "load_spatial_multipliers",
    "sample_transition_multipliers",
    "describe_multiplier_rules",
    "build_initial_age_from_raster",
    "build_initial_age_from_rules",
    "update_age",
    "age_gate_mask",
    "save_age_tif",
    "sample_patch_size_ha",
    "grow_patch",
    "grow_patches_for_group",
]
