"""Object types the fire task adds to kinder's kinematic3d state (PLAN.md D5, A58).

kinder states are dictionaries from Object to a feature vector, and every object's vector has the
features its Type lists in a shared table (relational_structs.create_state_from_dict). kinder's
table for this backend is Kinematic3DEnvTypeFeatures (robot, cuboid, triangle, point, fixture).
We copy it and add three types: one FireCell per grid cell, one Item per item to rescue (level 2) and
one FireLabStatus for the episode.
"""

from relational_structs import Type

from kinder.envs.kinematic3d.object_types import Kinematic3DEnvTypeFeatures

FireCellType = Type("FireCell")
ItemType = Type("Item")
StatusType = Type("FireLabStatus")
JOBS = (None, "pick", "drop")  # the status feature "job" is an index into this

FIRELAB_TYPE_FEATURES: dict[Type, list[str]] = dict(Kinematic3DEnvTypeFeatures)
# x, y: the cell centre in metres; state: firelab.fire NORMAL / BURNING / BURNT / BLOCKED as a float.
FIRELAB_TYPE_FEATURES[FireCellType] = ["x", "y", "state"]
# tick: env steps since reset; health: robot health (A59); coverage: FireGrid.coverage(); level: 1 or 2.
# job: 0 none, 1 pick, 2 drop (JOBS); job_item: the item being picked, else -1; job_left: ticks until it is done.
FIRELAB_TYPE_FEATURES[StatusType] = ["tick", "health", "coverage", "level", "job", "job_item", "job_left"]
# x, y: centre of the cell the item lay in at reset; value: worth now; start: worth at reset;
# where: firelab.items FLOOR / BAG / SAVED / LOST as a float.
FIRELAB_TYPE_FEATURES[ItemType] = ["x", "y", "value", "start", "where"]


def cell_name(i: int, j: int) -> str:
    """Zero-padded so that sorting the names gives FireGrid's row-major (i, j) order."""
    return f"cell_{i:02d}_{j:02d}"


def item_name(k: int) -> str:
    return f"item_{k:02d}"
