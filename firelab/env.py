"""Level 1 of the fire task: kinder's PrplLab3D room with a FireGrid under the floor.

Two classes mirror kinder's own pattern (prpl3d.py): ObjectCentricFireLab3DEnv is the inner env
with object-centric states, FireLab3DEnv is the gymnasium wrapper that fixes the object list and
hands out flat vectors. One env step is one tick (D2); the parent moves the robot, then the fire
spreads, then health is charged if the base stands in a burning cell (D6, A59). Level 1 has no goal:
the episode ends when health reaches 0 (terminated) or after max_ticks (truncated), and the score
is the health fraction left.

Level 2 so far is the room only (chunk 3e, A71): one partition with one 0.8 m doorway, the base stopped
by partition, furniture and the edge of the floor, a safe strip, and the rule check's ignition, so
that room `seed` here is room `seed` in firelab/gate.py. Chunk 3f adds the items (firelab/items.py):
a gripper "close" starts a pick, "open" starts a drop, the episode also ends when half the floor burns,
and the score is the value banked in the safe strip over the value on the floor at reset.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Type as TypingType

import numpy as np
import pybullet as p
from pybullet_helpers.geometry import Pose, set_pose
from pybullet_helpers.utils import create_pybullet_block
from relational_structs import Object, ObjectCentricState

from kinder.envs.kinematic3d.prpl3d import (
    ObjectCentricPrplLab3DEnv,
    PrplLab3DEnv,
    PrplLab3DEnvConfig,
    PrplLab3DObjectCentricState,
)
from firelab.fire import BLOCKED, BURNING, FireGrid, FireGridConfig
from firelab.items import FLOOR, SAVED, ItemRules, Items
from firelab.types import FIRELAB_TYPE_FEATURES, JOBS, FireCellType, ItemType, StatusType, cell_name, item_name


@dataclass(frozen=True)
class FireLabConfig:
    """Task constants (A57, A59). Not a kinder config subclass: those are final (core.FinalConfigMeta)."""

    level: int = 1
    tick_seconds: float = 0.1  # one env step; the real TidyBot's control period (A57)
    max_ticks: int = 500  # 50 s, HAZARD's 1500 frames at 30 fps
    max_base_step: float = 0.05  # metres per tick (0.5 m/s); kinder clips joint deltas by the same number
    health: float = 100.0
    health_loss_per_tick: float = 0.15  # 1.5 HP per second in fire
    block_max_z: float = 0.3  # a lab link whose box bottom is below this stands on the floor and blocks cells
    grid: FireGridConfig = FireGridConfig()
    partitions: tuple = ()  # boxes (x0, x1, y0, y1) in metres; each is a body the base hits and cells that never burn
    partition_height: float = 0.3  # under the folded arm (0.39 m and up), so only the base can touch it (A71)
    safe_y: float = 0.7  # the safe strip is y >= safe_y: grid rows 11 to 14, the rule check's SAFE_ROW (A71)
    items: ItemRules | None = None  # None: no items (level 1)
    end_coverage: float | None = None  # the episode ends when this share of the free floor burns
    items_block_base: bool = False  # False: the base passes over items on the floor, as in the rule check (A71 g)


# One partition across the room at y = -0.4 with a doorway from x = -0.3 to 0.5 (gate.LAYOUTS["one_wide"]).
ONE_DOORWAY = ((-1.5, -0.3, -0.5, -0.3), (0.5, 1.7, -0.5, -0.3))
LEVEL2 = FireLabConfig(
    level=2, partitions=ONE_DOORWAY, health_loss_per_tick=2.0, items=ItemRules(), end_coverage=0.5
)


def level_config(level: int = 1, num_items: int | None = None, values: tuple | None = None) -> FireLabConfig:
    """The settings of a level; at level 2 the number of items and their values can be changed."""
    if level != 2:
        return FireLabConfig(level=level)
    rules = ItemRules(count=num_items or ItemRules.count, values=tuple(values or ItemRules.values))
    return dataclasses.replace(LEVEL2, items=rules)


class FireLabObjectCentricState(PrplLab3DObjectCentricState):
    """kinder's lab state plus the fire task's look-ups."""

    @property
    def status(self) -> Object:
        return self.get_object_from_name("status")

    @property
    def health(self) -> float:
        return float(self.get(self.status, "health"))

    @property
    def tick(self) -> int:
        return int(round(self.get(self.status, "tick")))

    @property
    def coverage(self) -> float:
        return float(self.get(self.status, "coverage"))

    def burning_cells(self) -> list[tuple[float, float]]:
        """Centres (x, y) of the cells on fire, read from the state alone."""
        return [
            (float(self.get(o, "x")), float(self.get(o, "y")))
            for o in self.get_objects(FireCellType)
            if int(self.get(o, "state")) == BURNING
        ]


class ObjectCentricFireLab3DEnv(ObjectCentricPrplLab3DEnv):
    """The lab room, a FireGrid, a health counter and a tick counter."""

    def __init__(
        self,
        num_cubes: int = 0,
        firelab_config: FireLabConfig = FireLabConfig(),
        config: PrplLab3DEnvConfig = PrplLab3DEnvConfig(),
        **kwargs,
    ) -> None:
        step = firelab_config.max_base_step
        config = dataclasses.replace(config, max_action_mag=step, check_base_collisions=True)  # base_env.py:810
        super().__init__(num_cubes=num_cubes, config=config, **kwargs)
        self.fl = firelab_config
        self.fire = FireGrid(firelab_config.grid)
        self.blocked_links = self._block_furniture()
        self._partition_ids = self._build_partitions()
        self._seed: int | None = None
        self.room: int | None = None  # level 2: the number of the room this episode is in
        nx, ny = self.fire.config.shape
        # Same order as FireGrid.features(): i outer, j inner.
        self._cells = [Object(cell_name(i, j), FireCellType) for i in range(nx) for j in range(ny)]
        self._status = Object("status", StatusType)
        self._health = firelab_config.health
        self._tick = 0
        rules = firelab_config.items
        self.items = Items(rules) if rules else None
        n = rules.count if rules else 0
        self._item_objs = [Object(item_name(k), ItemType) for k in range(n)]
        blue = (0.1, 0.3, 0.9, 1.0)  # one 10 cm box per item, to see it and (if switched on) to bump into it
        self._item_ids = [create_pybullet_block(blue, (0.05, 0.05, 0.05), self.physics_client_id) for _ in range(n)]

    def _show_items(self) -> None:
        """Put each item's box on its cell, or under the floor once it has been picked."""
        for k, body in enumerate(self._item_ids):
            x, y = self.fire.center_of(*self.items.cells[k])
            z = 0.05 if self.items.where[k] == FLOOR else -1.0
            set_pose(body, Pose((x, y, z)), self.physics_client_id)

    def _block_furniture(self) -> list[str]:
        """Block the cells under every lab link that stands on the floor; return the link names."""
        cid, lab = self.physics_client_id, self._lab_id
        c = self.fire.config
        room_area = (c.x_max - c.x_min) * (c.y_max - c.y_min)
        names = []
        for link in range(-1, p.getNumJoints(lab, physicsClientId=cid)):
            (x0, y0, z0), (x1, y1, _) = p.getAABB(lab, link, physicsClientId=cid)
            if z0 > self.fl.block_max_z or (x1 - x0) * (y1 - y0) > 0.5 * room_area:
                continue  # hangs on the wall, or a room-sized shell (a guard; this URDF has none)
            if self.fire.block_box(x0, x1, y0, y1) > 0:
                name = "base" if link == -1 else p.getJointInfo(lab, link, physicsClientId=cid)[12].decode()
                names.append(name)
        return names

    def _build_partitions(self) -> list[int]:
        """One grey box per partition piece, standing on the floor; block the cells under it."""
        ids, half_z = [], self.fl.partition_height / 2
        for x0, x1, y0, y1 in self.fl.partitions:
            half = ((x1 - x0) / 2, (y1 - y0) / 2, half_z)
            body = create_pybullet_block((0.45, 0.45, 0.5, 1.0), half, self.physics_client_id)
            set_pose(body, Pose(((x0 + x1) / 2, (y0 + y1) / 2, half_z)), self.physics_client_id)
            self.fire.block_box(x0, x1, y0, y1)
            ids.append(body)
        return ids

    def in_safe_strip(self, x: float, y: float) -> bool:  # pylint: disable=unused-argument
        return y >= self.fl.safe_y

    def ignitable(self) -> list[tuple[int, int]]:
        """Cells where the fire may start: free floor, outside the safe strip, not under the robot's home."""
        home = self.config.robot_base_home_pose
        cells = [(int(i), int(j)) for i, j in np.argwhere(self.fire.state != BLOCKED)]
        return [
            c for c in cells
            if c != self.fire.cell_of(home.x, home.y) and not self.in_safe_strip(*self.fire.center_of(*c))
        ]

    # kinder hooks

    def _get_collision_object_ids(self) -> set[int]:
        solid = self._partition_ids + (self._item_ids if self.fl.items_block_base else [])
        return super()._get_collision_object_ids() | set(solid)

    def _robot_or_held_object_collision_exists(self) -> bool:
        """kinder reverts a move when this is true. Added: the fence, base centre off the floor grid (A71)."""
        pose = self.robot.base.get_pose()
        off_floor = self.fire.cell_of(pose.x, pose.y) is None
        return off_floor or super()._robot_or_held_object_collision_exists()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._seed = seed  # level 2 names its room by the seed
        return super().reset(seed=seed, options=options)

    @property
    def state_cls(self) -> TypingType[FireLabObjectCentricState]:
        return FireLabObjectCentricState

    @property
    def type_features(self) -> dict:
        return FIRELAB_TYPE_FEATURES

    def goal_reached(self) -> bool:
        return False  # level 1 has no goal state; the episode ends by health or the tick cap

    def _reset_objects(self) -> None:
        super()._reset_objects()
        if self.fl.level == 1:
            self.fire.rng = self.np_random  # gymnasium's seeded generator, so a seed fixes the ignition
            self.fire.reset()
        else:  # room `seed` of the rule check: the same generator and the same choice (gate.fire_history)
            self.room = self._seed if self._seed is not None else int(self.np_random.integers(2**31))
            self.fire.rng = np.random.default_rng(self.room)
            floor = self.ignitable()
            lit = self.fire.reset(floor[int(self.fire.rng.integers(len(floor)))])
            if self.items is not None:
                self.items.place([c for c in floor if c != lit], self.room)  # gate.make_world's floor
                self._show_items()
        self._health = self.fl.health
        self._tick = 0

    def _get_obs(self) -> FireLabObjectCentricState:
        data = dict(super()._get_obs().data)
        for obj, row in zip(self._cells, self.fire.features()):
            data[obj] = np.array(row)
        it = self.items
        kind, job_item = it.job if it and it.job else (None, -1)
        left = it.left if it else 0
        data[self._status] = np.array(
            [self._tick, self._health, self.fire.coverage(), self.fl.level, JOBS.index(kind), job_item, left],
            dtype=float,
        )
        for k, obj in enumerate(self._item_objs):
            x, y = self.fire.center_of(*it.cells[k])
            data[obj] = np.array([x, y, it.value[k], it.start[k], it.where[k]], dtype=float)
        return FireLabObjectCentricState(data, FIRELAB_TYPE_FEATURES)

    def _set_object_states(self, obs: PrplLab3DObjectCentricState) -> None:
        super()._set_object_states(obs)
        assert isinstance(obs, FireLabObjectCentricState)
        states = np.array([obs.get(o, "state") for o in self._cells])
        self.fire.state[:] = states.reshape(self.fire.config.shape).astype(np.int8)
        self._health, self._tick = obs.health, obs.tick
        self.fire.tick = self._tick
        it = self.items
        if it is None:
            return
        for k, obj in enumerate(self._item_objs):
            x, y, it.value[k], it.start[k], where = (float(v) for v in obs.data[obj])
            it.cells[k], it.where[k] = self.fire.cell_of(x, y), int(round(where))
        kind = JOBS[int(round(obs.get(self._status, "job")))]
        it.job = (kind, int(round(obs.get(self._status, "job_item")))) if kind else None
        it.left = int(round(obs.get(self._status, "job_left")))
        self._show_items()

    def step(self, action):
        it, pose = self.items, self.robot.base.get_pose()
        if it is not None:
            if action[-1] < -0.5:  # gripper "close": start a pick (refused if busy or nothing is in reach)
                it.start_pick(self.fire.cell_of(pose.x, pose.y))
            elif action[-1] > 0.5:  # gripper "open": start a drop (refused outside the safe strip)
                it.start_drop(self.in_safe_strip(pose.x, pose.y))
            if it.job:
                action = np.zeros_like(action)  # a pick or drop is under way: the robot stands still
        burning = set(self.fire.burning_cells())  # on fire when this tick began: these drain the items
        super().step(action)  # moves the robot (or reverts on collision); its obs and reward are discarded
        ignited = self.fire.step()
        pose = self.robot.base.get_pose()
        in_fire = self.fire.is_burning(pose.x, pose.y)
        loss = self.fl.health_loss_per_tick if in_fire else 0.0
        self._health = max(0.0, self._health - loss)
        self._tick += 1
        info = {
            "tick": self._tick,
            "health": self._health,
            "in_fire": in_fire,
            "in_safe": self.in_safe_strip(pose.x, pose.y),
            "ignited": ignited,
            "coverage": self.fire.coverage(),
            "score": self._health / self.fl.health,
        }
        end = self.fl.end_coverage
        terminated = self._health <= 0.0 or (end is not None and info["coverage"] >= end - 1e-9)
        truncated = self._tick >= self.fl.max_ticks
        if it is None:
            return self._get_obs(), -loss, terminated, truncated, info
        banked = it.total(SAVED)
        it.burn(burning)
        it.tick(running=not (terminated or truncated))
        self._show_items()
        info.update(score=it.score(), saved=it.total(SAVED), job=it.job, job_left=it.left)
        return self._get_obs(), it.total(SAVED) - banked, terminated, truncated, info


class FireLab3DEnv(PrplLab3DEnv):
    """gymnasium wrapper: flat vectors over a fixed object list (robot, lab, cubes, cells, status)."""

    def __init__(
        self,
        num_cubes: int = 0,
        level: int = 1,
        firelab_config: FireLabConfig | None = None,
        num_items: int | None = None,
        values: tuple | None = None,
        **kwargs,
    ) -> None:
        cfg = firelab_config if firelab_config is not None else level_config(level, num_items, values)
        super().__init__(num_cubes=num_cubes, firelab_config=cfg, **kwargs)

    def _create_object_centric_env(self, *args, **kwargs) -> ObjectCentricFireLab3DEnv:
        return ObjectCentricFireLab3DEnv(*args, **kwargs)

    def _get_constant_object_names(self, exemplar_state: ObjectCentricState) -> list[str]:
        names = super()._get_constant_object_names(exemplar_state)
        names += sorted(o.name for o in exemplar_state if o.type in (FireCellType, ItemType))  # cells, then items
        return names + ["status"]

    def _create_env_markdown_description(self) -> str:
        return (
            "The PRPL lab room with a fire that starts in one random floor cell and spreads to "
            "neighbouring cells each tick. Level 1: stay out of the fire. Level 2: carry items to the "
            "safe strip before they burn; only items dropped there count."
        )

    def _create_variant_specific_description(self) -> str:
        if self._object_centric_env.fl.level == 2:
            return "Level 2: one partition with a doorway; ends at health 0, half the floor burning, or max_ticks."
        return "Level 1: no objects, no goal; the episode ends at health 0 or after max_ticks."

    def _create_reward_markdown_description(self) -> str:
        if self._object_centric_env.fl.level == 2:
            return "Reward is the value banked this tick. Score = value banked / value on the floor at reset."
        return "Reward is minus the health lost this tick (0 outside fire). Score = health left / 100."
