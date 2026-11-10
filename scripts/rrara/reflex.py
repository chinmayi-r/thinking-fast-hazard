"""RRARA's Rapid-Reflex policy, reimplemented from the paper (Zheng et al. 2025, sec. 3.2, eq. 2): a deterministic rule
policy a_t = pi(O_t, C_t, M_t) over the perceived objects (ids, categories, positions), the task context (hazard, value per
category) and the local map (explored mask, height layer, shortest-path distances), that "prioritizes nearby candidate
objects" with constant-time decisions. No code was released; this is our reading of that paragraph.

Rules, in order, each decision:
  1. holding a target            -> ("drop", None)                       fire/flood: into the bag
  2. standing at a target        -> ("pick_up", id)                      HAZARD sets nearest_object after a walk succeeds
  3. targets on the map          -> ("walk_to", argmin cost) with cost = path_distance / value (HAZARD's greedy uses the
                                    same ratio, fire_flood_heuristic.py:107); path_distance from HAZARD's own A* over the
                                    semantic map (policy/astar.py, the functions the executor itself uses); a target with
                                    no path is skipped
  4. nothing reachable           -> ("explore", None)
Addition not in the paper (A34, flagged): a target whose walk failed k times has its cost multiplied by 2**k for this
episode, so a deterministic policy cannot repeat one failing walk until the budget runs out (seen Nov 9 with HAZARD's rule
agent, 13 failed walks to one handbag). The reflector is the paper's answer to obstruction; this is the reflex's floor.
Inputs are HAZARD's `state` (sem_map, goal_map; fireagent_controller.py:535-563) and `processed_input` (challenge.py:136-163).
"""
import numpy as np

from policy.astar import get_astar_path, get_astar_weight  # HAZARD module (src/HAZARD on PYTHONPATH)


class PaperReflex:
    def __init__(self, task, failure_penalty=True, coarse=4, max_searches=6):
        assert task in ("fire", "flood", "wind")
        self.task = task
        self.failure_penalty = failure_penalty
        # A* is run for at most `max_searches` targets, the nearest by straight line; the rest are skipped this decision.
        # pyastar2d's cost depends on how much of the map it must expand, and on a cluttered real kitchen one search can
        # take 0.2-0.5 s even pooled; this cap bounds the reflex's wall time (A34; the paper precomputes its distances).
        self.max_searches = max_searches
        # A* runs on the map pooled by `coarse` x `coarse` cells (block max of HAZARD's weight, so obstacles stay
        # obstacles). On the full 512x512 map one pyastar2d call costs 0.1-0.2 s and a kitchen with 11 targets took
        # 1.5-2 s per decision (Nov 9 laptop sweep, 47-64 frames charged); the paper's reflex uses precomputed
        # distances and reports 0.00 s. coarse=4 makes it about 16x cheaper; path lengths are rescaled to full cells.
        self.coarse = coarse
        self.agent_type = "custom"
        self.reset([], {})

    def reset(self, goal_objects, objects_info):
        self.target_objects = list(goal_objects)
        self.objects_info = objects_info
        self.failed_walks = {}  # target id -> number of walks that did not arrive
        self.last_action = None
        self.last_costs = {}  # for the log: id -> (path cells, value, cost) at the last ranking

    # ---- map helpers (the same conventions as HAZARD's LLM agent, llm.py:544-549 and 434-447) ----
    @staticmethod
    def _agent_cell(goal_map):
        pts = np.argwhere(goal_map == -2)
        return None if len(pts) == 0 else (int(pts[0][0]), int(pts[0][1]))

    @staticmethod
    def _object_cell(sem_map, obj_id):
        pts = np.argwhere((sem_map["explored"] * sem_map["id"]) == obj_id)
        return None if len(pts) == 0 else (int(round(pts[:, 0].mean())), int(round(pts[:, 1].mean())))

    def _coarse_weight(self, weight):
        c = self.coarse
        if c <= 1:
            return weight
        h, w = weight.shape
        h2, w2 = h // c, w // c
        return weight[:h2 * c, :w2 * c].reshape(h2, c, w2, c).max(axis=(1, 3)).astype(np.float32)

    def _path_cells(self, weight, origin, dest):
        c = self.coarse
        o = (origin[0] // c, origin[1] // c)
        d = (dest[0] // c, dest[1] // c)
        if o == d:
            return 0.0
        path = get_astar_path(weight, o, d)
        if path is None or len(path) == 0:
            return None
        pts = np.asarray(path, dtype=float)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) * c if len(pts) > 1 else 0.0

    def note_result(self, action, success):
        """The agent reports how the last action went (processed_input['action_result'] at the next decision)."""
        if action is not None and action[0] == "walk_to" and not success:
            k = str(action[1])
            self.failed_walks[k] = self.failed_walks.get(k, 0) + 1

    def choose_target(self, state, processed_input):
        holding = processed_input["holding_objects"]
        nearest = processed_input["nearest_object"]
        if holding and (self.task != "wind"):
            self.last_action = ("drop", None)
            return self.last_action
        if not holding and nearest and nearest[0]["category"] in self.target_objects:
            self.last_action = ("pick_up", str(nearest[0]["id"]))
            return self.last_action

        sem_map, goal_map = state["sem_map"], state["goal_map"]
        agent = self._agent_cell(goal_map)
        best, best_cost = None, None
        self.last_costs = {}
        if agent is not None and not holding:
            weight = None
            cands = []
            for obj in processed_input["explored_object_name_list"]:
                if obj["category"] not in self.target_objects:
                    continue
                cell = self._object_cell(sem_map, int(obj["id"]))
                if cell is not None:
                    cands.append((float(np.hypot(cell[0] - agent[0], cell[1] - agent[1])), obj, cell))
            cands.sort(key=lambda t: (t[0], str(t[1]["id"])))
            for _, obj, cell in cands[: self.max_searches]:
                oid = str(obj["id"])
                if weight is None:
                    weight = self._coarse_weight(get_astar_weight(sem_map=sem_map, origin=agent, destination=cell))
                dist = self._path_cells(weight, agent, cell)
                if dist is None:
                    continue
                value = max(float(self.objects_info.get(obj["category"], {}).get("value", 1)), 1e-6)
                cost = (dist + 1.0) / value
                if self.failure_penalty:
                    cost *= 2 ** self.failed_walks.get(oid, 0)
                self.last_costs[oid] = (round(dist, 1), value, round(cost, 2))
                if best_cost is None or cost < best_cost or (cost == best_cost and oid < best):
                    best, best_cost = oid, cost
        self.last_action = ("walk_to", best) if best is not None else ("explore", None)
        return self.last_action
