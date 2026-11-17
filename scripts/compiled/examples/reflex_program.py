"""Worked example of a compiled program (scripts/compiled/API.md): RRARA's rapid-reflex policy (Zheng et al. 2025,
sec. 3.2, eq. 2; our reimplementation scripts/rrara/reflex.py) written against the observation API. A compiler's
output must have exactly this shape: a module with `class Policy(task, target_values)` and `decide(obs) -> action`,
no imports beyond the allowed list, no file/network/model calls, deterministic.

Rules, in order, each decision:
  1. holding a target            -> ("drop", None)           fire/flood: into the bag, which is what scores
  2. standing at a target        -> ("pick_up", id)          at_object is set only after a walk_to arrived
  3. targets on the map          -> ("walk_to", argmin cost) cost = (path cells + 1) / value, x 2 per earlier failed
                                                             walk to that object (A34); the six nearest by straight
                                                             line are planned, the rest skipped this decision
  4. nothing reachable           -> ("explore", None)
"""
from compiled.api import Paths


class Policy:
    MAX_SEARCHES = 6  # A* calls per decision; one pooled search is a few ms, a full-map one 0.1-0.2 s
    COARSE = 4        # pool the map 4 x 4 before planning (0.4 m cells); obstacles stay obstacles

    def __init__(self, task, target_values):
        self.task = task
        self.values = dict(target_values)  # category -> value, as given to every HAZARD agent at reset
        self.failed_walks = {}             # memory across decisions: object id -> walks that did not arrive

    def decide(self, obs):
        # how the previous action went
        last, result = obs["last_action"], obs["last_result"]
        if last is not None and last[0] == "walk_to" and result is not None and not result["ok"]:
            self.failed_walks[last[1]] = self.failed_walks.get(last[1], 0) + 1

        if obs["holding"] and self.task != "wind":
            return ("drop", None)
        at = obs["at_object"]
        if not obs["holding"] and at is not None and at["is_target"]:
            return ("pick_up", at["id"])
        if obs["agent_cell"] is None or obs["holding"]:
            return ("explore", None)

        cands = [o for o in obs["objects"] if o["is_target"] and o["cell"] is not None]
        cands.sort(key=lambda o: (o["straight_cells"], o["id"]))
        paths = Paths(obs, coarse=self.COARSE)
        best, best_cost = None, None
        for o in cands[: self.MAX_SEARCHES]:
            dist = paths.cells(o["id"])
            if dist is None:
                continue
            value = max(o["value"], 1e-6)
            cost = (dist + 1.0) / value * 2 ** self.failed_walks.get(o["id"], 0)
            if best_cost is None or cost < best_cost or (cost == best_cost and o["id"] < best):
                best, best_cost = o["id"], cost
        return ("walk_to", best) if best is not None else ("explore", None)
