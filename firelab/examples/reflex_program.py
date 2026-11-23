"""Worked example of a program (firelab/programs.py): the official reflex, written against the observation.

It decides exactly as firelab.agents.reflex does (tested in firelab/test_programs.py), so the reflex's scores apply
to it. A compiled program has this shape: one class Policy with __init__(rules) and decide(obs), nothing imported
beyond the allowed list.

Rules, in order, each decision:
  1. the last "home" failed            -> stop (trapped with the load, or out of time)
  2. something is in the bag           -> ("home",)      bank after every pick
  3. items left that are worth fetching -> ("get", k)    the most valuable; the nearer one on a tie
  4. nothing left                       -> stop
An item whose "get" failed is not tried again: the fire only grows.
"""


class Policy:
    def __init__(self, rules):
        self.rules = rules
        self.skip = set()  # items a "get" could not reach

    def decide(self, obs):
        last, result = obs["last_command"], obs["last_result"]
        if last is not None and not result["done"]:
            if last[0] == "home":
                return None
            if last[0] == "get":
                self.skip.add(last[1])
        if obs["bag"]:
            return ("home",)
        todo = [i for i in obs["items"] if i["where"] == "floor" and i["value"] > 0 and i["id"] not in self.skip]
        if not todo:
            return None
        x, y = obs["robot"]["cell"]
        far = lambda i: abs(i["cell"][0] - x) + abs(i["cell"][1] - y)
        best = max(todo, key=lambda i: (i["value"], -far(i)))
        return ("get", best["id"])
