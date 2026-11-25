import math


class Policy:
    def __init__(self, rules):
        self.rules = rules
        self.skip = set()
        self.forced_home_tried = False

    def _home_est(self, i, j):
        if j >= 11:
            return 5
        if j > 5:
            return (11 - j) * 4 + 5
        d = min(abs(i - di) for di in (6, 7, 8, 9)) + (5 - j)
        return (d + 6) * 4 + 5

    def _time_est(self, obs):
        tick = obs.get("tick", 0) or 0
        share = obs.get("burning_share", 0.0) or 0.0
        end = obs.get("end_share", 0.5) or 0.5
        left = obs.get("ticks_left", 500) or 0
        if share <= 0.005 or tick <= 0:
            return left
        if share >= end:
            return 0
        est = tick * (math.sqrt(end / share) - 1.0) * 0.8
        return min(left, est)

    def decide(self, obs):
        try:
            return self._decide(obs)
        except Exception:
            try:
                if obs.get("bag"):
                    return ("home", True)
            except Exception:
                pass
            return None

    def _decide(self, obs):
        last, result = obs.get("last_command"), obs.get("last_result")
        bag = obs.get("bag") or []
        if last is not None and result is not None and not result.get("done", True):
            if last[0] == "home":
                if bag and not self.forced_home_tried:
                    self.forced_home_tried = True
                    return ("home", True)
                return None
            if last[0] == "get" and len(last) > 1:
                self.skip.add(last[1])

        time_est = self._time_est(obs)
        home = obs.get("home") or {}
        hticks = home.get("ticks")

        if bag:
            if hticks is None:
                return ("home", True)
            if obs["robot"].get("in_fire"):
                return ("home",)
            # near-free extra pick: item already in reach
            best = None
            for it in obs["items"]:
                if it["where"] != "floor" or it["id"] in self.skip:
                    continue
                t = it.get("ticks_to_get")
                if t is None or t > 10 or it["value"] <= 0.05:
                    continue
                if t + hticks + 10 > time_est:
                    continue
                if best is None or it["value"] > best["value"]:
                    best = it
            if best is not None:
                return ("get", best["id"])
            return ("home",)

        cands = []
        for it in obs["items"]:
            if it["where"] != "floor" or it["id"] in self.skip:
                continue
            t = it.get("ticks_to_get")
            if t is None or it["value"] <= 0.05:
                continue
            val = it["value"]
            if it.get("burning"):
                val = max(0.0, val - it["start"] * 0.02 * t)
            if val <= 0.05:
                continue
            he = self._home_est(it["cell"][0], it["cell"][1])
            cands.append((it, val, t, he))
        if not cands:
            return None
        feas = [c for c in cands if c[2] + c[3] + 5 <= time_est]
        if feas:
            best = max(feas, key=lambda c: (c[1], -c[2]))
        else:
            best = max(cands, key=lambda c: (c[1] / max(c[2] + c[3], 1.0), -c[2]))
        return ("get", best[0]["id"])
