"""The clock: what a thought costs in world time (PLAN.md section 6, chunk 6b; CR, Nov 24).

Every thought is costed two ways and both are logged, whichever one is charged:
  measured  real seconds of the whole decision on this machine;
  declared  search steps x a fixed number of seconds per step, the same on every machine
            (for a language model later: a fixed number of seconds per call).
Seconds become ticks as ceil(seconds x 10): one tick is 0.1 s (PLAN D2, D3; RRARA's rule with 10 for 30).

Modes: "off" charges nothing (the unlimited-time agent of CR's second condition); "measured" and "declared"
charge their own ticks; "fixed" charges a given number of ticks per thought, as the rule check did.

The estimate. A thinking agent plans for the moment its thought will end, and must say when that is before it
starts (decision 2). With "off" and "fixed" it knows exactly. Otherwise: the first thought of an episode uses
FIRST_ESTIMATE, later ones what the last thought was charged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

TICKS_PER_SECOND = 10
# Seconds one search step is declared to cost. FROZEN Nov 24 (decision 3, A83) from CR's timing on a Della compute
# node (della-r3c3n13, Xeon Gold 6242 at 2.80 GHz): the first plans of rooms 0 to 9 at 8 items took 14.314 s for
# 81 633 search steps, which is 0.1753 ms per step (scripts/seconds_per_step.py). At 10 items the same node took
# 0.2321 ms per step (42.350 s for 182 502 steps): a step tries more actions when more items are left.
SECONDS_PER_STEP = 0.000175
SECONDS_PER_STEP_10_ITEMS = 0.000232
# Ticks the first thought is expected to take, by items on the floor: the median first plan of that Della timing
# under the declared cost (8 items: 5206 steps, 0.91 s; 10 items: 40 990 steps, 9.5 s). Anything not listed: 1 tick.
FIRST_ESTIMATE = {8: 10, 10: 95}


def ticks(seconds: float) -> int:
    return math.ceil(seconds * TICKS_PER_SECOND - 1e-9)


@dataclass
class Clock:
    mode: str = "off"  # "off", "measured", "declared" or "fixed"
    fixed_ticks: int = 0
    seconds_per_step: float = SECONDS_PER_STEP
    last: int | None = field(default=None, repr=False)  # what the last thought of this episode was charged

    def start(self) -> None:
        """Call at the start of every episode."""
        self.last = None

    def estimate(self, items_on_floor: int) -> int:
        """Ticks the agent expects its next thought to take."""
        if self.mode in ("off", "fixed"):
            return self.fixed_ticks if self.mode == "fixed" else 0
        return self.last if self.last is not None else FIRST_ESTIMATE.get(items_on_floor, 1)

    def cost(self, measured_s: float, steps: int) -> dict:
        """Both costs of a thought that took measured_s real seconds and `steps` search steps, and the ticks charged."""
        declared_s = steps * self.seconds_per_step
        both = {"measured": ticks(measured_s), "declared": ticks(declared_s)}
        charged = {"off": 0, "fixed": self.fixed_ticks, **both}[self.mode]
        self.last = charged
        return {"measured_s": round(measured_s, 5), "steps": steps, "declared_s": round(declared_s, 5),
                "ticks_measured": both["measured"], "ticks_declared": both["declared"], "charged": charged}
