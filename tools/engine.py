"""Put the engine into the state a game puts it in, before measuring anything.

`searcher.CONTROL[USE_NNUE]` selects the evaluation, and CONTROL starts as zeros, so a tool that
imports searcher and calls `run` directly gets the hand-crafted evaluation -- not the network.
Only agent.py ever set the flag, so every tool that drives the search itself has been measuring
the wrong evaluation.

That is how tools/nodecost.py came to report five network shapes as costing the same to a few
percent: none of them was being used. It is also how tools/probe.py came to report that our
engine plays a losing move at every node count, which was a statement about the hand-crafted
evaluation and not about the network the agent actually ships.

One function, called by every tool that searches, and it says out loud which evaluation is in
force so a run cannot silently measure the other one.
"""

import nnue
import searcher
from searcher import USE_NNUE


def use_network(announce: bool = True) -> bool:
    """Select the network if one is trained, exactly as agent.py does. Returns what is in force."""
    on = bool(nnue.TRAINED)
    searcher.CONTROL[USE_NNUE] = 1 if on else 0
    if announce:
        which = (
            f"network, {nnue.HIDDEN} hidden x {nnue.BUCKETS} king buckets"
            if on
            else "hand-crafted evaluation (no trained network found)"
        )
        print(f"evaluating with the {which}")
    return on
