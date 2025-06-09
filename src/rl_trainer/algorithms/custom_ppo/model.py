from __future__ import annotations
from typing import Optional, Any

from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import Schedule


class CustomPPO(PPO):
    """
    The same architecture, but we allow the user to pass an *arbitrary* function for dynamic `clip_range`
    (in SB3, the default is the constant 0.2)

    Example schedule: linear decay from 0.2 to 0.05 over the entire training
    """

    def __init__(
        self,
        *args: Any,
        clip_range_schedule: Optional[Schedule] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._clip_range_schedule: Schedule = clip_range_schedule if clip_range_schedule is not None else lambda frac: 0.2

    def _update_clip_range(self, progress_remaining: float) -> float:
        return float(self._clip_range_schedule(progress_remaining))
