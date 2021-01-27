import ptan_self
import enum
import time
from typing import Optional


class EpisodeEvents(enum.Enum):
    EPISODE_COMPLETED = "episode_completed"
    BOUND_REWARD_TRACKER = "bound_reward_reached"
    BEST_REWARD_REACHED = "best_reward_reached"


class EndOfEpisodeHandler:
    def __init__(self, exp_source: ptan_self.experience.ExperienceSource):