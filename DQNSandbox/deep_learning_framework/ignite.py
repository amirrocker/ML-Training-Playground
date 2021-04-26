import deep_learning_framework
import enum
import time
from typing import Optional


class EpisodeEvents(enum.Enum):
    EPISODE_COMPLETED = "episode_completed"
    BOUND_REWARD_TRACKER = "bound_reward_reached"
    BEST_REWARD_REACHED = "best_reward_reached"


class EndOfEpisodeHandler:
    def __init__(self, exp_source: deep_learning_framework.experience.ExperienceSource):
        print("End of episode handler called")