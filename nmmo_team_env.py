
from typing import Any, Dict, Tuple

import gym
import nmmo
import numpy as np

from feature_extractor.feature_extractor import FeatureExtractor
from model.model import ModelArchitecture
from team_env import TeamEnv
from team_helper import TeamHelper


class NMMOTeamEnv(TeamEnv):
  def __init__(self, config: nmmo.config.Config(), team_helper: TeamHelper):
    super().__init__(nmmo.Env(config), team_helper)

    self._config = config
    self._feature_extractors = [
      FeatureExtractor(team_helper.teams, tid, config) for tid in team_helper.teams
    ]

  def _box(self, *shape):
    return gym.spaces.Box(low=-2**20, high=2**20, shape=shape, dtype=np.float32)

  def observation_space(self, team_id: int) -> gym.Space:
    team_size = self._team_helper.team_size[team_id]
    inventory_capacity = 1
    if self._config.ITEM_SYSTEM_ENABLED:
      inventory_capacity = self._config.ITEM_INVENTORY_CAPACITY

    action_space = gym.spaces.Dict({
      name: self._box(self._team_helper.team_size[team_id], dim)
      for name, dim in ModelArchitecture.ACTION_NUM_DIM.items()
    })

    return gym.spaces.Dict({
      "tile": self._box(ModelArchitecture.TILE_NUM_CHANNELS, *ModelArchitecture.TILE_IMG_SIZE),
      "item_type": self._box(team_size, inventory_capacity),
      "item": self._box(team_size, inventory_capacity, ModelArchitecture.ITEM_NUM_FEATURES),
      "team": self._box(team_size, ModelArchitecture.TEAM_NUM_FEATURES),
      "team_mask": self._box(team_size),
      "enemy": self._box(team_size, ModelArchitecture.ENTITY_NUM_FEATURES),
      "enemy_mask": self._box(team_size),
      "npc": self._box(team_size, ModelArchitecture.ENTITY_NUM_FEATURES),
      "npc_mask": self._box(team_size),
      "game": self._box(ModelArchitecture.GAME_NUM_FEATURES),
      "legal": action_space,
      "prev_act": action_space,
      "reset": self._box(1),
    })

  def action_space(self, team_id: int) -> gym.Space:
    return gym.spaces.Dict({
      name: gym.spaces.MultiDiscrete(
        [dim for _ in range(self._team_helper.team_size[team_id])])
      for name, dim in ModelArchitecture.ACTION_NUM_DIM.items()
    })

  def reset(self, **kwargs) -> Dict[int, Any]:
    obs = super().reset(**kwargs)
    for tid, team_obs in obs.items():
      team_obs = self._convert_team_obs_to_agent_ids(tid, team_obs)
      self._feature_extractors[tid].reset(team_obs)
      obs[tid] = self._feature_extractors[tid](team_obs)

    return obs

  def step(self, actions: Dict[int, Dict[str, Any]]):
    actions = {
      tid: self._feature_extractors[tid].translate_actions(a)
      for tid, a in actions.items()
    }

    obs, rewards, dones, infos = super().step(actions)
    for tid, team_obs in obs.items():
      obs[tid] = self._feature_extractors[tid](
        self._convert_team_obs_to_agent_ids(tid, team_obs))
    return obs, rewards, dones, infos

  def _convert_team_obs_to_agent_ids(self, team_id, team_obs):
    return {
      self._team_helper.agent_id(team_id, pos): obs
      for pos, obs in team_obs.items()
    }