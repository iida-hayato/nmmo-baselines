"""Manual test for creating learning curriculum manually"""
# pylint: disable=invalid-name,redefined-outer-name,bad-builtin
# pylint: disable=wildcard-import,unused-wildcard-import
from typing import List

import nmmo
import nmmo.lib.material as m
import nmmo.systems.item as Item
import nmmo.systems.skill as Skill
from nmmo.task import constraint as c
from nmmo.task.base_predicates import (
  norm,
  count,
  StayAlive,
  AttainSkill,
  BuyItem,
  CanSeeAgent,
  CanSeeGroup,
  CanSeeTile,
  ConsumeItem,
  CountEvent,
  EarnGold,
  EquipItem,
  GainExperience,
  HarvestItem,
  HoardGold,
  InventorySpaceGE,
  AllMembersWithinRange,
  ListItem,
  MakeProfit,
  OccupyTile,
  DefeatEntity,
  OwnItem,
  ScoreHit,
  SpendGold,
  TickGE, FullyArmed,
  DistanceTraveled,
)
from nmmo.task.task_spec import TaskSpec, check_task_spec
from nmmo.task.group import Group
from nmmo.task.game_state import GameState
from nmmo.lib import utils

EVENT_NUMBER_GOAL = [1, 2, 3, 5, 7, 9, 12, 15, 20, 30, 50]
INFREQUENT_GOAL = list(range(1, 10))
EXP_GOAL = [10, 20, 30, 40, 50, 100, 150, 200, 300, 500, 700]
STAY_ALIVE_GOAL = [200, 500, 700]

STAY_ALIVE_GOAL1 = [50, 100, 175, ]

LEVEL_GOAL = list(range(2, 10))  # TODO: get config
AGENT_NUM_GOAL = ITEM_NUM_GOAL = [1, 2, 3, 4, 5]  # competition team size: 8
SKILLS = c.combat_skills + c.harvest_skills
COMBAT_STYLE = c.combat_skills
ALL_ITEM = c.armour + c.weapons + c.tools + c.ammunition + c.consumables
EQUIP_ITEM = c.armour + c.weapons + c.tools + c.ammunition
HARVEST_ITEM = c.weapons + c.ammunition + c.consumables
TOOL_FOR_SKILL = {
  Skill.Melee: Item.Spear,
  Skill.Range: Item.Bow,
  Skill.Mage: Item.Wand,
  Skill.Fishing: Item.Rod,
  Skill.Herbalism: Item.Gloves,
  Skill.Carving: Item.Axe,
  Skill.Prospecting: Item.Pickaxe,
  Skill.Alchemy: Item.Chisel,
}

curriculum: List[TaskSpec] = []

# explore, eat, drink, attack any agent, harvest any item, level up any skill
#   which can happen frequently
most_essentials = [
  "EAT_FOOD",
  "DRINK_WATER",
]
MAX_AGENTS = 8


def AliveTick(gs: GameState, subject: Group, num_tick: int):
  """True if the current tick is greater than or equal to the specified num_tick.
  Is progress counter.And agent is alive.
  Lose when agents are dead.
  """
  rewardPerAgent = count(subject.health > 0) / MAX_AGENTS * 0.01  # 0.01~0.00125
  reward = rewardPerAgent * gs.current_tick / num_tick
  if (gs.current_tick >= 30):
    reward += 0.1 + rewardPerAgent * 20
  if (gs.current_tick >= 50):
    reward += 0.1 + rewardPerAgent * 20
  if (gs.current_tick >= 70):
    reward += 0.1 + rewardPerAgent * 20
  print("AliveTick", reward)
  print("entity", subject)
  return norm(reward)


def KeepFullHungry(gs: GameState, subject: Group):
  reward = 1
  if any(subject.food < 75):
    reward -= 0.1
  if any(subject.food < 50):
    reward -= 0.2
  if any(subject.food < 25):
    reward -= 0.3
  if any(subject.food < 10):
    reward -= 0.5
  return norm(reward)


def AliveTickTargert(gs: GameState, subject: Group, num_tick_from: int, num_tick: int):
  """True if the current tick is greater than or equal to the specified num_tick.
  Is progress counter. And agent is alive.
  Lose when agents are dead.
  """
  max_agents = 8
  return norm((gs.current_tick - num_tick_from) / (num_tick - num_tick_from) * count(subject.health > 0) / max_agents)


def TickGEMod(gs: GameState, subject: Group, mod: int, num_tick: int):
  """True if the current tick is greater than or equal to the specified num_tick.
  Is progress counter.
  """
  return norm((gs.current_tick % mod) / num_tick)


def TravelWithTick(gs, subject, dist, num_tick):
  """Agents travel a certain distance. After num_tick, the agent will be rewarded."""
  return norm(DistanceTraveled(gs, subject, dist) * TickGE(gs, subject, num_tick=num_tick))


def FirstDistanceTraveled(gs: GameState, subject: Group, dist: int):
  """True if the summed l-inf distance between each agent's current pos and spawn pos
        is greater than or equal to the specified _dist.
  """
  if not any(subject.health > 0):
    return 0
  r = subject.row
  c = subject.col
  dists = utils.linf(list(zip(r, c)), [gs.spawn_pos[id_] for id_ in subject.entity.id])
  distance = dists.sum()
  reward = 0 if distance == 0 else max(0.06, distance / dist)
  if distance >= dist / 6:
    reward += 0.1
  if distance >= dist / 3:
    reward += 0.1
  if distance >= dist * 2 / 3:
    reward += 0.1
  print("FirstDistanceTraveled", reward)
  return norm(reward)


def PracticeEating(gs, subject, target_eat_num):
  """
  Reward agents for eating food.
  This is most important for survival.
  """
  num_eat = len(subject.event.EAT_FOOD)
  progress = num_eat * 0.01
  if num_eat >= 5:
    progress += 0.1
  if num_eat >= 10:
    progress += 0.1
  if subject.entity.food <= 75:
    progress += 0.1
  if subject.entity.food <= 50:
    progress += 0.2
  if subject.entity.food <= 25:
    progress += 0.3
  print("PracticeEating", progress)
  return norm(progress)


def CanSeeTileWithTick(gs, subject, num_tick, tile_type: str):
  if (gs.current_tick > num_tick):
    return norm(CanSeeTile(gs, subject, tile_type))
  return norm(CanSeeTile(gs, subject, tile_type) * 0.01)


def CanSeeTileWithHunger(gs, subject, tile_type: str):
  reward = 0
  if (CanSeeTile(gs, subject, tile_type) > 0.5):
    reward += 0.01
  if (subject.entity.food < 50):
    reward += CanSeeTile(gs, subject, tile_type)
  return norm(reward)


def PracticeDrinking(gs, subject):
  """
  Reward agents for drinking water
  """
  num_drink = len(subject.event.DRINK_WATER)
  progress = num_drink * 0.06
  return norm(progress)


def DefeatEntityWithTick(gs, subject, num_tick, agent_type: str, level: int, num_agent: int):
  return norm(DefeatEntity(gs, subject, agent_type, level, num_agent) * TickGE(gs, subject, num_tick=num_tick))


def CountEventWithTick(gs, subject, num_tick, event: str, N: int):
  return norm(CountEvent(gs, subject, event, N) * TickGE(gs, subject, num_tick=num_tick))


# practice particular skill with a tool
def PracticeSkillWithTool(gs, subject, skill, exp, num_tick):
  return (0.3 * EquipItem(gs, subject, item=TOOL_FOR_SKILL[skill], level=1, num_agent=1) + \
          0.7 * GainExperience(gs, subject, skill, exp, num_agent=1)) * TickGE(gs, subject, num_tick=num_tick)


def AttainSkillWithTick(gs, subject, num_tick, skill: type[Skill], level: int, num_agent: int):
  return norm(AttainSkill(gs, subject, skill, level, num_agent) * TickGE(gs, subject, num_tick=num_tick))


def CanSeeAgentWithTick(gs, subject, num_tick, target: str):
  return norm(CanSeeAgent(gs, subject, target) * TickGE(gs, subject, num_tick=num_tick))


def AllMembersWithinRangeWithTick(gs, subject, num_tick, dist: int):
  return norm(AllMembersWithinRange(gs, subject, dist) * TickGE(gs, subject, num_tick=num_tick))


def CanSeeGroupWithTick(gs, subject, num_tick, target: str):
  return norm(CanSeeGroup(gs, subject, target) * TickGE(gs, subject, num_tick=num_tick))


def ScoreHitWithTick(gs, subject, num_tick, combat_style: str, N: int):
  return norm(ScoreHit(gs, subject, combat_style, N) * TickGE(gs, subject, num_tick=num_tick))


def HoardGoldWithTick(gs, subject, num_tick, amount: int):
  return norm(HoardGold(gs, subject, amount) * TickGE(gs, subject, num_tick=num_tick))


def EarnGoldWithTick(gs, subject, num_tick, amount: int):
  return norm(EarnGold(gs, subject, amount) * TickGE(gs, subject, num_tick=num_tick))


def SpendGoldWithTick(gs, subject, num_tick, amount: int):
  return norm(SpendGold(gs, subject, amount) * TickGE(gs, subject, num_tick=num_tick))


def MakeProfitWithTick(gs, subject, num_tick, amount: int):
  return norm(MakeProfit(gs, subject, amount) * TickGE(gs, subject, num_tick=num_tick))


# managing inventory space
def PracticeInventoryManagement(gs, subject, space, num_tick):
  return InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick)


def FullyArmedWithTick(gs, subject, num_tick, combat_style: str, level: int, num_agent: int):
  return norm(FullyArmed(gs, subject, combat_style, level, num_agent) * TickGE(gs, subject, num_tick=num_tick))


def OwnItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
  return norm(OwnItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))


def EquipItemWithTick(gs, subject, num_tick, item: type[Item], level: int, num_agent: int):
  return norm(EquipItem(gs, subject, item, level, num_agent) * TickGE(gs, subject, num_tick=num_tick))


def ConsumeItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
  return norm(ConsumeItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))


def HarvestItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
  return norm(HarvestItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))


def ListItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
  return norm(ListItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))


def BuyItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
  return norm(BuyItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))


# curriculum.append(TaskSpec(eval_fn=AliveTick, eval_fn_kwargs={"num_tick": 1024}, sampling_weight=10))
#
# curriculum.append(TaskSpec(eval_fn=PracticeEating, eval_fn_kwargs={"target_eat_num": 10}, sampling_weight=10))
#
# curriculum.append(TaskSpec(eval_fn=KeepFullHungry, eval_fn_kwargs={}, sampling_weight=10))

curriculum.append(TaskSpec(
  eval_fn=FirstDistanceTraveled,
  eval_fn_kwargs={"dist": 12},
  sampling_weight=5
))

curriculum.append(TaskSpec(
  eval_fn=FirstDistanceTraveled,
  eval_fn_kwargs={"dist": 60},
  sampling_weight=5
))

# for resource in {m.Foilage}:
#   curriculum.append(
#     TaskSpec(
#       eval_fn=CanSeeTileWithHunger,
#       eval_fn_kwargs={"tile_type": resource},
#       sampling_weight=8,
#     )
#   )

if __name__ == "__main__":
  import multiprocessing as mp
  from contextlib import contextmanager

  import dill
  import numpy as np
  import psutil


  @contextmanager
  def create_pool(num_proc):
    pool = mp.Pool(processes=num_proc)
    yield pool
    pool.close()
    pool.join()


  # 1609 task specs: divide the specs into chunks
  num_workers = round(psutil.cpu_count(logical=False) * 0.7)
  spec_chunks = np.array_split(curriculum, num_workers)
  with create_pool(num_workers) as pool:
    pool.map(check_task_spec, spec_chunks)

  # test if the task spec is pickalable
  with open("pickle_test.pkl", "wb") as f:
    dill.dump(curriculum, f)
