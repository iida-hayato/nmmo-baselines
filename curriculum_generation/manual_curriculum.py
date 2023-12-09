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


def TickTargert(gs: GameState, subject: Group, num_tick_from: int, num_tick: int):
  """True if the current tick is greater than or equal to the specified num_tick.
  Is progress counter.
  """
  return norm((gs.current_tick - num_tick_from) / (num_tick - num_tick_from))


def TickGEMod(gs: GameState, subject: Group, mod: int, num_tick: int):
  """True if the current tick is greater than or equal to the specified num_tick.
  Is progress counter.
  """
  return norm((gs.current_tick % mod) / num_tick)


# stay alive ... like ... for 300 ticks
# i.e., getting incremental reward for each tick alive as an individual or a team
for num_tick in STAY_ALIVE_GOAL:
  curriculum.append(
    TaskSpec(eval_fn=TickGE, eval_fn_kwargs={"num_tick": num_tick}, sampling_weight=100)
  )

for num_tick in STAY_ALIVE_GOAL1:
  curriculum.append(
    TaskSpec(eval_fn=TickTargert, eval_fn_kwargs={"num_tick_from": 25, "num_tick": num_tick},
             sampling_weight=100 + num_tick)
  )


def PracticeEating(gs, subject, num_tick):
  """The progress, the max of which is 1, should
  * increase small for each eating
  * increase big for the 1st and 3rd eating
  * reach 1 with 10 eatings
  """
  num_eat = len(subject.event.EAT_FOOD)
  progress = num_eat * 0.06
  return norm(
    progress * TickGEMod(gs, subject, mod=100,
                         num_tick=num_tick % 100))  # norm is a helper function to normalize the value to [0, 1]


# for w in range(1, 10):
curriculum.append(TaskSpec(eval_fn=PracticeEating, eval_fn_kwargs={"num_tick": 10}, sampling_weight=10))


def PracticeDrinking(gs, subject):
  num_drink = len(subject.event.DRINK_WATER)
  progress = num_drink * 0.06
  return norm(progress)


curriculum.append(TaskSpec(eval_fn=PracticeDrinking, eval_fn_kwargs={}, sampling_weight=10))

# def TravelWithTick(gs, subject, dist, num_tick):
#   return norm(DistanceTraveled(gs, subject, dist) * TickGE(gs, subject, num_tick=num_tick))
#
# # for event_code in most_essentials:
# #   for cnt in range(1, 200):
# #     curriculum.append(
# #       TaskSpec(
# #         eval_fn=CountEvent,
# #         eval_fn_kwargs={"event": event_code, "N": cnt},
# #         sampling_weight=100,
# #       )
# #     )
# # attack npc
#
# def DefeatEntityWithTick(gs, subject, num_tick, agent_type: str, level: int, num_agent: int):
#   return norm(DefeatEntity(gs, subject, agent_type, level, num_agent) * TickGE(gs, subject, num_tick=num_tick))
#
#
# for level in range(0, 10):
#   curriculum.append(
#     TaskSpec(
#       eval_fn=DefeatEntity,
#       eval_fn_kwargs={"agent_type": "npc", 'level': level, "num_agent": 20 - level, "num_tick": 50},
#       sampling_weight=50 - level * 2,
#     )
#   )
#
#
# def CountEventWithTick(gs, subject, num_tick, event: str, N: int):
#   return norm(CountEvent(gs, subject, event, N) * TickGE(gs, subject, num_tick=num_tick))
#
#
# essential_skills = [
#   "SCORE_HIT",
#   "PLAYER_KILL",
#   "HARVEST_ITEM",
#   "EQUIP_ITEM",
#   "CONSUME_ITEM",
#   "LEVEL_UP",
#   "EARN_GOLD",
#   "LIST_ITEM",
#   "BUY_ITEM",
# ]
# for event_code in essential_skills:
#   for cnt in EVENT_NUMBER_GOAL:
#     curriculum.append(
#       TaskSpec(
#         eval_fn=CountEventWithTick,
#         eval_fn_kwargs={"event": event_code, "N": cnt, "num_tick": 50},
#         sampling_weight=20,
#       )
#     )
#
# # item/market skills, which happen less frequently or should not do too much
# item_skills = [
#   "GIVE_ITEM",
#   "DESTROY_ITEM",
#   "GIVE_GOLD",
# ]
# for event_code in item_skills:
#   curriculum += [
#     TaskSpec(eval_fn=CountEventWithTick, eval_fn_kwargs={
#       "event": event_code, "N": cnt, "num_tick": 50}, sampling_weight=1)
#     for cnt in INFREQUENT_GOAL
#   ]  # less than 10
#
# for resource in {m.Foilage}:
#   curriculum.append(
#     TaskSpec(
#       eval_fn=CanSeeTile,
#       eval_fn_kwargs={"tile_type": resource},
#       sampling_weight=8,
#     )
#   )
# for i in range(1, 10):
#   curriculum.append(TaskSpec(
#     eval_fn=TravelWithTick,
#     eval_fn_kwargs={"dist": i * 5, "num_tick": i * 10},
#     sampling_weight=i
#   ))
#
#
# def CanSeeTileWithTick(gs, subject, num_tick, tile_type: str):
#   return norm(CanSeeTile(gs, subject, tile_type) * TickGE(gs, subject, num_tick=num_tick))
#
#
# # find resource tiles
# for resource in m.Harvestable:
#   curriculum.append(
#     TaskSpec(
#       eval_fn=CanSeeTile,
#       eval_fn_kwargs={"tile_type": resource, "num_tick": 50},
#       sampling_weight=10,
#     )
#   )
#
#
# # practice particular skill with a tool
# def PracticeSkillWithTool(gs, subject, skill, exp, num_tick):
#   return (0.3 * EquipItem(gs, subject, item=TOOL_FOR_SKILL[skill], level=1, num_agent=1) + \
#           0.7 * GainExperience(gs, subject, skill, exp, num_agent=1)) * TickGE(gs, subject, num_tick=num_tick)
#
#
# def AttainSkillWithTick(gs, subject, num_tick, skill: type[Skill], level: int, num_agent: int):
#   return norm(AttainSkill(gs, subject, skill, level, num_agent) * TickGE(gs, subject, num_tick=num_tick))
#
#
# for skill in SKILLS:
#   # level up a skill
#   for level in LEVEL_GOAL[1:]:
#     # since this is an agent task, num_agent must be 1
#     curriculum.append(
#       TaskSpec(
#         eval_fn=AttainSkillWithTick,
#         eval_fn_kwargs={"skill": skill, "level": level, "num_agent": 1, "num_tick": 50},
#         sampling_weight=10 * (6 - level) if level < 6 else 5,
#       )
#     )
#
#   # gain experience on particular skill
#   for exp in EXP_GOAL:
#     curriculum.append(
#       TaskSpec(
#         eval_fn=PracticeSkillWithTool,
#         eval_fn_kwargs={"skill": skill, "exp": exp, "num_tick": 50},
#         sampling_weight=50,
#       )
#     )
#
#
# # occupy the center tile, assuming the Medium map size
# # TODO: it'd be better to have some intermediate targets toward the center
# # curriculum.append(TaskSpec(eval_fn=OccupyTile,
# #                            eval_fn_kwargs={"row": 16, "col": 16}))
#
# def CanSeeAgentWithTick(gs, subject, num_tick, target: str):
#   return norm(CanSeeAgent(gs, subject, target) * TickGE(gs, subject, num_tick=num_tick))
#
#
# def AllMembersWithinRangeWithTick(gs, subject, num_tick, dist: int):
#   return norm(AllMembersWithinRange(gs, subject, dist) * TickGE(gs, subject, num_tick=num_tick))
#
#
# def CanSeeGroupWithTick(gs, subject, num_tick, target: str):
#   return norm(CanSeeGroup(gs, subject, target) * TickGE(gs, subject, num_tick=num_tick))
#
#
# def ScoreHitWithTick(gs, subject, num_tick, combat_style: str, N: int):
#   return norm(ScoreHit(gs, subject, combat_style, N) * TickGE(gs, subject, num_tick=num_tick))
#
#
# def HoardGoldWithTick(gs, subject, num_tick, amount: int):
#   return norm(HoardGold(gs, subject, amount) * TickGE(gs, subject, num_tick=num_tick))
#
#
# def EarnGoldWithTick(gs, subject, num_tick, amount: int):
#   return norm(EarnGold(gs, subject, amount) * TickGE(gs, subject, num_tick=num_tick))
#
#
# def SpendGoldWithTick(gs, subject, num_tick, amount: int):
#   return norm(SpendGold(gs, subject, amount) * TickGE(gs, subject, num_tick=num_tick))
#
#
# def MakeProfitWithTick(gs, subject, num_tick, amount: int):
#   return norm(MakeProfit(gs, subject, amount) * TickGE(gs, subject, num_tick=num_tick))
#
#
# # find the other team leader
# for target in ["left_team_leader", "right_team_leader"]:
#   curriculum.append(TaskSpec(eval_fn=CanSeeAgentWithTick,
#                              eval_fn_kwargs={"target": target, "num_tick": 50}))
#
# curriculum.append(TaskSpec(eval_fn=AllMembersWithinRange,
#                            eval_fn_kwargs={"dist": 9, "num_tick": 50}))
#
# # find the other team (any agent)
# for target in ["left_team", "right_team"]:
#   curriculum.append(TaskSpec(eval_fn=CanSeeGroupWithTick,
#                              eval_fn_kwargs={"target": target, "num_tick": 50}))
#
# # practice specific combat style
# for style in COMBAT_STYLE:
#   for cnt in EVENT_NUMBER_GOAL:
#     curriculum.append(
#       TaskSpec(
#         eval_fn=ScoreHitWithTick,
#         eval_fn_kwargs={"combat_style": style, "N": cnt, "num_tick": 50},
#         sampling_weight=5,
#       )
#     )
#
# # hoarding gold -- evaluated on the current gold
# for amount in EVENT_NUMBER_GOAL:
#   curriculum.append(
#     TaskSpec(
#       eval_fn=HoardGoldWithTick, eval_fn_kwargs={"amount": amount, "num_tick": 50}, sampling_weight=10
#     )
#   )
#
# # earning gold -- evaluated on the total gold earned by selling items
# for amount in EVENT_NUMBER_GOAL:
#   curriculum.append(
#     TaskSpec(eval_fn=EarnGoldWithTick, eval_fn_kwargs={
#       "amount": amount, "num_tick": 50}, sampling_weight=10)
#   )
#
# # spending gold, by buying items
# for amount in EVENT_NUMBER_GOAL:
#   curriculum.append(
#     TaskSpec(
#       eval_fn=SpendGoldWithTick, eval_fn_kwargs={"amount": amount, "num_tick": 50}, sampling_weight=5
#     )
#   )
#
# # making profits by trading -- only buying and selling are counted
# for amount in EVENT_NUMBER_GOAL:
#   curriculum.append(
#     TaskSpec(
#       eval_fn=MakeProfitWithTick, eval_fn_kwargs={"amount": amount, "num_tick": 50}, sampling_weight=3
#     )
#   )
#
#
# # managing inventory space
# def PracticeInventoryManagement(gs, subject, space, num_tick):
#   return InventorySpaceGE(gs, subject, space) * TickGE(gs, subject, num_tick)
#
# def FullyArmedWithTick(gs, subject, num_tick, combat_style: str, level: int, num_agent: int):
#   return norm(FullyArmed(gs, subject, combat_style, level, num_agent) * TickGE(gs, subject, num_tick=num_tick))
#
#
# for num_agent in [1, 2]:
#   for level in LEVEL_GOAL:
#     for combat_style in COMBAT_STYLE:
#       curriculum.append(
#         TaskSpec(eval_fn=FullyArmedWithTick,
#                  eval_fn_kwargs={"combat_style": combat_style, "level": level, "num_agent": num_agent, "num_tick": 50},
#                  sampling_weight=3 * num_agent))
#
# for space in [2, 4, 8]:
#   curriculum += [
#     TaskSpec(
#       eval_fn=PracticeInventoryManagement,
#       eval_fn_kwargs={"space": space, "num_tick": num_tick},
#     )
#     for num_tick in STAY_ALIVE_GOAL
#   ]
#
# def OwnItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
#   return norm(OwnItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))
#
# # own item, evaluated on the current inventory
# for item in ALL_ITEM:
#   for level in LEVEL_GOAL:
#     # agent task
#     for quantity in ITEM_NUM_GOAL:
#       if level + quantity <= 6 or quantity == 1:  # heuristic prune
#         curriculum.append(
#           TaskSpec(
#             eval_fn=OwnItemWithTick,
#             eval_fn_kwargs={
#               "item": item,
#               "level": level,
#               "quantity": quantity,
#               "num_tick": 50,
#             },
#             sampling_weight=4 - level if level < 4 else 1,
#           )
#         )
#
# def EquipItemWithTick(gs, subject, num_tick, item: type[Item], level: int, num_agent: int):
#   return norm(EquipItem(gs, subject, item, level, num_agent) * TickGE(gs, subject, num_tick=num_tick))
#
# # equip item, evaluated on the current inventory and equipment status
# for item in EQUIP_ITEM:
#   for level in LEVEL_GOAL:
#     # agent task
#     curriculum.append(
#       TaskSpec(
#         eval_fn=EquipItemWithTick,
#         eval_fn_kwargs={"item": item, "level": level, "num_agent": 1,"num_tick": 50},
#         sampling_weight=4 - level if level < 4 else 1,
#       )
#     )
#
# def ConsumeItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
#   return norm(ConsumeItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))
#
# # consume items (ration, potion), evaluated based on the event log
# for item in c.consumables:
#   for level in LEVEL_GOAL:
#     # agent task
#     for quantity in ITEM_NUM_GOAL:
#       if level + quantity <= 6 or quantity == 1:  # heuristic prune
#         curriculum.append(
#           TaskSpec(
#             eval_fn=ConsumeItemWithTick,
#             eval_fn_kwargs={
#               "item": item,
#               "level": level,
#               "quantity": quantity,
#               "num_tick": 50,
#             },
#             sampling_weight=4 - level if level < 4 else 1,
#           )
#         )
#
# def HarvestItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
#   return norm(HarvestItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))
#
# # harvest items, evaluated based on the event log
# for item in HARVEST_ITEM:
#   for level in LEVEL_GOAL:
#     # agent task
#     for quantity in ITEM_NUM_GOAL:
#       if level + quantity <= 6 or quantity == 1:  # heuristic prune
#         curriculum.append(
#           TaskSpec(
#             eval_fn=HarvestItemWithTick,
#             eval_fn_kwargs={
#               "item": item,
#               "level": level,
#               "quantity": quantity,
#               "num_tick": 50,
#             },
#             sampling_weight=4 - level if level < 4 else 1,
#           )
#         )
# def ListItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
#   return norm(ListItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))
#
# # list items, evaluated based on the event log
# for item in ALL_ITEM:
#   for level in LEVEL_GOAL:
#     # agent task
#     for quantity in ITEM_NUM_GOAL:
#       if level + quantity <= 6 or quantity == 1:  # heuristic prune
#         curriculum.append(
#           TaskSpec(
#             eval_fn=ListItemWithTick,
#             eval_fn_kwargs={
#               "item": item,
#               "level": level,
#               "quantity": quantity,
#               "num_tick": 50,
#             },
#             sampling_weight=4 - level if level < 4 else 1,
#           )
#         )
#
# def BuyItemWithTick(gs, subject, num_tick, item: type[Item], level: int, quantity: int):
#   return norm(BuyItem(gs, subject, item, level, quantity) * TickGE(gs, subject, num_tick=num_tick))
#
# # buy items, evaluated based on the event log
# for item in ALL_ITEM:
#   for level in LEVEL_GOAL:
#     # agent task
#     for quantity in ITEM_NUM_GOAL:
#       if level + quantity <= 6 or quantity == 1:  # heuristic prune
#         curriculum.append(
#           TaskSpec(
#             eval_fn=BuyItemWithTick,
#             eval_fn_kwargs={
#               "item": item,
#               "level": level,
#               "quantity": quantity,
#               "num_tick": 50,
#             },
#             sampling_weight=4 - level if level < 4 else 1,
#           )
#         )

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
