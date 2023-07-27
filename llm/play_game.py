from pdb import set_trace as T
from tqdm import tqdm

import numpy as np

import nmmo

from scripted import baselines
from llm.generated_agent import Agent
from environment import process_event_log

SEED = 42

def get_agent_info(env, agent_id):
    # some other old info can be accessed via
    # env.realm.players[agent_id].history
    task = env.agent_task_map[agent_id][0]
    log = {
        "lifetime": env.realm.tick,
        "task_name": "Default StayAlive task" if task.spec_name is None else task.spec_name,
        "task_completed": task.completed,
    }
    achieved, performed, _ = process_event_log(env.realm, [agent_id])
    log.update(achieved)
    log.update(performed)
    return log

config = nmmo.config.Default()
config.PLAYERS = [Agent]

env = nmmo.Env(config, seed=SEED)

agent_info = []
obs = env.reset(seed=SEED)
for t in tqdm(range(128)):
    _, _, d, _ = env.step({})
    agent_info += [get_agent_info(env, agent_id) for agent_id in d if d[agent_id] is True]

# remaining agents
agent_info += [get_agent_info(env, agent_id) for agent_id in env.realm.players]

for key in agent_info[0]:
    if isinstance(agent_info[0][key], (bool, np.bool_)):
        mean_val = np.mean([info[key] for info in agent_info])
        print(f"{key}: Mean value of {mean_val}")
    else:
      max_val = max([info[key] for info in agent_info])
      print(f"{key}: Maximum value of {max_val}")