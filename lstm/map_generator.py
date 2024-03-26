import time
from test import *
from matplotlib import pyplot as plt
import multiprocessing as mp
import numpy as np 

plt.rcParams['figure.figsize'] = [20, 10]

NUMBER_OF_AGENTS: int = 30
ENVIRONMENT_DIMEMSION: "tuple[int, int]" = (20, 20)

def create_random_map(*ENVIRONMENT_DIMEMSION): 

    randomMapGenerator = RandomMapGenerator(*ENVIRONMENT_DIMEMSION)
    env, agent_st = get_env_agents(randomMapGenerator)

    print(f'agent st: {agent_st}')
    start, goal = update_agent_st(agent_st)

    print(f'generated env: {env}')
    print(f'agent start {start} and goal {goal}')

    env_dict = convert_environment_to_dict(env) # convert to dict 

    # print(f'env dict converted version: {env_dict}')

    return env_dict, env, agent_st

def update_agent_st(agent_st):
    starting_poses = {}
    starting_goals = {}
    
    for idx, (start, goal) in enumerate(agent_st):
        starting_poses[idx] = start
        starting_goals[idx] = goal
    
    return starting_poses, starting_goals

def get_env_agents(randomMapGenerator): # -> tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]]:

    # Get a new environment
    _, environment = randomMapGenerator.get_graph()

    # Create a graph from the environment
    G = Graph(environment)

    # Create agents. Creates a start and goal for each agent
    agents = [Agent(G) for _ in range(NUMBER_OF_AGENTS)]
    
    # Collet the start and goal of each agent
    agent_st = [(agent.start, agent.goal) for agent in agents]

    return environment, agent_st


# make into format that cbs code will accept 
def get_neighbors(row, col, environment):
    neighbors = []
    rows, cols = len(environment), len(environment[0])
    
    for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols and environment[new_row][new_col] == 1:
            neighbors.append((new_row, new_col))
    
    return neighbors

def convert_environment_to_dict(environment):
    graph_dict = {}
    rows, cols = len(environment), len(environment[0])
    
    for row in range(rows):
        for col in range(cols):
            if environment[row][col] == 1:
                graph_dict[(row, col)] = get_neighbors(row, col, environment)
    
    return graph_dict

# testing purposes 
env_dict, env, state = create_random_map(*ENVIRONMENT_DIMEMSION) # env_dict is graph rep from env
