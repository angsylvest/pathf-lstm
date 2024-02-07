import heapq
from typing import List, Tuple, Dict, Union
import pandas as pd 

class Node:
    def __init__(self, paths, cost):
        self.paths = paths  # Dictionary mapping agent ID to its path
        self.cost = cost  # Total cost of the paths

    # Define custom comparison methods
    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return self.cost == other.cost

def heuristic(current, goal):
    return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

# generates dictionary of edges for graph of size (x_dim, y_dim)
def generate_edges_dict(x_dim, y_dim):
    edges_dict = {}

    for current_x in range(x_dim):
        for current_y in range(y_dim):
            current_pose = (current_x, current_y)
            neighbors = []

            for direction in [(1, 0), (0, -1), (-1, 0), (0, 1), (0, 0)]:
                dir_x, dir_y = direction
                neighbor_x, neighbor_y = (current_x + dir_x, current_y + dir_y)

                if 0 <= neighbor_x < x_dim and 0 <= neighbor_y < y_dim:
                    neighbor_pose = (neighbor_x, neighbor_y)
                    neighbors.append(neighbor_pose)

            edges_dict[current_pose] = neighbors

    return edges_dict

def is_conflict_free(paths):
    # Check if there are any conflicting positions for each agent
    conflict_positions = {}

    for agent1, path1 in paths.items():
        agent_conflicts = set()

        for pos in path1[1:-1]:  # Exclude the first and last position of the path (endpoint) (since these can't change)
            # Check if pos appears in the path of another agent
            for agent2, path2 in paths.items():
                if agent1 != agent2 and pos in path2[:-1]:  # Exclude the last position of the other path
                    agent_conflicts.add(pos)

        if agent_conflicts:
            conflict_positions[agent1] = list(agent_conflicts)

        else:
            conflict_positions[agent1] = []

    conflict_free = all(not conflicts for conflicts in conflict_positions.values())
    return conflict_positions, conflict_free

def find_path(graph, start, goal, occupied_positions, other_agents_goals) -> List[Tuple[int, int]]:
    open_set = [(0, start, [])]  # (f-value, current node, path)
    closed_set = set()

    while open_set:
        f, current, path = heapq.heappop(open_set)

        if current == goal:
            return path + [current]

        if current in closed_set or current in occupied_positions:
            continue

        closed_set.add(current)

        for neighbor in graph[current]:
            neighbor_pos = neighbor  # Extract position from neighbor tuple

            # Skip neighbors that are occupied or in conflict positions
            if neighbor_pos in occupied_positions:
                continue

            g_value = len(path)
            h_value = heuristic(neighbor_pos, goal)
            new_f = g_value + h_value

            heapq.heappush(open_set, (new_f, neighbor_pos, path + [current]))

    raise ValueError('Could not find path that satisfies constraints')


def cbs(graph, agents, goal_positions) -> Union[Dict[str, List[Tuple[int, int]]], None]:
    open_set = [Node({}, 0)]  # Start with an empty set of paths
    closed_set = set()

    paths = {}
    conflict_positions = {}

    # Initially run A* for each agent (no consideration of others)
    for agent, goal in agents.items():
        initial_position = agents[agent]
        path = find_path(graph, initial_position, goal_positions[agent], [], [])
        paths[agent] = path
        conflict_positions[agent] = []

    # print(f"All paths: {paths}")

    problem_pos, conflict_free = is_conflict_free(paths)


    while not conflict_free and any(len(problem_pos[agent]) > 0 for agent in agents):  # While there is a conflict
        # Identify conflict nodes + remove and recalculate solution
        for agent, initial_pose in agents.items():
            problem_pos, conflict_free = is_conflict_free(paths)

            if len(problem_pos[agent]) == 0:
                break

            # Remove one node that is causing conflict
            agent_path = paths[agent]

            selected_pos = problem_pos[agent].pop()
            conflict_positions[agent].append(selected_pos)

            # print(f"Agent path {agent_path} with problematic pos at {selected_pos} and conflict pos {conflict_positions}")

            # print(f'path arguments initial pose: {initial_pose} with goal position {goal_positions[agent]} and conflict positions {conflict_positions[agent]} in graph {graph}')
            path = find_path(graph, initial_pose, goal_positions[agent], conflict_positions[agent], [])
            # print(f"Updated generated path for agent {agent} with initial pos {initial_pose} with goal {goal_positions[agent]} and occupied pos {conflict_positions[agent]}\nPath: {path}")

            paths[agent] = path

            # Recalculate conflict-free status after updating paths
            problem_pos, conflict_free = is_conflict_free(paths)

    return paths if conflict_free else None


def update_env(num_rows, num_cols, agent_positions, goal_positions): 
    # Initialize an empty grid with tuples representing each position
    grid = [[(0, 0, 0, 0)] * num_cols for _ in range(num_rows)]
    
    # Update the grid with current positions and goal positions
    for i, (agent_pos, goal_pos) in enumerate(zip(agent_positions, goal_positions)):
        row, col = agent_pos
        goal_row, goal_col = goal_pos
        
        # Update the grid with the agent's current position and goal position
        grid[row][col] = (row, col, goal_row, goal_col)
    
    return grid


def add_to_dataset(env, paths, goal_positions): # env is initially just edge representation 
    # will be initially just one env but will create relevant rows for each agent to be added to dataset
    env_representation = update_env(5, 5, [path[0] for path in paths.values()], [goal_position for goal_position in goal_positions.values()])

    max_length = max(len(value_list) for value_list in paths.values())

    for i in range(1, max_length): 
        for agent in paths.keys():
            agent_id = agent 

            if len(paths[agent]) > i: 
                current_position_x, current_position_y = paths[agent][i-1]
                next_position_x, next_position_y = paths[agent][i]
                
            goal_position_x, goal_position_y = goal_positions[agent]

            x_context_row = {"time_stamp": i,"env_representation": env_representation}
            x_time_row = {"time_stamp": i,"agent_id": agent,"current_position_x": current_position_x,"current_position_y": current_position_y,"goal_position_x": goal_position_x ,"goal_position_y": goal_position_y}
            y_train_row = {"time_stamp": i,"agent_id": agent, "next_position_x": next_position_x,"next_position_y": next_position_y}

            env_representation = update_env(5, 5, [path[i] for path in paths.values()], [goal_position for goal_position in goal_positions.values()])

            # append to dataset 

graph = generate_edges_dict(5, 5)

initial_positions = {0: (0, 0), 1: (1, 0)}  # Adjust initial positions as needed
goal_positions = {0: (2, 1), 1: (2, 2)}  # Adjust goal positions as needed

agents = {0: initial_positions[0], 1: initial_positions[1]}

# path1 = find_path(graph, initial_positions[0], goal_positions[0], [], [])
# print(path1)
# path2 = find_path(graph, initial_positions[1], goal_positions[1], [], [])
# print(path2)
print('Output after running CBS')
paths = cbs(graph, agents, goal_positions)
print(paths)

