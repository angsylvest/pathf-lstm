import heapq
from typing import List, Tuple, Dict, Union
import pandas as pd 
import csv 
import numpy as np 

from environment import Environment

# we are assuming that csv already has correct csv label 
dataset_x_path = "dataset/x_time_train_test.csv"
dataset_y_path = "dataset/y_train_test.csv"

# dictionary mapping actions to movements
action_to_movement = {0: (1, 0), 1: (-1, 0), 2: (0, 1), 3: (0, -1)}

# update x_time_train.csv
with open(dataset_x_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['time_stamp', 'env_input'])
    if file.tell() == 0:
        writer.writeheader()  # Write the header row if the file is empty

# update y_train.csv
with open(dataset_y_path, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['time_stamp','action'])
    if file.tell() == 0:
        writer.writeheader()  # Write the header row if the file is empty

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


class CBS: 
    def __init__(self, starting_poses, goal_poses, env_size):
        self.starting_poses = starting_poses
        self.goal_poses = goal_poses
        self.env_size = env_size 

        self.environments = {}  # TODO: each agent should have own env with pos of other agents
        for i in range(len(goal_poses)):

            # creates separate env (no consideration of other agents yet)
            self.environments[i] = Environment(self.env_size, self.env_size)
            self.environments[i].grid_representation(starting_poses[i], goal_poses[i])

            # updated graph rep of env (initially no prohibitee nodes)
            self.environments[i].update_edges_dict(self.env_size, self.env_size, [])


    def is_conflict_free(self, paths):
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

    def find_path(self, graph, start, goal, occupied_positions, other_agents_goals) -> List[Tuple[int, int]]:
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


    def cbs(self) -> Union[Dict[str, List[Tuple[int, int]]], None]:
        graph = self.environments[0].graph_rep
        agents = self.starting_poses
        goal_positions = self.goal_poses
        
        open_set = [Node({}, 0)]  # Start with an empty set of paths
        closed_set = set()

        paths = {}
        conflict_positions = {}

        # Initially run A* for each agent (no consideration of others)
        for agent, goal in agents.items():
            initial_position = agents[agent]
            path = self.find_path(graph, initial_position, goal_positions[agent], [], [])
            paths[agent] = path
            conflict_positions[agent] = []

        # print(f"All paths: {paths}")

        problem_pos, conflict_free = self.is_conflict_free(paths)

        while not conflict_free and any(len(problem_pos[agent]) > 0 for agent in agents):  # While there is a conflict
            # Identify conflict nodes + remove and recalculate solution
            for agent, initial_pose in agents.items():
                problem_pos, conflict_free = self.is_conflict_free(paths)

                if len(problem_pos[agent]) == 0:
                    break

                # Remove one node that is causing conflict
                agent_path = paths[agent]

                selected_pos = problem_pos[agent].pop()
                conflict_positions[agent].append(selected_pos)

                # print(f'path arguments initial pose: {initial_pose} with goal position {goal_positions[agent]} and conflict positions {conflict_positions[agent]} in graph {graph}')
                path = self.find_path(graph, initial_pose, goal_positions[agent], conflict_positions[agent], [])
                # print(f"Updated generated path for agent {agent} with initial pos {initial_pose} with goal {goal_positions[agent]} and occupied pos {conflict_positions[agent]}\nPath: {path}")

                paths[agent] = path

                # Recalculate conflict-free status after updating paths
                problem_pos, conflict_free = self.is_conflict_free(paths)

        return paths if conflict_free else None
    
    def generate_seq_dataset(self, cbs_output):

        max_length = max(len(path) for path in cbs_output.values())
        curr_index = 0 
        goal_poses = [pos[-1] for pos in cbs_output.values()]

        while curr_index < max_length:

            
            for a in cbs_output: 
                other_poses = []

                # for each agent, will update env and subsequent pos 
                if curr_index <= len(cbs_output[a]) - 1:
                    curr_agent_pos = cbs_output[a][curr_index]
                else: 
                    curr_agent_pos = cbs_output[a][-1]
                cur_agent_goal = self.goal_poses[a]

                for a_other in cbs_output: 
                    if a != a_other:
                        if curr_index <= len(cbs_output[a_other]) - 1: 
                            other_agent_pos = cbs_output[a_other][curr_index]
                        else: 
                            other_agent_pos = cbs_output[a_other][-1]
                        other_poses.append(other_agent_pos)
                print(f'updated other poses: {other_poses}')

                # update input rep  
                self.environments[a].grid_representation(curr_agent_pos, cur_agent_goal, other_poses, goal_poses)
                input_rep = self.environments[a].grid_rep
                input_rep = np.array2string(input_rep, separator=',').replace('\n', '').replace('  ', ' ')

                if curr_index <= len(cbs_output[a]) - 2: 
                    next = curr_index + 1
                else: 
                    next = len(cbs_output[a]) - 2 # just stay at current spot once reach goal 
                
                action = self.calculate_action(curr_agent_pos, cbs_output[a][next])
                # expected_outx, expected_outy = cbs_output[a][next]

                x_row = {'time_stamp': curr_index, 'env_input': input_rep}
                y_row = {'time_stamp': curr_index, 'action': action}

                # update x_time_train.csv
                with open(dataset_x_path, mode='a+', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['time_stamp', 'env_input'])
                    writer.writerow(x_row)

                # update y_train.csv
                with open(dataset_y_path, mode='a+', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=['time_stamp','action'])
                    writer.writerow(y_row)


            curr_index += 1

    def calculate_action(self, curr_pos, next_pos):
        # Assuming movements are limited to up, down, left, and right
        diff = tuple(map(lambda x, y: y - x, curr_pos, next_pos))
        action_to_movement = {(0, 1): 0, (0, -1): 1, (1, 0): 2, (-1, 0): 3}
        return action_to_movement.get(diff, -1)  # Default to -1 if no valid action found


def main():
    starting_pose = {0: (0, 0), 1: (1, 0), 2: (1,3)} 
    goal_pos = {0: (2, 1), 1: (2, 2), 2: (3,3)}
    cbs = CBS(starting_pose, goal_pos, 5)
    paths = cbs.cbs()
    print(f'paths: {paths}')
    cbs.generate_seq_dataset(paths)

    # now want to translate to multiple diff environments that can be saved 

main()
    