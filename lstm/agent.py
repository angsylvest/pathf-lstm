import pyray as pr
import numpy as np
import networkx as nx

from typing import Union

class Agent:
    def __init__(self, env: 'Environment') -> None:
        """
        Initializes the agent with a start and goal node.
        args:
            start: The start node of the agent.
            goal: The goal node of the agent.
        """
        self.env = env

        self.start: str = ''
        self.goal: str = ''

        self.x, self.y = (-1, -1)

        self.orientation: str = 'w'
        self.path: list[str] = []

        self.color = pr.Color(np.random.randint(0, 200), np.random.randint(0, 200), np.random.randint(0, 200), 200)

        self.reset()

    def reset(self) -> None:
        """
        Resets the agent to its initial state.
        """
        self.start = self.__unoccupied_random_node()
        while True:
            self.goal = self.__unoccupied_random_node()
            if nx.has_path(self.env.graph, self.start, self.goal):
                break

        self.x, self.y = self.env.get_cell_coordinates(self.start)

        self.orientation = "w"
        self.path = []
    
    def set_start(self, start: str) -> None:
        self.start = start
        self.x, self.y = self.env.get_cell_coordinates(self.start)
    
    def __unoccupied_random_node(self) -> str:
        """
        Returns a random unoccupied node in the environment.
        """
        while True:
            idx = np.random.choice(list(range(len(self.env.graph.nodes))))
            node = list(self.env.graph.nodes)[idx]
            if node not in self.env.occupied:
                self.env.occupied.add(node)
                return node

    def step(self) -> None:
        """
        Takes a step in the environment.
        """
        if self.path:
            self.x, self.y = self.env.get_cell_coordinates(self.path.pop(0))
    
    def draw(self) -> None:
        """
        Draws the agent on the screen.
        """
        width = pr.get_screen_width()
        height = pr.get_screen_height()
        radius = min(width // self.env.rows, height // self.env.columns) // 2

        pr.draw_circle(int(self.x * (width // self.env.rows) + radius), 
                       int(self.y * (height// self.env.columns) + radius), radius, 
                       self.color)
        
    def draw_path(self) -> None:
        """
        Draws the path of the agent on the screen.
        """
        width = pr.get_screen_width()
        height = pr.get_screen_height()
        radius = min(width // self.env.rows, height // self.env.columns) // 2

        for i, j in zip(self.path[:-1], self.path[1:]):
            x1, y1 = self.env.get_cell_coordinates(i)
            x2, y2 = self.env.get_cell_coordinates(j)
            # pr.draw_line(x1 * (width // self.env.rows) + radius, 
            #              y1 * (height// self.env.columns) + radius, 
            #              x2 * (width // self.env.rows) + radius, 
            #              y2 * (height// self.env.columns) + radius, 
            #              self.color)
            
            pr.draw_line_ex((x1 * (width // self.env.rows) + radius, 
                             y1 * (height// self.env.columns) + radius), 
                            (x2 * (width // self.env.rows) + radius, 
                             y2 * (height// self.env.columns) + radius), 
                            3, self.color)

    def find_path_in(self, graph: nx.DiGraph) -> Union[list, dict]: # list | dict:
        """
        Finds a path for the given agent.
        args:
            agent: The agent for which a path is to be found.
        """
        length = nx.shortest_path_length(self.env.graph, self.start, self.goal, weight='weight')

        for t in range(length, 5*length):
            source, sink = f"{self.start}: 0", f"{self.goal}: {t}"
            if not graph.has_node(sink):
                continue

            if nx.has_path(graph, source, sink):
                sp = nx.shortest_path(graph, source, sink, weight='weight')
                if len(sp) > 0:
                    return sp

        raise Exception("No path found.")
    
    def find_path_in_st(self, graph: nx.DiGraph, source, sink):
        if graph.has_node(source):
            while not graph.has_node(sink) or int(sink.split(':')[1]) > 40:
                sink = f"{sink.split(':')[0]}: {int(sink.split(':')[1]) + 1}"

        if nx.has_path(graph, source, sink):
            sp = nx.shortest_path(graph, source, sink, weight='weight')
            if len(sp) > 0:
                return sp
        return []
    
    
        