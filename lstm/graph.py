# %%
from agent import Agent
import numpy as np
import networkx as nx
import pyray as pr
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt



class Environment:
    def __init__(self, rows: int, columns: int): # -> None:
        """
        Initializes the environment with a grid graph with the given number of rows and columns.
        args:
            rows: The number of rows in the grid graph.
            columns: The number of columns in the grid graph.
        """
        self.grid: np.ndarray = np.zeros((rows, columns))
        self.rows: int = rows
        self.columns: int = columns

        self.occupied: set = set()

        self.original_graph: nx.DiGraph = self.__get_grid_graph(rows, columns)
        self.space_time_graph: nx.Digraph | None = None


    def get_cell_coordinates(self, node: str): # -> tuple[int, int]:
        """
        Returns the coordinates of the cell corresponding to the given node.
        args:
            node: The node whose cell coordinates are to be returned.
        """
        return self.original_graph.nodes[node]['position']

    def __get_grid_graph(self, rows: int, columns: int): # -> nx.DiGraph:
        """
        Returns a grid graph with the given number of rows and columns.
        args:
            rows: The number of rows in the grid graph.
            columns: The number of columns in the grid graph.
        """

        node_positions = {f"n{i}": (i % rows, i // columns) for i in range(rows * columns)}

        grid = nx.DiGraph()

        for i in range(rows * columns):

            if i % columns != columns - 1:
                grid.add_edge(f"n{i}", f"n{i+1}", weight=1)
                grid.add_edge(f"n{i+1}", f"n{i}", weight=1)

            if i // columns != rows - 1:
                grid.add_edge(f"n{i}", f"n{i+columns}", weight=1)
                grid.add_edge(f"n{i+columns}", f"n{i}", weight=1)

        # Create obstacles
        for node in grid.copy().nodes:
            if np.random.random() < 0.2:
                grid.remove_node(node)
                self.occupied.add(node)
            else:
                grid.nodes[node].update({'position': node_positions[node]})

        # Remove isolated nodes
        for node in grid.copy().nodes:
            if grid.out_degree(node) == 0 and grid.in_degree(node) == 0:
                grid.remove_node(node)
        
        # Update grid
        # print(grid.nodes)
        for n, node in node_positions.items():
            # print(node, grid.has_node(n))
            if grid.has_node(n):
                self.grid[node[0], node[1]] = 1

        return grid

    def expand_in_time(self, horizon : int = 10): # -> None:
        """
        Updates the space-time graph of the given graph, with the given horizon.
        args:
            graph: The graph to be expanded in time.
            horizon: The horizon of the time-expanded graph.
        """
        self.space_time_graph = nx.DiGraph()
        max_horizon = 2 * horizon

        for node in self.original_graph.nodes:

            for t in range(horizon):
                if t < horizon - 1:
                    self.space_time_graph.add_edge(f"{node}: {t}", f"{node}: {t+1}", weight=1)

                for outEdge in self.original_graph.out_edges(node):
                    w = self.original_graph.edges[outEdge]['weight']
                    if t + w > horizon-1:
                        continue
                    self.space_time_graph.add_edge(f"{outEdge[0]}: {t}", f"{outEdge[1]}: {t+w}", weight=w)

            for t in range(max_horizon):
                for ni, n in enumerate(self.original_graph.nodes):
                    nn = n.split(": ")[0]
                    self.space_time_graph.add_node(f"{n}: {t}")
                    # node_position = (self.original_graph.nodes[nn]['position'][0], t)
                    node_position = (ni, t)
                    self.space_time_graph.nodes[f"{n}: {t}"].update({'position': node_position})
        
        for node in self.space_time_graph.copy().nodes:
            if self.space_time_graph.out_degree(node) == 0 and self.space_time_graph.in_degree(node) == 0:
                self.space_time_graph.remove_node(node)

    def draw_graph(self, space_time: bool = False, file_path: str = None) -> None:
        g = self.space_time_graph if space_time else self.original_graph
        pos = {node: g.nodes[node]['position'] for node in g.nodes}
        nx.draw(g, with_labels=True, pos=pos)

        if file_path:
            plt.savefig(file_path)
        else:
            plt.show()

    def draw_grid(self):
        width = pr.get_screen_width()
        height = pr.get_screen_height()

        for i in range(0, width, width//self.rows):
            pr.draw_line(i, 0, i, width, pr.LIGHTGRAY)

        for j in range(0, height, height//self.columns):
            pr.draw_line(0, j, height, j, pr.LIGHTGRAY)
            
        for node in self.original_graph.nodes:
            x, y = self.original_graph.nodes[node]['position']
            pr.draw_text(str(node), x * (width // self.rows), y * (height // self.columns), 18, pr.GRAY)

    def draw_obstacles(self):
        width = pr.get_screen_width()
        height = pr.get_screen_height()

        for row, row_data in enumerate(self.grid):
            for col, cell_data in enumerate(row_data):
                if cell_data == 1:
                    continue

                pr.draw_rectangle(
                    row * (width // self.rows), col * (height // self.columns), 
                    width // self.rows, height // self.columns, pr.BLACK
                )

    def find_path(self, agent: Agent): # -> list | dict:
        """
        Finds a path for the given agent.
        args:
            agent: The agent for which a path is to be found.
        """
        length = nx.shortest_path_length(self.original_graph, agent.start, agent.goal, weight='weight')
        self.expand_in_time(length*2)

        for t in range(length, 2*length):
            source, sink = f"{agent.start}: 0", f"{agent.goal}: {t}"
            sp = nx.shortest_path(self.space_time_graph, source, sink, weight='weight')
            if len(sp) > 0:
                return sp

        return []

    def find_path_in(self, graph: nx.DiGraph, agent: Agent): # -> list | dict:
        """
        Finds a path for the given agent.
        args:
            agent: The agent for which a path is to be found.
        """
        length = nx.shortest_path_length(self.original_graph, agent.start, agent.goal, weight='weight')

        for t in range(length, 3*length):
            source, sink = f"{agent.start}: 0", f"{agent.goal}: {t}"
            if not nx.has_path(graph, source, sink):
                continue
            sp = nx.shortest_path(graph, source, sink, weight='weight')
            if len(sp) > 0:
                # last_node, last_time = sp[-1].split(': ')
                # last_time = int(last_time)
                # for tt in range(last_time, last_time + 120):
                #     sp.append(f"{last_node}: {tt}")
                return sp

        return []



class Display:
    # def __init__(self, agents: list[Agent], environment: Environment) -> None:
    #     self.agents = agents
    #     self.environment = environment
    #     self.cell_size = self.environment.grid.shape[0]
    def __init__(self, agents: list, environment: Environment) -> None:
        self.agents = agents
        self.environment = environment
        self.cell_size = self.environment.grid.shape[0]

    def draw(self) -> None:
        self.environment.draw_grid()
        # self.environment.draw_obstacles()
        for agent in self.agents:
            agent.draw()
            agent.draw_path()


# %%
if __name__ == "__main__":
    
    graph, grid = get_graph(5, 5)
    
    env = Environment(3, 3)
    env.expand_in_time(10)
    env.draw_graph(space_time=False) 
    plt.show()
    env.draw_graph(space_time=True)
    plt.show()
