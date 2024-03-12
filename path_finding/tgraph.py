import numpy as np
import networkx as nx
import pyray as pr
from matplotlib import pyplot as plt


from agent import Agent
from randomMap import RandomMapGenerator


class Graph:
    def __init__(self, grid: np.ndarray) -> None:
        grid = np.rot90(grid, -1)
        self.grid = grid

        self.rows, self.columns = grid.shape
        
        self.graph = self.to_graph(grid)
        self.time_expanded_graph = None
        
        self.occupied: set = set()
        
    def reset(self):
        self.occupied = set()
        
    def topological_sort(self, t: int):
        """
        For the time expanded graph, only need to sort the nodes by their time
        Any random ordering will work for the nodes in the same time
        """
        return ((node, t) for node in self.graph.nodes)
    
    @staticmethod
    def to_graph(grid):
        rows, columns = grid.shape
        graph = nx.grid_2d_graph(rows, columns)  # 5x5 grid
        for node in graph.copy().nodes:
            if grid[node] == 0:
                graph.remove_node(node)
                
        g = nx.DiGraph(graph)
        nx.set_edge_attributes(g, {e: 1 for e in g.edges()}, name='weight')
        return g
    
    def get_cell_coordinates(self, node: str): # -> tuple[int, int]:
        """
        Returns the coordinates of the cell corresponding to the given node.
        """
        return node
    
    def expand_in_time(self, horizon : int = 10) -> None:
        """
        Updates the space-time graph of the given graph, with the given horizon.
        args:
            graph: The graph to be expanded in time.
            horizon: The horizon of the time-expanded graph.
        """
        self.time_expanded_graph = nx.DiGraph()
        max_horizon = 2 * horizon
        self.T = max_horizon

        nodes = []
        
        for t in range(max_horizon):
            for ni, n in enumerate(self.graph.nodes):
                # nn = n.split(": ")[0]
                # self.time_expanded_graph.add_node(f"{n}: {t}")
                # node_position = (self.original_graph.nodes[nn]['position'][0], t)
                node_position = (ni, t)
                # self.time_expanded_graph.nodes[f"{n}: {t}"].update({'position': node_position})
                nodes.append((f"{n}: {t}", {'position': node_position}))

        self.time_expanded_graph.add_nodes_from(nodes)
        
        edges = []
        for node in self.graph.nodes:

            for t in range(horizon):
                if t < horizon - 1:
                    # self.time_expanded_graph.add_edge(f"{node}: {t}", f"{node}: {t+1}", weight=1)
                    edges.append((f"{node}: {t}", f"{node}: {t+1}", {'weight': 1}))
                    
                
                for outEdge in self.graph.out_edges(node):
                    # w = self.graph.edges[outEdge]['weight']
                    if t + 1 <= horizon-1:      # w = 1
                        # self.time_expanded_graph.add_edge(f"{outEdge[0]}: {t}", f"{outEdge[1]}: {t+1}", weight=1)
                        edges.append((f"{outEdge[0]}: {t}", f"{outEdge[1]}: {t+1}", {'weight': 1}))

        self.time_expanded_graph.add_edges_from(edges)
        
        for node in self.time_expanded_graph.copy().nodes:
            if self.time_expanded_graph.out_degree(node) == 0 and self.time_expanded_graph.in_degree(node) == 0:
                self.time_expanded_graph.remove_node(node)

        print(self.time_expanded_graph.number_of_nodes(), self.time_expanded_graph.number_of_edges())
    
    
    def draw_grid(self):
        width = pr.get_screen_width()
        height = pr.get_screen_height()

        for i in range(0, width, width // self.rows):
            pr.draw_line(i, 0, i, width, pr.LIGHTGRAY)
        
        for i in range(0, height, height // self.columns):
            pr.draw_line(0, i, height, i, pr.LIGHTGRAY)
        
        # for node in self.graph.nodes:
        #     x, y = node
        #     pr.draw_text(str(node), x * (width // self.rows) + 10, y * (height// self.columns) + 10, 10, pr.GRAY)

        # Obstacles
        for row, row_data in enumerate(self.grid):
            for col, cell_data in enumerate(row_data):
                if cell_data == 1:
                    continue

                pr.draw_rectangle(
                    row * (width // self.rows), col * (height // self.columns), 
                    width // self.rows, height // self.columns, pr.BLACK
                )
    
    def draw_graph(self, space_time: bool = False) -> None:
        g = self.time_expanded_graph if space_time else self.graph

        if space_time:
            pos = {node: g.nodes[node]['position'] for node in g.nodes}
        else:
            pos = {node: node for node in g.nodes}

        nx.draw(g, with_labels=True, pos=pos)
            


class Tgraph(Graph):
    def __init__(self, grid: np.ndarray) -> None:
        super().__init__(grid)
        self.reset()
        
    def reset(self):
        self.occupied_nodes = set()
        self.occupied_edges = set()
        
        self.labels = {}
        self.preds = {}
        
        self.done = set()
    
    def outEdges(self, node: str, t: int): # -> list[str]:
        for outEdge in self.graph.out_edges(node):
            i, j = outEdge
            if {(i, t), (j, t+1)}.issubset(self.occupied_nodes):
                continue
            if ((i, t), (j, t+1)) in self.occupied_edges or ((j, t), (i, t+1)) in self.occupied_edges:
                continue
            yield ((i, t), (j, t+1))
            
    def reserve_node(self, node: str, t: int) -> None:
        self.occupied_nodes.add((node, t))
    
    def reserve_edge(self, i: str, j: str, t: int) -> None:
        self.occupied_edges.add(((i, t), (j, t+1)))

    # def reserve_path(self, path: list[tuple[str, int]]) -> None:
    def reserve_path(self, path: list) -> None:
        for node, t in path:
            self.reserve_node(node, t)
        
        for (i, t0), (j, t1) in zip(path[:-1], path[1:]):
            self.reserve_edge(i, j, t0)
            self.reserve_edge(j, i, t0)     # Prevents swaps

    def shortest_path(self, source, sink, time_horizon: int = 1000): #  -> list[tuple[str, int]]:
        
        self.labels[(source, 0)] = 0

        for t in range(time_horizon):

            # Use caching to avoid re-computing the same nodes
            if t in self.done: 
                continue
            self.done.add(t)
            
            for node in self.graph.nodes:
                for edgeT in self.outEdges(node, t):
                    (i, t0), (j, t1) = edgeT
                    L_it0 = self.labels.get((i, t0), float('inf'))
                    L_jt1 = self.labels.get((j, t1), float('inf'))
                    
                    if L_it0 + 1 < L_jt1:
                        self.labels[(j, t1)] = L_it0 + 1
                        self.preds[(j, t1)] = (i, t0)

            # Won't work, need to check if sink is reachable from source
            # Fix this later
            if (sink, t) in self.preds and (source, 0) in self.preds:
                break
        
        path = []
        node = sink
        while node != source:
            path.append((node, t))
            node, t = self.preds[(node, t)] 
        return path[::-1]


if __name__ == "__main__":
    env = np.array([
        [1, 1, 1],
        [0, 1, 0],
    ])
    
    g = Graph(env)
    g.expand_in_time(20)
    
    g.draw_graph()
    # g.draw_graph(space_time=True)
    # plt.show()
    a0 = Agent(g)
    a0.start = (0, 1)
    a0.goal = (2, 1)
    
    a1 = Agent(g)
    a1.start = (2, 1)
    a1.goal = (0, 1)
    
    agents = [a0, a1]

    
    WIDTH, HEIGHT = 500, 500
    pr.init_window(WIDTH, HEIGHT, "Auction")
    pause = False

    while not pr.window_should_close():
        pr.clear_background(pr.RAYWHITE)
        pr.begin_drawing()

        g.draw_grid()

        for agent in agents:
            agent.draw()
            agent.draw_path()

        pr.end_drawing()
        
    pr.close_window()   
    
    

