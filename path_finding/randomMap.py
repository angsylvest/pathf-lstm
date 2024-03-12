import random
import numpy as np
import networkx as nx

np.random.seed(0)

rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
savedState = rs.get_state()


class RandomMapGenerator:
    def __init__(self, rows: int, columns: int) -> None:
        self.rows = rows
        self.cols = columns
        self.reset()

    @staticmethod
    def reset():
        rs.set_state(savedState)
    
    def get_graph(self, probability_of_obstacle=0.2):
        graph = nx.grid_2d_graph(self.rows, self.cols)
        node_pos = {n: (n[0], self.cols-n[1]) for n in graph.nodes()}

        # Delete some random nodes
        for node in graph.copy().nodes:
            if rs.random() < probability_of_obstacle:
                graph.remove_node(node)

        # Remove nodes that have no connection
        graph.remove_nodes_from(list(nx.isolates(graph)))

        for node in graph.copy().nodes:
            if len(list(graph.neighbors(node))) < 1:
                graph.remove_node(node)

        # Draw the graph
        for node in graph.nodes:
            graph.nodes[node]['pos'] = node_pos[node]
        
        grid = np.zeros((self.rows, self.cols))
        for node in graph.nodes:
            grid[node] = 1

        # nx.draw(graph, pos=node_pos, with_labels=True)
        return graph, grid.T
    
    
    def get_warehouse_like_graph(self):
        graph = nx.grid_2d_graph(self.rows, self.cols)
        node_pos = {n: (n[0], self.cols-n[1]) for n in graph.nodes()}

        # Delete random squares to make it look like a warehouse
        for node in graph.copy().nodes:
            # Every 3rd row to 5th row, every 3rd column to 5th column

            # Basically just used trial and error to find the right 
            # numbers to make it look like a warehouseh
            for i in range(3, self.rows-1, 4):
                for j in range(3, self.cols-1, 6):
                    if i <= node[0] <= i+2 and j <= node[1] <= j+2:
                        graph.remove_node(node)

        # Remove nodes that have no connection
        graph.remove_nodes_from(list(nx.isolates(graph)))

        for node in graph.copy().nodes:
            if len(list(graph.neighbors(node))) < 1:
                graph.remove_node(node)

        # For drawing
        for node in graph.nodes:
            graph.nodes[node]['pos'] = node_pos[node]
        
        grid = np.zeros((self.rows, self.cols))
        for node in graph.nodes:
            grid[node] = 1

        # nx.draw(graph, pos=node_pos, with_labels=True)
        return graph, grid.T
    
    def random_graph(self):
        while True:
            yield self.get_graph()


if __name__ == "__main__":
    env = RandomMapGenerator(10, 10)
    print(next(env.random_graph()))
    