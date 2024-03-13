# %%
from graph import *
from tgraph import *
from agent import *
import pyray as pr
import time
import random
import cProfile


class Auction:
    # def __init__(self, agents: list[Agent], environment: Environment) -> None:
    def __init__(self, agents: list, environment: Environment) -> None:
        self.agents = agents
        self.environment = environment
        self.cell_size = self.environment.grid.shape[0]
        self.graph = self.environment.time_expanded_graph.copy()
        
        self.paths = [] 
    
    def reset(self) -> None:
        self.graph = self.environment.time_expanded_graph.copy()
        for agent in self.agents:
            agent.reset()
        self.paths = []

    # def extend_path(self, agent: Agent, path: list[str], graph: nx.DiGraph, horizon: int) -> list[str]:
    def extend_path(self, agent: Agent, path: list, graph: nx.DiGraph, horizon: int): # -> list[str]:   
        start = time.time()
        
        last_node, last_time = path[-1].split(':')
        last_time = int(last_time)
        t = last_time

        while t < horizon:
            t += 1
            if graph.has_node(f"{agent.goal}: {t}"):
                path.append(f"{agent.goal}: {t}")
            else:
                last_node, lt = path[-1].split(':')
                lt = int(lt)
                for tt in range(lt+1, horizon):
                    if graph.has_node(f"{agent.goal}: {tt}"):
                        path_to_end = agent.find_path_in_st(graph, f"{last_node}: {lt}", f"{agent.goal}: {tt}")
                        path.extend(path_to_end[1:])
                        t = tt
                        break
            if t >= self.environment.T:
                break
            
            if time.time() - start > 2:
                break
            
        return path
    
    def run(self) -> None:
        # reserve the goal nodes
        
        for agent in self.agents:
            # pos = {node: self.graph.nodes[node]['position'] for node in self.graph.nodes}
            # nx.draw(self.graph, with_labels=True, pos=pos)
            # plt.show()

            path = agent.find_path_in(self.graph)
            path = self.extend_path(agent, path, self.graph, 50)
            self.paths.append(path)
        
            agent.path = [eval(p.split(':')[0]) for p in path]
            self.reserve_path(path)

    def main_auction(self, auctioneer_type: str) -> None:
        
        agent_queue = {agent for agent in self.agents}

        while agent_queue:

            for agent in agent_queue:
                agent._temp_path = agent.find_path_in(self.graph)
                agent.bid = len(agent._temp_path)
            
            if auctioneer_type == 'max':
                winner_agent = self.bid_maximizer_auctioneer(agent_queue)
            elif auctioneer_type == 'min':
                winner_agent = self.bid_minimizer_auctioneer(agent_queue)
            elif auctioneer_type == 'random':
                winner_agent = self.bid_random_auctioneer(agent_queue)

            winner_path = winner_agent._temp_path
            winner_agent.winning_bid = len(winner_path)

            winner_path = self.extend_path(winner_agent, winner_path, self.graph, 100)

            winner_agent.path = [eval(p.split(':')[0]) for p in winner_path]
            self.reserve_path(winner_path)
            
            agent_queue.remove(winner_agent)
        
        # for agent in self.agents:
        #     for agent2 in self.agents:
        #         if agent == agent2:
        #             continue
        #         if self.collision_check(agent._temp_path, agent2._temp_path):
        #             print("Collision detected")
        #             print(agent._temp_path)
        #             print(agent2._temp_path)
        #             break
        
    @staticmethod
    # def collision_check(path1: list[str], path2: list[str]) -> bool:
    def collision_check(path1: list, path2: list) -> bool:
        for p1, p2 in zip(path1, path2):
            if p1 == p2:
                return True
        for p01, p02 in zip(path1[:-1], path1[1:]):
            for p11, p12 in zip(path2[:-1], path2[1:]):
                if p01 == p12 and p02 == p11:
                    return True
        return False
            
    # def bid_maximizer_auctioneer(self, agents: list[Agent]) -> Agent:
    def bid_maximizer_auctioneer(self, agents: list) -> Agent:
        """Maximizer auctioneer"""
        max_bid = -1
        max_bidder = None
        for agent in agents:
            if agent.bid > max_bid:
                max_bid = agent.bid
                max_bidder = agent
        return max_bidder
    
    # def bid_minimizer_auctioneer(self, agents: list[Agent]) -> Agent:
    def bid_minimizer_auctioneer(self, agents: list) -> Agent:
        """Minimizer auctioneer"""
        min_bid = float('inf')
        min_bidder = None
        for agent in agents:
            if agent.bid < min_bid:
                min_bid = agent.bid
                min_bidder = agent
        return min_bidder
    
    # def bid_random_auctioneer(self, agents: list[Agent]) -> Agent:
    def bid_random_auctioneer(self, agents: list) -> Agent:
        """Random auctioneer"""
        return random.choice(list(agents))
            
    def reserve_node(self, node: str) -> None:
        if self.graph.has_node(node):
            self.graph.remove_node(node)
            self.graph.remove_edges_from(list(self.graph.edges(node)))
    
    def reserve_edge(self, i, j) -> None:
        if self.graph.has_edge(i, j):
            self.graph.remove_edge(i, j)
            
    # def reserve_path(self, path: list[str]) -> None:
    def reserve_path(self, path: list) -> None:
        for node in path:
            self.reserve_node(node)

        # Prevent swap collisions
        for n1, n2 in zip(path[:-1], path[1:]):
            node1, t1 = n1.split(':')
            node2, t2 = n2.split(':')
            self.reserve_edge(i=f"{node1}:{t1}", j=f"{node2}:{t2}")
            self.reserve_edge(i=f"{node2}:{t1}", j=f"{node1}:{t2}")
    
    def flowtime(self) -> int:
        return sum(agent.winning_bid for agent in self.agents)

    def makespan(self) -> int:
        return max(agent.winning_bid for agent in self.agents)
            
# %%
if __name__ == "__main__":

    profiler = cProfile.Profile()
    profiler.enable()
    
    NUM_AGENTS = 20
    atype = 'min'

    # env = Environment(10, 10)
    # env.expand_in_time(30)

    randomMapGenerator = RandomMapGenerator(20, 20)
    env = next(randomMapGenerator.random_graph())[1]

    g = Graph(env)
    g.expand_in_time(40)
    
    agents = [Agent(g) for _ in range(NUM_AGENTS)]
    auction = Auction(agents, g)

    # auction.run()
    auction.main_auction(atype)
    
    profiler.disable()
    profiler.print_stats(sort='cumtime')
    
    displayer = Display(auction.agents, g)

    WIDTH, HEIGHT = 500, 500

    pr.init_window(WIDTH, HEIGHT, "Auction")

    DT = 0.05

    t = 0
    pr.set_target_fps(120)

    while not pr.window_should_close():
 
        if pr.get_key_pressed(pr.KEY_R):
            failed = True
            while failed:
                try:
                    g.reset()
                    auction.reset()
                    # auction.run()
                    auction.main_auction(atype)
                    # print(auction.paths)
                    failed = False
                except Exception as e:
                    print(e)
                    failed = True

        t += DT
        if t > 1:
            t = 0
            
        for a in agents:
            if len(a.path) == 0:
                continue
            if t == 0:
                x, y = g.get_cell_coordinates(a.path.pop(0))
                a.set_start((x, y))
            else:
                nx, ny = g.get_cell_coordinates(a.path[0])
                a.x = a.x + (nx - a.x) * t
                a.y = a.y + (ny - a.y) * t

            # Discrete movement
            # if len(a.path) > 0:
            #     x, y = g.get_cell_coordinates(a.path.pop(0))
            #     a.set_start((x, y))

        pr.clear_background(pr.RAYWHITE)
        pr.begin_drawing()

        # time.sleep(0.2)
        displayer.draw()
        pr.end_drawing()

        
    pr.close_window()

    
