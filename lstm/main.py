from test import *
from tqdm import tqdm


def run(auctioneer_type: str, num_agents: int, horizon: int, num_tests: int, env_shape: tuple) -> None:

    flowtimes = []
    makespans = []
    
    randomMapGenerator = RandomMapGenerator(*env_shape)
    
    
    count = 0
    pbar = tqdm(total=num_tests)
    for grid, env in randomMapGenerator.random_graph():

        if count == num_tests:
            break

        try:
            g = Graph(env)
            g.expand_in_time(horizon)
            
            agents = [Agent(g) for _ in range(num_agents)]
            auction = Auction(agents, g)
            
            auction.main_auction(auctioneer_type)
            
            flowtimes.append(auction.flowtime())
            makespans.append(auction.makespan())

        except Exception as e:
            print("Exception caught", e)
            flowtimes.append(-1)
            makespans.append(-1)
        
        count += 1
        pbar.update(1)

        
    print(f"Average flowtime: {np.mean(flowtimes)}")
    print(f"Average makespan: {np.mean(makespans)}")
    
    return flowtimes, makespans


if __name__ == "__main__":
    # run('max', 10, 30, 10, (10, 10))
    run('random', 10, 30, 10, (10, 10))