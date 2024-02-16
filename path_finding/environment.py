import numpy as np 

class Environment: 
    def __init__(self, width, height): 
        self.width = width 
        self.height = height 

        self.env = self.update_edges_dict(height, width, []) # y, x
        self.grid_rep = []
        self.graph_rep = {}

    # generates dictionary of edges for graph of size (x_dim, y_dim)
    def update_edges_dict(self, x_dim, y_dim, prohibited_spots):
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
                        if neighbor_pose not in prohibited_spots:
                            neighbors.append(neighbor_pose)

                edges_dict[current_pose] = neighbors

        self.graph_rep = edges_dict

    def grid_representation(self, curr_pos = (0,0), goal_pos = (1,1), taken_pos = []):
        # will be input for lstm 
        curr_x, curr_y = curr_pos
        goal_x, goal_y = goal_pos

        array = np.full((self.height, self.width), 2)
        array[curr_y, curr_x] = 1
        array[goal_y, goal_x] = -1
        for (posx, posy) in taken_pos:
            pos_x, pos_y = (posx, posy)
            array[pos_y, pos_x] = 0

        self.grid_rep = array



def main():
    env = Environment(5,5)
    print(env.grid_representation())

main()