import numpy as np
import random


class Robot(object):
    def __init__(self, maze_dim):
        """
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        """
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.location = [maze_dim - 1, 0]
        self.goal_area = [self.maze_dim/2 - 1, self.maze_dim/2]
        self.dir_grid = [[0 for row in range(0, self.maze_dim)] for col in range(0, self.maze_dim)]
        self.count_grid = [[0 for row in range(1, self.maze_dim + 1)] for col in range(1, self.maze_dim + 1)]
        self.action_grid = [[0 for row in range(1, self.maze_dim + 1)] for col in range(1, self.maze_dim + 1)]
        self.model = [[0 for row in range(0, self.maze_dim)] for col in range(0, self.maze_dim)]
        self.cell_count = 0
        self.action_count = 0
        self.goal_success = False
        self.training = False
        random.seed(0)

    def reset(self):
        """
        Resets the Robot class attributes to the initialized values, for most of the varibles.

        :return: NULL
        """
        self.location = [self.maze_dim - 1, 0]
        self.heading = 'up'
        print('\n\nResetting robot for Training')
        print('\nCount Grid:\n{}'.format(self.count_grid))
        print('\nDirection Grid:\n{}'.format(self.dir_grid))
        print('\nAction Grid:\n{}'.format(self.action_grid))

    def map_cell(self, sensors):
        """
        Records the directional values for each given cell based on the sensor
        input data.

        :param sensors: the sensor values of agent for a given cell 
            (a list of ints, i.e. [0, 0, 1])
        
        :return: NULL
        """
        x, y = self.location
        headings = ['left', 'up', 'right', 'down']
        directions = [8, 1, 2, 4]
        
        if self.dir_grid[x][y] == 0:
            for i in range(len(headings)):
                if self.heading == headings[i]:
                    self.dir_grid[x][y] += directions[(i + 2) % 4]
                    if sensors[0] > 0:
                        self.dir_grid[x][y] += directions[i - 1]
                    if sensors[1] > 0:
                        self.dir_grid[x][y] += directions[i]
                    if sensors[2] > 0:
                        self.dir_grid[x][y] += directions[(i + 1) % 4]
        
        self.dir_grid[self.maze_dim - 1][0] = 1
    
    def breadcrumb(self):
        """
        Counts the number of unique cells visited.

        :param: NULL
        
        :return: NULL
        """
        x, y = self.location
        
        if self.count_grid[x][y] == 0:
            self.count_grid[x][y] = 1
            self.cell_count += 1
    
    def update_model(self, location, tune):
        """
        Updates the values within the model.
        
        :param location: the model agent location within the map grid
            (a tuple of ints, i.e. [0, 1])
            
        :param tune: Model tuning variable
            (a int, i.e. 1)
        
        :return: NULL
        """
        x, y = location
        self.model[x][y] = tune

    def act_legal(self, location):
        """
        Determines what actions are legal and returns the value
        
        :param location: Location of agent model
            (a tuple of ints, i.e. (0, 1))
        
        :return actions: NULL
            (a list of strings, i.e. ['left', 'up'])
        """
        x, y = location
        actions = []
        
        vals = [[1, 3, 5, 7, 9, 11, 13, 15],
                [2, 3, 6, 7, 10, 11, 14, 15],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [8, 9, 10, 11, 12, 13, 14, 15]]
                
        possible = ['up', 'right', 'down', 'left']
        
        for i in range(len(vals)):
            if self.dir_grid[x][y] in vals[0]:
                actions.extend([possible[0]])
            if self.dir_grid[x][y] in vals[1]:
                actions.extend([possible[1]])
            if self.dir_grid[x][y] in vals[2]:
                actions.extend([possible[2]])
            if self.dir_grid[x][y] in vals[3]:
                actions.extend([possible[3]])
            
            return actions
            
    def make_model(self):
        """
        Creates ML model for agent based on the first training run recorded
        sensor data and cell information

        :param: NULL
        
        :return: NULL
        """
        opened = []
        tune = 1
        trans = [[-1, 0], [0, 1], [1, 0], [0, -1]]

        x = self.goal_area[0]
        y = self.goal_area[1]

        # Store adjacent directional options based on current cell location
        opened.append(((x, y), tune))
        opened.append(((x + trans[2][0],    y + trans[2][1]), tune))
        opened.append(((x + trans[3][0],    y + trans[3][1]), tune))
        opened.append(((x + trans[2][0],    y + trans[3][1]), tune))
        
        # Update agent's model's cell values
        for cell in opened:
            self.update_model(cell[0], cell[1])

        # Check all valid possible directions for model agent
        while self.model[self.maze_dim -1][0] == 0 and len(opened) != 0:
            location, tune = opened.pop(0)
            actions = self.act_legal(location)

            if 'up' in actions and self.count_grid[location[0]][location[1]] != 0:
                x2 = location[0] + trans[0][0]
                y2 = location[1] + trans[0][1]
                new_location = x2, y2
                if self.model[x2][y2] == 0 and self.count_grid[location[0]][location[1]] != 0:
                    opened.append((new_location, tune + 1))
                    self.update_model(new_location, tune + 1)
            
            if 'right' in actions:
                x2 = location[0] + trans[1][0]
                y2 = location[1] + trans[1][1]
                new_location = x2, y2
                if self.model[x2][y2] == 0 and self.count_grid[location[0]][location[1]] != 0:
                    opened.append((new_location, tune + 1))
                    self.update_model(new_location, tune + 1)
            
            if 'down' in actions:
                x2 = location[0] + trans[2][0]
                y2 = location[1] + trans[2][1]
                new_location = x2, y2
                if self.model[x2][y2] == 0 and self.count_grid[location[0]][location[1]] != 0:
                    opened.append((new_location, tune + 1))
                    self.update_model(new_location, tune + 1)
        
            if 'left' in actions:
                x2 = location[0] + trans[3][0]
                y2 = location[1] + trans[3][1]
                new_location = x2, y2
                if self.model[x2][y2] == 0:
                    opened.append((new_location, tune + 1))
                    self.update_model(new_location, tune + 1)
    
    def make_action_grid(self):
        """
        Determines and stores the best action for the agent for each cell.
        
        :param: NULL
        
        :return: NULL
        """
        vals = [[1, 3, 5, 7, 9, 11, 13, 15],
                [2, 3, 6, 7, 10, 11, 14, 15],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [8, 9, 10, 11, 12, 13, 14, 15]]
                
        possible = ['up', 'right', 'down', 'left']
        
        trans = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        
        for i in range(len(self.model)):
            for j in range(len(self.model[0])):
                for k in range(len(vals)):
                    if self.dir_grid[i][j] in vals[k]:
                        if self.model[i + trans[k][0]][j + trans[k][1]] == self.model[i][j] - 1:
                            self.action_grid[i][j] = possible[k]
        
    def next_move(self, sensors):
        """
        Use this function to determine the next move the robot should make,
        based on the input from the sensors after its previous move. Sensor
        inputs are a list of three distances from the robot's left, front, and
        right-facing sensors, in that order.

        Outputs should be a tuple of two values. The first value indicates
        robot rotation (if any), as a number: 0 for no rotation, +90 for a
        90-degree rotation clockwise, and -90 for a 90-degree rotation
        counterclockwise. Other values will result in no rotation. The second
        value indicates robot movement, and the robot will attempt to move the
        number of indicated squares: a positive number indicates forwards
        movement, while a negative number indicates backwards movement. The
        robot may move a maximum of three units per turn. Any excess movement
        is ignored.

        If the robot wants to end a run (e.g. during the first training run in
        the maze) then returing the tuple ('Reset', 'Reset') will indicate to
        the tester to end the run and return the robot to the start.
        """

        rotation = 0
        movement = 0
        
        # Record agent sensor data current location cell
        x, y = self.location
        self.map_cell(sensors)
        self.breadcrumb()

        if [x, y] == self.goal_area:
            self.goal_success = True
            print('Successfully found goal. Agent at {}, {}.'.format(x, y))

        if not self.training and self.goal_success:
            print('\ntest')
            print(self.cell_count)
            print(self.maze_dim)
            print(self.action_count)
            print((self.cell_count >= (self.maze_dim ** 2)))
            print('\n')
            if (self.cell_count >= (self.maze_dim ** 2)) or (self.action_count >= 300):
                # Training run results
                print('\nTraining Run Results:\n')
                print('{}'.format(self.dir_grid))
                    
                # Make model
                self.make_model()
                print('\nTraining Model:\n')
                print('{}'.format(self.model))
                
                self.make_action_grid()
                self.reset()
                self.training = True
                return 'Reset', 'Reset'

        return rotation, movement
