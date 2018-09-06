import numpy as np

class Robot(object):
    def __init__(self, maze_dim):
        '''
        Use the initialization function to set up attributes that your robot
        will use to learn and navigate the maze. Some initial attributes are
        provided based on common information, including the size of the maze
        the robot is placed in.
        '''

        self.location = [0, 0]
        self.heading = 'up'
        self.maze_dim = maze_dim
        self.dir_grid = [[0 for row in range(0, self.maze_dim)] for col in range(0, self.maze_dim)]
        self.model = [[0 for row in range(0,self.maze_dim)] for col in range(0,self.maze_dim)]
        self.cell_count = 0
        self.goal_area = [[maze_dim / 2 - 1, maze_dim / 2 - 1],
                          [maze_dim / 2 - 1, maze_dim / 2],
                          [maze_dim / 2, maze_dim / 2 - 1],
                          [maze_dim / 2, maze_dim / 2]]
        self.goal_success = False
                          
    def map_cell(self, sensors):
        '''
        Records the directional values for each given cell based on the sensor
        input data.

        :param sensors: the sensor values of agent for a given cell 
            (a list of ints, i.e. [0, 0, 1])
        
        :return: NULL
        '''
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
        '''
        Counts the number of unique cells visited.

        :param: NULL
        
        :return: NULL
        '''
        x, y = self.location
        
        if self.count_grid[x][y] == 0:
            self.count_grid[x][y] = 1
            self.cell_count += 1
    
    def update_model(self, location, tune):
        '''
        Updates the values within the model.
        
        :param location: the model agent location within the map grid
            (a tuple of ints, i.e. [0, 1])
            
        :param tune: Model tuning variable
            (a int, i.e. 1)
        
        :return: NULL
        '''
        x, y = location
        self.model[x][y] = tune
        
        return self.model
        
    def make_model(self):
        '''
        Creates ML model for agent based on the first training run recorded
        sensor data and cell information

        :param: NULL
        
        :return: NULL
        '''
        
        opened = []
        tune = 1
        trans = [[-1, 0], [0, 1], [1, 0], [0, -1]]
        x, y = [self.maze_dim/2 - 1, self.maze_dim/2]

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


        
    def next_move(self, sensors):
        '''
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
        '''

        rotation = 0
        movement = 0
        
        # Record agent sensor data current location cell
        self.map_cell(sensors)
        self.breadcrumb()

        if [self.x, self.y] in self.goal_area:
            self.goal_success = True
            print('Successfully found goal. Agent at {}, {}.'.format(self.x, self.y))
            
        if self.run == 0 and self.goal_success:
            if self.cell_count >= ((self.maze_dim ** 2)) or self.moves >= 950:
                # Training run results
                print('Training Run Results:\n')
                print('{}'.format(self.dir_grid))
                    
                # Make model
                self.make_model()
                print('Training Model: \n')
                print('{}'.format(self.model))
                    
        return rotation, movement