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
        self.cell_count = 0
        
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
            self.unique += 1
        
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

        return rotation, movement